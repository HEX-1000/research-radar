"""GitHub: new repositories through the search API, plus star counts and READMEs.

GitHub has no official trending API, so "new" means repositories created since
the last scan, ordered by stars. Without a token the search API allows about 10
requests a minute, with a token about 30, so queries are spaced out. GitHub also
refuses bursts with a secondary limit, so the spacing is wider than the minimum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..http import Http, SourceError
from ..models import Item
from .base import FetchContext, clean_text, settle

NAME = "github"
API = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


def _quote(term: str) -> str:
    return f'"{term}"' if " " in term else term


def build_query(term: str, *, is_topic: bool, since_date: str, min_stars: int) -> str:
    base = f"topic:{term}" if is_topic else _quote(term)
    return f"{base} created:>={since_date} stars:>={min_stars} fork:false archived:false"


def _license(raw: dict) -> str | None:
    spdx = (raw.get("license") or {}).get("spdx_id")
    return None if spdx in (None, "NOASSERTION") else spdx


def parse_repo(raw: dict) -> Item | None:
    full_name = raw.get("full_name")
    url = raw.get("html_url")
    if not full_name or not url:
        return None
    topics = raw.get("topics") or []
    description = clean_text(raw.get("description"))
    text = description
    if topics:
        text = f"{description} Topics: {', '.join(topics)}".strip()
    spdx = _license(raw)
    return Item(
        source=NAME,
        kind="repo",
        title=full_name,
        url=url,
        text=text,
        published=raw.get("created_at"),
        meta={
            "stars": raw.get("stargazers_count"),
            "language": raw.get("language"),
            "license": spdx,
            "topics": topics,
            "pushed_at": raw.get("pushed_at"),
            "full_name": full_name,
            "size_kb": raw["size"] if isinstance(raw.get("size"), int) else None,
            "forks": raw.get("forks_count"),
            "owner": (raw.get("owner") or {}).get("login"),
        },
    )


def fetch(ctx: FetchContext) -> list[Item]:
    opts = ctx.config.sources[NAME]
    min_stars = int(opts.get("min_stars", 5))
    per_query = int(opts.get("per_query", 10))
    interval = 3.0 if ctx.http.github_token else 6.5
    since_date = ctx.since.date().isoformat()

    items: dict[str, Item] = {}
    errors: list[str] = []
    ok = 0
    stop = False
    for topic in ctx.config.topics:
        if stop:
            break
        queries = topic.source_option(NAME, "queries") or topic.keywords[:3]
        topics = topic.source_option(NAME, "topics") or []
        specs = [(str(q), False) for q in queries] + [(str(t), True) for t in topics]
        for term, is_topic in specs:
            params = {
                "q": build_query(term, is_topic=is_topic, since_date=since_date, min_stars=min_stars),
                "sort": "stars",
                "order": "desc",
                "per_page": per_query,
            }
            try:
                data = ctx.http.get_json(
                    f"{API}/search/repositories",
                    params=params,
                    headers=HEADERS,
                    bucket="github-search",
                    min_interval=interval,
                )
                ok += 1
            except SourceError as exc:
                errors.append(f"query '{term}': {exc}")
                if "rate limit" in str(exc):
                    stop = True
                    break
                continue
            for raw in data.get("items", []) if isinstance(data, dict) else []:
                item = parse_repo(raw)
                if item and item.url not in items:
                    items[item.url] = item
    settle(ctx, NAME, ok, errors)
    return list(items.values())


GONE = (404, 410, 451)  # deleted, or taken down for legal reasons

# A repository is "thin" when it is small, already has stars and comes from an account nobody
# follows: the mark of a lure that bought its stars. The numbers come from the scan results of
# 2026-11-13 (see DECISIONS-2026-11-13.md); they are constants, not settings.
THIN_MAX_KB = 200
THIN_MIN_STARS = 10
THIN_MAX_FOLLOWERS = 2
HOLD_DAYS = 7


def looks_thin(size_kb: int | None, stars: int | None) -> bool:
    """The two conditions that need no extra request. An unknown size is not thin."""
    return (
        isinstance(size_kb, int)
        and isinstance(stars, int)
        and size_kb < THIN_MAX_KB
        and stars >= THIN_MIN_STARS
    )


def owner_followers(http: Http, login: str | None, cache: dict[str, int | None]) -> int | None:
    """Followers of a GitHub account, or None when GitHub could not say. Asked once per scan."""
    if not login:
        return None
    key = login.lower()
    if key not in cache:
        try:
            data = http.get_json(f"{API}/users/{login}", headers=HEADERS)
            count = data.get("followers") if isinstance(data, dict) else None
            cache[key] = count if isinstance(count, int) else None
        except SourceError:
            cache[key] = None
    return cache[key]


def thin_reason(
    http: Http, size_kb: int | None, stars: int | None, owner: str | None, cache: dict[str, int | None]
) -> str | None:
    """Why a repository is held back, or None. A failed request about the owner counts as thin."""
    if not looks_thin(size_kb, stars):
        return None
    followers = owner_followers(http, owner, cache)
    if followers is not None and followers > THIN_MAX_FOLLOWERS:
        return None
    who = "owner with unknown followers" if followers is None else (
        f"owner with {followers} follower" + ("" if followers == 1 else "s")
    )
    return f"{size_kb} KB, {stars} stars, {who}"


def fetch_repo(http: Http, full_name: str) -> dict | None:
    """The raw repository, for a recheck. SourceError passes through (the caller reads .status)."""
    data = http.get_json(f"{API}/repos/{full_name}", headers=HEADERS)
    return data if isinstance(data, dict) else None


@dataclass
class RepoCheck:
    """What GitHub says today about repositories reported earlier (for 'rising')."""

    current: dict[str, dict] = field(default_factory=dict)  # full name: stars, license, language
    gone: list[str] = field(default_factory=list)


def check_repos(http: Http, full_names: list[str]) -> RepoCheck:
    """Ask about every repository. A deleted one is noted and the check goes on; a rate limit, a
    network failure or a server error ends it, and the rest are asked at the next scan."""
    check = RepoCheck()
    for full_name in full_names:
        try:
            data = http.get_json(f"{API}/repos/{full_name}", headers=HEADERS)
        except SourceError as exc:
            if exc.status in GONE:
                check.gone.append(full_name)
                continue
            if exc.status == 403:
                continue  # this repository is blocked; the others may not be
            break
        stars = data.get("stargazers_count") if isinstance(data, dict) else None
        if isinstance(stars, int):
            check.current[full_name] = {"stars": stars, "license": _license(data), "language": data.get("language")}
    return check


_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML = re.compile(r"<[^>]+>")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def fetch_readme(http: Http, full_name: str, max_chars: int = 3000) -> str:
    """The start of a README as plain text, with badges and markup stripped."""
    try:
        response = http.get(
            f"{API}/repos/{full_name}/readme",
            headers={"Accept": "application/vnd.github.raw+json", "X-GitHub-Api-Version": "2022-11-28"},
        )
    except SourceError:
        return ""
    text = response.text
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _HTML.sub(" ", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return clean_text(text)[:max_chars]
