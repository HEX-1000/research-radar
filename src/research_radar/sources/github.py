"""GitHub: new repositories through the search API, plus star counts and READMEs.

GitHub has no official trending API, so "new" means repositories created since
the last scan, ordered by stars. Without a token the search API allows about 10
requests a minute, with a token about 30, so queries are spaced out.
"""

from __future__ import annotations

import re

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
    license_info = raw.get("license") or {}
    spdx = license_info.get("spdx_id")
    if spdx in (None, "NOASSERTION"):
        spdx = None
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
        },
    )


def fetch(ctx: FetchContext) -> list[Item]:
    opts = ctx.config.sources[NAME]
    min_stars = int(opts.get("min_stars", 5))
    per_query = int(opts.get("per_query", 10))
    interval = 2.2 if ctx.http.github_token else 6.5
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


def star_counts(http: Http, full_names: list[str], limit: int = 15) -> dict[str, int]:
    """Current star counts for repositories we reported earlier (for 'rising')."""
    counts: dict[str, int] = {}
    for full_name in full_names[:limit]:
        try:
            data = http.get_json(f"{API}/repos/{full_name}", headers=HEADERS)
        except SourceError:
            break
        stars = data.get("stargazers_count") if isinstance(data, dict) else None
        if isinstance(stars, int):
            counts[full_name] = stars
    return counts


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
