"""Forgejo and Gitea servers, Codeberg included.

Codeberg runs Forgejo, and Forgejo and Gitea expose the same /api/v1 REST API,
so one module reads any of them. List the servers under `sources.forgejo.instances`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..http import SourceError
from ..models import Item
from .base import FetchContext, clean_text, settle

NAME = "forgejo"


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_repos(data: object, instance: str, since: datetime) -> list[Item]:
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise SourceError(f"{instance}: unexpected search response")
    items: list[Item] = []
    for raw in rows:
        if not isinstance(raw, dict) or raw.get("fork") or raw.get("archived"):
            continue
        full_name = raw.get("full_name")
        url = raw.get("html_url")
        if not full_name or not url:
            continue
        created = _parse_time(raw.get("created_at"))
        if created is not None:
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created < since:
                continue
        topics = raw.get("topics") or []
        description = clean_text(raw.get("description"))
        text = f"{description} Topics: {', '.join(topics)}".strip() if topics else description
        items.append(
            Item(
                source=NAME,
                kind="repo",
                title=f"{instance}/{full_name}",
                url=url,
                text=text,
                published=raw.get("created_at"),
                meta={
                    "stars": raw.get("stars_count"),
                    "language": raw.get("language"),
                    "license": None,
                    "instance": instance,
                },
            )
        )
    return items


def search_url(instance: str) -> str:
    return f"https://{instance}/api/v1/repos/search"


def auth_headers(token: str | None) -> dict[str, str] | None:
    # Sent only with requests to the server the token belongs to. httpx drops the header on a
    # redirect to another host.
    return {"Authorization": f"token {token}"} if token else None


def token_hint(env: str, token: str | None) -> str:
    """What to do about a refusal when no token was sent."""
    if token:
        return ""
    if env:
        return f". Searches with an access token still work: put one in the environment variable {env} (see the README)"
    return ". Searches with an access token may still work: name its environment variable under sources.forgejo.token_env"


def fetch(ctx: FetchContext) -> list[Item]:
    opts = ctx.config.sources[NAME]
    instances = [str(i).strip().removeprefix("https://").strip("/") for i in opts.get("instances", ["codeberg.org"])]
    per_query = int(opts.get("per_query", 10))
    items: dict[str, Item] = {}
    errors: list[str] = []
    ok = 0
    for instance in instances:
        env, token = ctx.config.forgejo_token(instance)
        terms = [str(t) for topic in ctx.config.topics for t in (topic.source_option(NAME, "queries") or topic.keywords[:2])]
        for term in terms:
            params = {"q": term, "sort": "created", "order": "desc", "limit": per_query}
            try:
                data = ctx.http.get_json(
                    search_url(instance),
                    params=params,
                    headers=auth_headers(token),
                    bucket=instance,
                    min_interval=1.0,
                )
                for item in parse_repos(data, instance, ctx.since):
                    items.setdefault(item.url, item)
                ok += 1
            except SourceError as exc:
                # The other queries would fail the same way, so they are not sent; the next scan collects the gap.
                if exc.status == 429:
                    errors.append(f"{instance} refuses searches for now, the scan skipped it: {exc}{token_hint(env, token)}")
                    break
                if exc.status == 401 and token:
                    errors.append(f"{instance} did not accept the access token in {env}; create a new one ({exc})")
                    break
                errors.append(f"{instance} '{term}': {exc}")
    settle(ctx, NAME, ok, errors)
    return list(items.values())
