"""Hugging Face: the daily papers list and trending models, from the public API."""

from __future__ import annotations

from datetime import timedelta

from ..http import SourceError
from ..models import Item
from .base import FetchContext, clean_text, settle

PAPERS = "huggingface-papers"
MODELS = "huggingface-models"
API = "https://huggingface.co/api"
MAX_DAYS = 14


def parse_papers(data: object) -> list[Item]:
    if not isinstance(data, list):
        raise SourceError("huggingface: daily papers response is not a list")
    items: list[Item] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        paper = entry.get("paper") if isinstance(entry.get("paper"), dict) else entry
        paper_id = paper.get("id")
        title = clean_text(paper.get("title") or entry.get("title"))
        if not paper_id or not title:
            continue
        items.append(
            Item(
                source=PAPERS,
                kind="paper",
                title=title,
                url=f"https://huggingface.co/papers/{paper_id}",
                text=clean_text(paper.get("summary") or entry.get("summary")),
                published=paper.get("publishedAt") or entry.get("publishedAt"),
                meta={
                    "arxiv_id": paper_id,
                    "upvotes": paper.get("upvotes"),
                    "github_repo": paper.get("githubRepo"),
                },
            )
        )
    return items


def fetch_papers(ctx: FetchContext) -> list[Item]:
    day = ctx.since.date()
    last = ctx.until.date()
    if (last - day).days > MAX_DAYS:
        day = last - timedelta(days=MAX_DAYS)
    items: dict[str, Item] = {}
    errors: list[str] = []
    ok = 0
    while day <= last:
        try:
            data = ctx.http.get_json(
                f"{API}/daily_papers",
                params={"date": day.isoformat()},
                bucket="huggingface",
                min_interval=0.5,
            )
            for item in parse_papers(data):
                items.setdefault(item.url, item)
            ok += 1
        except SourceError as exc:
            errors.append(f"{day.isoformat()}: {exc}")
        day += timedelta(days=1)
    settle(ctx, PAPERS, ok, errors)
    return list(items.values())


_SKIP_TAG_PREFIXES = ("region:", "license:", "arxiv:", "endpoints_compatible", "autotrain_compatible")


def parse_models(data: object) -> list[Item]:
    if not isinstance(data, list):
        raise SourceError("huggingface: models response is not a list")
    items: list[Item] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        model_id = raw.get("id") or raw.get("modelId")
        if not model_id:
            continue
        tags = [t for t in (raw.get("tags") or []) if not str(t).startswith(_SKIP_TAG_PREFIXES)]
        task = raw.get("pipeline_tag") or ""
        text = clean_text(f"{task} model. Tags: {', '.join(map(str, tags[:15]))}")
        items.append(
            Item(
                source=MODELS,
                kind="model",
                title=str(model_id),
                url=f"https://huggingface.co/{model_id}",
                text=text,
                published=raw.get("createdAt"),
                meta={
                    "likes": raw.get("likes"),
                    "downloads": raw.get("downloads"),
                    "task": task or None,
                },
            )
        )
    return items


def fetch_models(ctx: FetchContext) -> list[Item]:
    limit = int(ctx.config.sources[MODELS].get("limit", 25))
    errors: list[str] = []
    items: list[Item] = []
    ok = 0
    try:
        data = ctx.http.get_json(
            f"{API}/models",
            params={"sort": "trendingScore", "direction": "-1", "limit": limit},
            bucket="huggingface",
            min_interval=0.5,
        )
        items = parse_models(data)
        ok = 1
    except SourceError as exc:
        errors.append(str(exc))
    settle(ctx, MODELS, ok, errors)
    return items
