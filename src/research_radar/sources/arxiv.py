"""arXiv: the official Atom API, one query per topic.

arXiv asks API clients to wait about three seconds between requests, and it
answers with HTTP 429 when they do not. This module waits, and the shared
client retries once or twice before giving up.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime

from ..http import SourceError
from ..models import Item
from .base import FetchContext, clean_text, settle

NAME = "arxiv"
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%d%H%M")


def build_query(categories: list[str], terms: list[str], since: datetime, until: datetime) -> str:
    parts: list[str] = []
    if categories:
        parts.append("(" + " OR ".join(f"cat:{c}" for c in categories) + ")")
    if terms:
        parts.append("(" + " OR ".join(f'all:"{t}"' for t in terms) + ")")
    parts.append(f"submittedDate:[{_stamp(since)} TO {_stamp(until)}]")
    return " AND ".join(parts)


def parse_feed(xml_text: str) -> list[Item]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SourceError(f"arxiv: response is not valid Atom XML ({exc})") from exc
    items: list[Item] = []
    for entry in root.findall("a:entry", NS):
        raw_id = (entry.findtext("a:id", default="", namespaces=NS) or "").strip()
        title = clean_text(entry.findtext("a:title", default="", namespaces=NS))
        if not raw_id or not title:
            continue
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
        authors = [
            clean_text(a.findtext("a:name", default="", namespaces=NS))
            for a in entry.findall("a:author", NS)
        ]
        categories = [c.get("term") for c in entry.findall("a:category", NS) if c.get("term")]
        items.append(
            Item(
                source=NAME,
                kind="paper",
                title=title,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                text=clean_text(entry.findtext("a:summary", default="", namespaces=NS)),
                published=entry.findtext("a:published", default=None, namespaces=NS),
                meta={"arxiv_id": arxiv_id, "authors": authors[:4], "categories": categories},
            )
        )
    return items


def fetch(ctx: FetchContext) -> list[Item]:
    max_results = int(ctx.config.sources[NAME].get("max_results", 100))
    items: dict[str, Item] = {}
    errors: list[str] = []
    ok = 0
    done: set[str] = set()
    for topic in ctx.config.topics:
        categories = [str(c) for c in topic.source_option(NAME, "categories", [])]
        terms = [str(t) for t in topic.source_option(NAME, "terms", [])]
        if not categories and not terms:
            terms = topic.keywords[:3]
        if not categories and not terms:
            continue
        query = build_query(categories, terms, ctx.since, ctx.until)
        if query in done:
            continue
        done.add(query)
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": max_results,
        }
        try:
            response = ctx.http.get(API, params=params, bucket="arxiv", min_interval=3.2)
            parsed = parse_feed(response.text)
            ok += 1
        except SourceError as exc:
            errors.append(f"topic '{topic.name}': {exc}")
            continue
        if len(parsed) >= max_results:
            ctx.warn(f"arxiv: topic '{topic.name}' hit the {max_results} result cap, older papers in the window were left out")
        for item in parsed:
            items.setdefault(item.url, item)
    settle(ctx, NAME, ok, errors)
    return list(items.values())
