"""Scoring: which items fit which topic, and which few to keep.

With an embedding model, an item's score for a topic is the cosine similarity
between the item text and the topic description (or its keywords, whichever is
closer), plus a small bonus per keyword found, plus a nudge from your earlier
+ and - votes. Without Ollama the score falls back to counting keywords.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import Topic
from .models import Item

EmbedFn = Callable[[list[str], str], list[list[float]]]


@dataclass
class Scored:
    item: Item
    topic: Topic
    score: float
    similarity: float = 0.0
    keyword_hits: int = 0
    adjust: float = 0.0
    vector: list[float] = field(default_factory=list)
    selected: bool = False


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def item_text(item: Item) -> str:
    return f"{item.title}. {item.text}"[:1200]


def _haystack(item: Item) -> str:
    return f"{item.title} {item.text}".lower()


def keyword_hits(item: Item, topic: Topic) -> int:
    hay = _haystack(item)
    return sum(1 for kw in topic.keywords if kw.lower() in hay)


def is_excluded(item: Item, phrases: Sequence[str]) -> bool:
    hay = _haystack(item)
    return any(p.lower() in hay for p in phrases if p)


def feedback_adjust(vector: Sequence[float], likes: list[list[float]], dislikes: list[list[float]], weight: float) -> float:
    """A small heuristic: push items that look like past + votes up and past - votes down."""
    like = max((cosine(vector, v) for v in likes), default=0.0)
    dislike = max((cosine(vector, v) for v in dislikes), default=0.0)
    return weight * (max(0.0, like - 0.5) - max(0.0, dislike - 0.5))


def rank(
    items: list[Item],
    topics: list[Topic],
    *,
    embed: EmbedFn | None,
    global_exclude: Sequence[str] = (),
    likes: list[list[float]] | None = None,
    dislikes: list[list[float]] | None = None,
    feedback_weight: float = 0.2,
) -> tuple[list[Scored], str]:
    """Return every item with its best topic and score, best first, and the mode used."""
    candidates = [i for i in items if not is_excluded(i, global_exclude)]
    if not candidates or not topics:
        return [], "embeddings" if embed else "keywords"

    mode = "keywords"
    item_vectors: list[list[float]] = []
    topic_vectors: list[list[list[float]]] = []
    if embed is not None:
        # An Ollama failure here raises; the pipeline catches it and retries in keyword mode.
        topic_vectors = []
        for topic in topics:
            texts = [topic.embed_text()]
            if topic.keywords:
                texts.append(", ".join(topic.keywords))
            topic_vectors.append(embed(texts, "query"))
        item_vectors = embed([item_text(i) for i in candidates], "document")
        mode = "embeddings"

    scored: list[Scored] = []
    for index, item in enumerate(candidates):
        best: Scored | None = None
        for t_index, topic in enumerate(topics):
            if is_excluded(item, topic.exclude):
                continue
            hits = keyword_hits(item, topic)
            if mode == "embeddings":
                vector = item_vectors[index]
                sim = max(cosine(vector, tv) for tv in topic_vectors[t_index])
                adjust = feedback_adjust(vector, likes or [], dislikes or [], feedback_weight)
                score = sim + min(0.09, 0.03 * hits) + adjust
                candidate = Scored(item, topic, score, sim, hits, adjust, vector)
            else:
                score = min(0.95, 0.5 + 0.15 * hits) if hits else 0.0
                candidate = Scored(item, topic, score, 0.0, hits, 0.0, [])
            if best is None or candidate.score > best.score:
                best = candidate
        if best is not None:
            scored.append(best)
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored, mode


def select(scored: list[Scored], *, min_score: float, max_items: int, max_per_topic: int) -> list[Scored]:
    """Take the best items above the bar, with a cap per topic and overall. No padding."""
    chosen: list[Scored] = []
    per_topic: dict[str, int] = {}
    for entry in scored:
        if len(chosen) >= max_items:
            break
        if entry.score < min_score:
            break
        count = per_topic.get(entry.topic.name, 0)
        if count >= max_per_topic:
            continue
        per_topic[entry.topic.name] = count + 1
        entry.selected = True
        chosen.append(entry)
    return chosen
