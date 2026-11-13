"""Scoring: which items fit which topic, and which few to keep.

With an embedding model, an item's score for a topic is the cosine similarity
between the item text and the topic description (or its keywords, whichever is
closer), plus a small bonus per keyword found, plus a nudge from your earlier
+ and - votes. Without Ollama the score falls back to counting keywords.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
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
    held: str = ""  # why the trust check kept it back, when it did


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


# Not next to a letter or digit. Hyphens, underscores, dots and spaces separate words in repository names.
_START = r"(?<![^\W_])"
_END = r"(?![^\W_])"


@lru_cache(maxsize=1024)
def keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Keywords of four letters or fewer match whole words ("stac" does not match "stack"),
    with a plural s allowed. Longer ones match at the start of a word ("backtest" finds "backtesting")."""
    word = re.escape(keyword.lower().strip())
    if len(keyword.strip()) <= 4:
        return re.compile(f"{_START}{word}s?{_END}")
    return re.compile(f"{_START}{word}")


def keyword_hits(item: Item, topic: Topic) -> int:
    hay = _haystack(item)
    return sum(1 for kw in topic.keywords if kw.strip() and keyword_pattern(kw).search(hay))


def is_excluded(item: Item, phrases: Sequence[str]) -> bool:
    hay = _haystack(item)
    return any(p.lower() in hay for p in phrases if p)


# Wording of repositories that offer cracked or "activated" software, a common way to spread
# malware; such repositories often buy stars to rank high in a search. Papers are not checked: an
# abstract can say "full version" or "free download" for honest reasons. A bare "crack" is not
# enough either, because crack detection, "crack-free" coatings and crack patching are real subjects.
_LURE = re.compile(
    _START
    + r"(?:"
    + r"crack(?:ed)?[-_ ]?(?:20\d\d|windows|win|mac|pc|downloads?|full|version|keys?|serial|activation)"
    + r"|cracked|keygen|warez|nulled"
    + r"|serial[-_ ]?keys?|license[-_ ]?keys?|activation[-_ ]?(?:keys?|codes?)|activators?|pre[-_ ]?activated"
    + r"|free[-_ ]?downloads?|full[-_ ]?version|premium[-_ ]?unlocked|unlocked[-_ ]?premium"
    + r")"
    + _END
)


def is_lure(item: Item) -> bool:
    """True for a repository or model whose name, description or topics offer cracked software."""
    return item.kind != "paper" and _LURE.search(_haystack(item)) is not None


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
    candidates = [i for i in items if not is_excluded(i, global_exclude) and not is_lure(i)]
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


def select(
    scored: list[Scored],
    *,
    min_score: float,
    max_items: int,
    max_per_topic: int,
    hold: Callable[[Scored], str | None] | None = None,
) -> list[Scored]:
    """Take the best items above the bar, with a cap per topic and overall. No padding.

    `hold` is asked about an entry only when it would be chosen. It returns a reason to keep the
    entry back, or None. A held entry takes no place, so the next candidate does."""
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
        reason = hold(entry) if hold else None
        if reason:
            entry.held = reason
            continue
        per_topic[entry.topic.name] = count + 1
        entry.selected = True
        chosen.append(entry)
    return chosen
