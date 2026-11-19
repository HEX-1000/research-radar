"""Scoring: which items fit which topic, and which few to keep.

With an embedding model, an item's score for a topic is the cosine similarity
between the item text and the topic description (or its keywords, whichever is
closer), plus a small bonus per keyword found. Without Ollama the score falls back to counting keywords.
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

# Items this far below the bar of their topic are still asked about by `radar label`, and the scan
# asks the follower count of a thin looking repository inside this band. It lives here because
# labelling.py imports pipeline.py.
LABEL_BAND = 0.08


@dataclass
class Scored:
    item: Item
    topic: Topic
    score: float
    bar: float = 0.0  # the bar of its topic; set by rank
    similarity: float = 0.0
    keyword_hits: int = 0
    adjust: float = 0.0
    vector: list[float] = field(default_factory=list)
    selected: bool = False
    held: str = ""  # why the trust check kept it back, when it did
    outcome: str = ""  # set by select: shown, held, below_bar, topic_full or day_full

    @property
    def margin(self) -> float:
        """How far the score is above the bar of its topic (negative when below)."""
        return self.score - self.bar


@dataclass
class Ranking:
    """What rank found. `scored` holds every item that has a topic it may go under, best first."""

    scored: list[Scored]
    mode: str  # embeddings or keywords
    left_out: dict[str, str] = field(default_factory=dict)  # item key: "lure" or "excluded"
    vectors: dict[str, list[float]] = field(default_factory=dict)  # item key: vector made by this call
    # item key: (topic name, similarity, keyword hits) for each topic the item competed for
    measures: dict[str, list[tuple[str, float, int]]] = field(default_factory=dict)


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


def _separated(keyword: str) -> str:
    """The keyword escaped, with each run of spaces, hyphens and underscores as exactly one of the three."""
    parts = [re.escape(part) for part in re.split(r"[\s_-]+", keyword.lower().strip()) if part]
    return "[-_ ]".join(parts) if parts else re.escape(keyword.lower().strip())


@lru_cache(maxsize=1024)
def keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Keywords of four letters or fewer match whole words ("rust" does not match "rustic"),
    with a plural s allowed. Longer ones match at the start of a word ("caption" finds "captioning").
    A space, hyphen or underscore in a keyword matches any one of the three ("screen reader" finds "screen-reader")."""
    word = _separated(keyword)
    if len(keyword.strip()) <= 4:
        return re.compile(f"{_START}{word}s?{_END}")
    return re.compile(f"{_START}{word}")


@lru_cache(maxsize=1024)
def _exclude_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(_separated(phrase))


def keyword_hits(item: Item, topic: Topic) -> int:
    hay = _haystack(item)
    return sum(1 for kw in topic.keywords if kw.strip() and keyword_pattern(kw).search(hay))


def is_excluded(item: Item, phrases: Sequence[str]) -> bool:
    hay = _haystack(item)
    return any(_exclude_pattern(p).search(hay) for p in phrases if p and p.strip())


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


# Selling endings of a label such as "nvidia-broadcast-download". One such ending is ordinary
# ("ad-free", "voice-activated", "mit-license"); two different ones on one stem are not
# (decision 56, tested on 3,518 repositories in review-data/lure_gap_test.py).
_SELLING_LABEL = re.compile(r"^(?P<stem>.+)[-_](?P<ending>downloads?|free|activated|crack(?:ed)?|license)$")
_SAME_ENDING = {"downloads": "download", "cracked": "crack"}


def _labels(item: Item) -> list[str]:
    """The repository name and its topics (or a model's tags), lower case."""
    out = [item.title.split("/")[-1].lower()]
    topics = item.meta.get("topics")
    if not topics:
        text = f" {item.text}"
        for mark in (" Topics: ", " Tags: "):
            if mark in text:
                topics = [t.strip() for t in text.split(mark, 1)[1].split(",")]
    out += [str(t).lower() for t in topics or []]
    return out


def _two_endings_on_one_stem(item: Item) -> bool:
    found: dict[str, set[str]] = {}
    for label in _labels(item):
        m = _SELLING_LABEL.match(label.replace(".", "-"))
        if m:
            found.setdefault(m["stem"], set()).add(_SAME_ENDING.get(m["ending"], m["ending"]))
    return any(len(endings) >= 2 for endings in found.values())


def is_lure(item: Item) -> bool:
    """True for a repository or model whose name, description or topics offer cracked software, or
    whose name and topics carry two different selling endings on one stem."""
    if item.kind == "paper":
        return False
    return _LURE.search(_haystack(item)) is not None or _two_endings_on_one_stem(item)


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
    competes: Callable[[Item, Topic], bool] | None = None,
    known: dict[str, list[float]] | None = None,
    bars: dict[str, float] | None = None,
) -> Ranking:
    """Give every item its best topic and score, best first.

    `bars` maps a topic name to its bar. An item goes to the topic where its score clears the bar
    by the most; with equal bars (the default is 0 for every topic) that is the best score. The
    result is sorted by that margin.

    `competes(item, topic)` says whether an item competes for a topic in this scan (default:
    always). An item that competes for no topic is left out and not embedded. `known` maps an
    item key to a stored vector of the same embedding model, used instead of embedding again."""
    ranking = Ranking([], "embeddings" if embed else "keywords")
    plan: list[tuple[Item, list[Topic]]] = []
    for item in items:
        mine = [t for t in topics if competes is None or competes(item, t)]
        if not mine:
            continue
        if is_lure(item):
            ranking.left_out[item.key] = "lure"
        elif is_excluded(item, global_exclude):
            ranking.left_out[item.key] = "excluded"
        else:
            plan.append((item, mine))
    if not plan:
        return ranking

    item_vectors: dict[str, list[float]] = {}
    topic_vectors: dict[str, list[list[float]]] = {}
    if embed is not None:
        # An Ollama failure here raises; the pipeline catches it and retries in keyword mode.
        for topic in topics:
            if any(topic in mine for _, mine in plan):
                texts = [topic.embed_text()]
                if topic.keywords:
                    texts.append(", ".join(topic.keywords))
                topic_vectors[topic.name] = embed(texts, "query")
        stored = known or {}
        missing = [item for item, _ in plan if item.key not in stored]
        if missing:
            ranking.vectors = dict(zip([i.key for i in missing], embed([item_text(i) for i in missing], "document")))
        item_vectors = {**ranking.vectors, **{item.key: stored[item.key] for item, _ in plan if item.key in stored}}

    for item, mine in plan:
        best: Scored | None = None
        measures: list[tuple[str, float, int]] = []
        for topic in mine:
            bar = (bars or {}).get(topic.name, 0.0)
            hits = keyword_hits(item, topic)
            if embed is not None:
                vector = item_vectors[item.key]
                sim = max(cosine(vector, tv) for tv in topic_vectors[topic.name])
                measures.append((topic.name, sim, hits))
                if is_excluded(item, topic.exclude):
                    continue
                adjust = feedback_adjust(vector, likes or [], dislikes or [], feedback_weight)
                score = sim + min(0.09, 0.03 * hits) + adjust
                candidate = Scored(item, topic, score, bar, sim, hits, adjust, vector)
            else:
                if is_excluded(item, topic.exclude):
                    continue
                score = min(0.95, 0.5 + 0.15 * hits) if hits else 0.0
                candidate = Scored(item, topic, score, bar, 0.0, hits, 0.0, [])
            if best is None or candidate.margin > best.margin:
                best = candidate
        if measures:
            ranking.measures[item.key] = measures
        if best is None:
            ranking.left_out[item.key] = "excluded"  # by the exclude list of every topic it competed for
        else:
            ranking.scored.append(best)
    ranking.scored.sort(key=lambda s: s.margin, reverse=True)
    return ranking


def select(
    scored: list[Scored],
    *,
    bars: dict[str, float],
    max_items: int,
    caps: dict[str, int],
    hold: Callable[[Scored], str | None] | None = None,
) -> list[Scored]:
    """Take the items that clear the bar of their topic by the most, with a cap per topic and
    overall. No padding. `bars` and `caps` are keyed by topic name.

    `hold` is asked about an entry only when it would be chosen. It returns a reason to keep the
    entry back, or None. A held entry takes no place, so the next candidate does. Every entry gets
    an `outcome` that says what selection did with it."""
    chosen: list[Scored] = []
    per_topic: dict[str, int] = {}
    for entry in scored:
        if entry.score < bars[entry.topic.name]:
            entry.outcome = "below_bar"
            continue
        if len(chosen) >= max_items:
            entry.outcome = "day_full"
            continue
        count = per_topic.get(entry.topic.name, 0)
        if count >= caps[entry.topic.name]:
            entry.outcome = "topic_full"
            continue
        reason = hold(entry) if hold else None
        if reason:
            entry.held = reason
            entry.outcome = "held"
            continue
        per_topic[entry.topic.name] = count + 1
        entry.selected = True
        entry.outcome = "shown"
        chosen.append(entry)
    return chosen
