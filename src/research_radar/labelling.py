"""The labelled set: `radar label` asks about candidates one at a time, `radar replay` shows what
each bar would have kept, missed and let through.

A label says whether an item fits a topic (`fits`, `no`) or is a lure (topic `*`). A vote counts as
a label when the item has none. Both commands need state/radar.db; replay only reads it.
"""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from .candidates import DB_NAME, CandidateStore
from .config import Config, Topic
from .llm import OllamaError
from .models import Item
from .pipeline import topic_fingerprint
from .ranker import cosine, keyword_hits
from .sources.github import THIN_MAX_FOLLOWERS, looks_thin

LABEL_BAND = 0.08  # items this far below the bar of their topic are still asked about
POOL_OUTCOMES = ("shown", "below_bar", "topic_full", "day_full", "held")
NO_STORE = "No candidate store yet. Run a scan first."
TEXT_SHOWN = 400  # characters of the stored text shown while labelling
BAR_FIRST, BAR_LAST, BAR_STEP = 0.30, 0.60, 0.02


def topic_bar(cfg: Config, topic: Topic) -> float:
    return float(cfg.settings["min_score"] if topic.min_score is None else topic.min_score)


# radar label -----------------------------------------------------------------------------------


@dataclass
class Candidate:
    key: str
    topic: str
    bar: float
    score: float
    outcome: str
    title: str
    text: str
    kind: str
    url: str
    published: str | None
    source: str = ""


def build_pool(store: CandidateStore, cfg: Config, now: datetime, days: int) -> list[Candidate]:
    """Candidates worth a label, nearest to the bar first, one topic after another in turn."""
    cutoff = (now - timedelta(days=days)).astimezone(timezone.utc).isoformat()
    bars = {t.name: topic_bar(cfg, t) for t in cfg.topics}
    rows = store.conn.execute(
        "SELECT d.key, d.topic, d.score, d.outcome, i.title, i.text, i.kind, i.url, i.published"
        " FROM decision d JOIN item i ON i.key = d.key"
        " WHERE d.scan = (SELECT MAX(d2.scan) FROM decision d2 JOIN scan s ON s.id = d2.scan"
        "                 WHERE d2.key = d.key AND s.at >= ?)",
        (cutoff,),
    ).fetchall()
    done_topics: dict[str, set[str]] = {}
    for key, topic, _value in store.labels():
        done_topics.setdefault(key, set()).add(topic)
    voted = store.votes_by_key()

    by_topic: dict[str, list[Candidate]] = {name: [] for name in bars}
    for key, topic, score, outcome, title, text, kind, url, published in rows:
        if topic not in bars or outcome not in POOL_OUTCOMES or score < bars[topic] - LABEL_BAND:
            continue
        if key in voted or {topic, "*"} & done_topics.get(key, set()):
            continue
        by_topic[topic].append(Candidate(key, topic, bars[topic], score, outcome, title, text, kind, url, published))
    queues = []
    for name in bars:  # the order of interests.yaml
        queue = sorted(by_topic[name], key=lambda c: (abs(c.score - c.bar), c.key))
        if queue:
            queues.append(queue)
    pool: list[Candidate] = []
    for position in range(max((len(q) for q in queues), default=0)):
        pool.extend(q[position] for q in queues if position < len(q))
    return pool


def _source_of(store: CandidateStore, candidate: Candidate) -> str:
    row = store.conn.execute("SELECT source FROM link WHERE key = ? AND source != '' ORDER BY rowid LIMIT 1", (candidate.key,)).fetchone()
    return row[0] if row else candidate.key.split(":", 1)[0]


def _show(store: CandidateStore, c: Candidate, left: int, show: Callable[[str], None]) -> None:
    text = " ".join(c.text.split())[:TEXT_SHOWN] or "(no text stored)"
    show("")
    show(f"[{left} left] {c.topic} · bar {c.bar:.2f} · score {c.score:.3f} · {c.outcome}")
    show(c.title)
    show(f"{c.kind} · {_source_of(store, c)} · {(c.published or 'no date')[:10]}")
    show(c.url)
    show(text)


def run_label(
    cfg: Config,
    days: int = 14,
    *,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    open_url: Callable[[str], object] = webbrowser.open,
    now: datetime | None = None,
) -> int:
    """Ask about candidates until the pool or the person runs out. Each answer is saved at once."""
    if not (cfg.state_dir / DB_NAME).exists():
        show(NO_STORE)
        return 1
    now = now or datetime.now(timezone.utc)
    names = [t.name for t in cfg.topics]
    with CandidateStore.open(cfg.state_dir, embed_model=str(cfg.settings["embed_model"]), now=now, write=True) as store:
        pool = build_pool(store, cfg, now, days)
        if not pool:
            show(f"Nothing to label from the last {days} days.")
            return 0
        show("Topics:")
        for number, topic in enumerate(cfg.topics, start=1):
            show(f"  {number}  {topic.name} (bar {topic_bar(cfg, topic):.2f})")
        prompt = f"y fits / n no / 1-{len(names)} fits that topic instead / l lure / o open / s skip / q quit: "
        answered = 0
        for index, c in enumerate(pool):
            _show(store, c, len(pool) - index, show)
            while True:
                try:
                    answer = ask(prompt).strip().lower()
                except EOFError:
                    answer = "q"
                labels: list[tuple[str, str]] | None = None
                if answer == "y":
                    labels = [(c.topic, "fits")]
                elif answer == "n":
                    labels = [(c.topic, "no")]
                elif answer == "l":
                    labels = [("*", "lure")]
                elif answer.isdigit() and 1 <= int(answer) <= len(names):
                    other = names[int(answer) - 1]
                    labels = [(c.topic, "fits")] if other == c.topic else [(other, "fits"), (c.topic, "no")]
                elif answer == "o":
                    open_url(c.url)
                    continue
                elif answer == "s":
                    break
                elif answer == "q":
                    show(f"Labelled {answered} item{'' if answered == 1 else 's'}.")
                    return 0
                else:
                    show("Not an answer. " + prompt.rstrip(": "))
                    continue
                store.set_labels(c.key, labels, now=datetime.now(timezone.utc))
                answered += 1
                break
        show(f"No more candidates. Labelled {answered} item{'' if answered == 1 else 's'}.")
    return 0


# radar replay ----------------------------------------------------------------------------------


@dataclass
class TopicReplay:
    topic: Topic
    bar: float
    fits: list[float] = field(default_factory=list)  # scores of the items labelled fits
    no: list[float] = field(default_factory=list)
    not_scored: int = 0


@dataclass
class TrustReplay:
    lures: dict[str, int] = field(default_factory=dict)  # caught, not caught, unknown, no observation
    others: dict[str, int] = field(default_factory=dict)
    minus_only: int = 0  # repositories whose only signal is a - vote: lure or not, nobody said


@dataclass
class Replay:
    topics: list[TopicReplay]
    mode: str  # embeddings or stored scores
    gone: int = 0  # labels for topics no longer in interests.yaml
    left_out: int = 0
    trust: TrustReplay = field(default_factory=TrustReplay)

    @property
    def labelled(self) -> int:
        return sum(len(t.fits) + len(t.no) for t in self.topics)


TRUST_CLASSES = ("caught", "not caught", "unknown", "no observation")


def labelled_set(
    store: CandidateStore, topics: list[Topic]
) -> tuple[dict[tuple[str, str], str], int, set[str], set[str], set[str]]:
    """The labelled set: {(key, topic name): fits or no}, the number of labels for topics that are
    gone, and three sets of keys for the trust section: lures, others (a fits or no label, or a
    + vote) and items whose only signal is a - vote.

    A lure label is a no for every topic, whatever the item's other labels say: `radar label` never
    asks about a lure again, so the lure answer is the later one. A label wins over a vote."""
    names = {t.name for t in topics}
    entries: dict[tuple[str, str], str] = {}
    lures: set[str] = set()
    judged: set[str] = set()  # asked with "lure" among the answers, and answered fits or no
    gone: set[tuple[str, str]] = set()
    for key, topic, value in store.labels():
        if topic == "*" and value == "lure":
            lures.add(key)
        elif value in ("fits", "no"):
            judged.add(key)
            if topic in names:
                entries[(key, topic)] = value
            else:
                gone.add((key, topic))
    for key in lures:
        for name in names:
            entries[(key, name)] = "no"
    votes = store.votes_by_key()
    shown = store.known(votes)
    for key, vote in votes.items():
        topic = shown[key].topic if key in shown else None
        if topic is None or key in lures or (key, topic) in entries or (key, topic) in gone:
            continue
        if topic in names:
            entries[(key, topic)] = "fits" if vote > 0 else "no"
        else:
            gone.add((key, topic))
    others = (judged | {key for key, vote in votes.items() if vote > 0}) - lures
    minus_only = {key for key, vote in votes.items() if vote < 0} - judged - lures
    return entries, len(gone), lures, others, minus_only


def _scored_with_ollama(store, topics, wanted, items, model, ollama) -> dict[tuple[str, str], float] | None:
    """Score of each labelled (key, topic) from the stored vector and the current topic text. None
    when Ollama cannot do it."""
    try:
        if not ollama.ping():
            return None
        topic_vectors = {}
        for topic in topics:
            texts = [topic.embed_text()]
            if topic.keywords:
                texts.append(", ".join(topic.keywords))
            topic_vectors[topic.name] = ollama.embed(texts, "query")
    except OllamaError:
        return None
    vectors = store.vectors({key for key, _ in wanted}, model)
    by_name = {t.name: t for t in topics}
    scores: dict[tuple[str, str], float] = {}
    for key, name in wanted:
        if key not in vectors or key not in items:
            continue
        similarity = max(cosine(vectors[key], tv) for tv in topic_vectors[name])
        scores[(key, name)] = similarity + min(0.09, 0.03 * keyword_hits(items[key], by_name[name]))
    return scores


def _scored_from_rows(store, topics, wanted, model) -> dict[tuple[str, str], float]:
    """Score rows of the current topic text and model, as the scans wrote them."""
    scores: dict[tuple[str, str], float] = {}
    for topic in topics:
        keys = [key for key, name in wanted if name == topic.name]
        rows = store._rows_in(
            "SELECT key, similarity, keyword_hits FROM score WHERE model = ? AND topic = ? AND topic_mark = ?"
            " AND key IN ({})",
            keys,
            (model, topic.name, topic_fingerprint(topic)),
        )
        for key, similarity, hits in rows:
            scores[(key, topic.name)] = similarity + min(0.09, 0.03 * hits)
    return scores


def _trust_class(store: CandidateStore, key: str) -> str:
    row = store.conn.execute(
        "SELECT size_kb, stars, followers FROM observation WHERE key = ? ORDER BY at LIMIT 1", (key,)
    ).fetchone()
    if row is None:
        return "no observation"
    size_kb, stars, followers = row
    if not looks_thin(size_kb, stars):
        return "not caught"
    if followers is None:
        return "unknown"  # the follower count was never asked
    return "caught" if followers <= THIN_MAX_FOLLOWERS else "not caught"


def _trust(store: CandidateStore, lures: set[str], others: set[str], minus_only: set[str]) -> TrustReplay:
    """A - vote alone does not say whether a repository is a lure, so it is in neither row."""
    trust = TrustReplay({c: 0 for c in TRUST_CLASSES}, {c: 0 for c in TRUST_CLASSES})
    for key in sorted(k for k in lures | others if k.startswith("github:")):
        (trust.lures if key in lures else trust.others)[_trust_class(store, key)] += 1
    trust.minus_only = sum(1 for key in minus_only if key.startswith("github:"))
    return trust


def replay(cfg: Config, ollama=None) -> Replay | None:
    """What each bar would keep, miss and let through for the labelled set. None without a store."""
    if not (cfg.state_dir / DB_NAME).exists():
        return None
    model = str(cfg.settings["embed_model"])
    with CandidateStore.open(cfg.state_dir, embed_model=model, now=datetime.now(timezone.utc), write=False) as store:
        entries, gone, lures, others, minus_only = labelled_set(store, cfg.topics)
        wanted = sorted(entries)
        rows = store._rows_in("SELECT key, kind, url, title, text FROM item WHERE key IN ({})", {k for k, _ in wanted})
        items = {key: Item("", kind, title, url, text) for key, kind, url, title, text in rows}
        scores = _scored_with_ollama(store, cfg.topics, wanted, items, model, ollama) if ollama is not None else None
        mode = "embeddings"
        if scores is None:
            mode = "stored scores"
            scores = _scored_from_rows(store, cfg.topics, wanted, model)
        result = Replay([TopicReplay(t, topic_bar(cfg, t)) for t in cfg.topics], mode, gone=gone)
        by_name = {r.topic.name: r for r in result.topics}
        for (key, name), value in entries.items():
            row = by_name[name]
            score = scores.get((key, name))
            if score is None:
                row.not_scored += 1
                result.left_out += 1
            else:
                (row.fits if value == "fits" else row.no).append(score)
        result.trust = _trust(store, lures, others, minus_only)
    return result


def is_current(bar: float, row: TopicReplay) -> bool:
    return abs(bar - row.bar) < 1e-9


def bar_text(bar: float) -> str:
    """Two decimals, or three for a bar such as 0.435."""
    text = f"{bar:.2f}"
    return text if abs(float(text) - bar) < 1e-9 else f"{bar:.3f}"


def bar_rows(row: TopicReplay) -> list[tuple[float, int, int, int]]:
    """(bar, kept, noise, missed) from 0.30 to 0.60 in steps of 0.02, limited to the range the
    labelled scores reach plus one step each side. The current bar always has a row, at its exact
    value: off the grid or outside that range, it is added in order."""
    scores = row.fits + row.no
    if not scores:
        return []
    grid = [round(BAR_FIRST + BAR_STEP * i, 2) for i in range(int(round((BAR_LAST - BAR_FIRST) / BAR_STEP)) + 1)]
    low = max((i for i, b in enumerate(grid) if b <= min(scores)), default=0)
    high = min((i for i, b in enumerate(grid) if b > max(scores)), default=len(grid) - 1)
    bars = grid[max(0, low - 1) : min(len(grid) - 1, high + 1) + 1]
    eps = 1e-9
    if not any(is_current(b, row) for b in bars):
        bars = sorted([*bars, row.bar])
    return [
        (
            b,
            sum(1 for s in row.fits if s >= b - eps),
            sum(1 for s in row.no if s >= b - eps),
            sum(1 for s in row.fits if s < b - eps),
        )
        for b in bars
    ]


def format_replay(result: Replay) -> list[str]:
    lines: list[str] = []
    if result.mode == "stored scores":
        lines.append(
            "Ollama is not ready, so replay used the scores stored at scan time for the current topic text"
            f" and model. {result.left_out} label{'' if result.left_out == 1 else 's'} left out."
        )
        lines.append("")
    elif result.left_out:
        lines.append(f"{result.left_out} labelled item{'' if result.left_out == 1 else 's'} not scored (no stored vector).")
        lines.append("")
    for row in result.topics:
        suffix = f" ({row.not_scored} not scored)" if row.not_scored and result.mode == "embeddings" else ""
        lines.append(f"{row.topic.name}: {len(row.fits)} fits, {len(row.no)} no{suffix}")
        table = bar_rows(row)
        if table:
            lines.append(" bar   kept  noise  missed")
            for bar, kept, noise, missed in table:
                mark = "   < current" if is_current(bar, row) else ""
                label = bar_text(bar)
                lines.append(f"{label}{kept:{11 - len(label)}d}{noise:7d}{missed:8d}{mark}")
        lines.append("")
    if result.gone:
        lines.append(f"{result.gone} label{'' if result.gone == 1 else 's'} for topics no longer in interests.yaml.")
        lines.append("")
    trust = result.trust
    lines.append("Trust check on labelled GitHub repositories (first observation of each):")
    lines.append(f"{'':8}" + "".join(f"{c:>16}" for c in TRUST_CLASSES))
    for name, counts in (("lures", trust.lures), ("others", trust.others)):
        lines.append(f"{name:8}" + "".join(f"{counts.get(c, 0):16d}" for c in TRUST_CLASSES))
    if trust.minus_only:
        one = trust.minus_only == 1
        lines.append(
            f"{trust.minus_only} repositor{'y' if one else 'ies'} with only a - vote {'is' if one else 'are'} left out:"
            " a - vote does not say whether it is a lure."
        )
    return lines


def run_replay(cfg: Config, ollama=None, show: Callable[[str], None] = print) -> int:
    result = replay(cfg, ollama)
    if result is None:
        show(NO_STORE)
        return 1
    if not result.labelled and not result.left_out and not result.gone:
        show("No labels yet. Run `radar label`.")
        return 0
    for line in format_replay(result):
        show(line)
    return 0
