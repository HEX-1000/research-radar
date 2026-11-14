"""One scan, from the last successful run up to now."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .candidates import CandidateStore, ScanRecord, StoredItem
from .config import Config, Topic
from .deliver import DeliveryResult, collect_telegram_votes, deliver_pending, write_digest_file
from .digest import MONTHS
from .http import Http, SourceError
from .llm import Ollama, OllamaError
from .lock import scan_lock
from .models import Item, Report, SkippedSource
from .notes import write_note
from .ranker import LABEL_BAND, Ranking, Scored, rank, select
from .sources import SOURCES, FetchContext
from .sources.arxiv import NAME as ARXIV, OVERLAP_DAYS as ARXIV_OVERLAP_DAYS
from .sources.feeds import fold_earlier
from .sources.github import (
    HOLD_DAYS,
    check_repos,
    fetch_readme,
    fetch_repo,
    looks_thin,
    owner_followers,
    parse_repo,
    thin_reason,
)
from .sources.github import GONE as GITHUB_GONE
from .store import Store
from .summarize import WATCH_MIN_TEXT, fallback_note, summarize, watch_note

Progress = Callable[[str], None]

MAX_UNCHOSEN_OWNERS = 10  # follower counts asked per scan for repositories that selection did not ask about


@dataclass
class RunResult:
    date: str
    new_reports: list[Report] = field(default_factory=list)
    all_reports: list[Report] = field(default_factory=list)
    skipped: list[SkippedSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ranking: str = "embeddings"
    note_path: Path | None = None
    digest_path: Path | None = None
    deliveries: list[DeliveryResult] = field(default_factory=list)
    fetched: int = 0
    explain: list[Scored] = field(default_factory=list)
    backfilled: list[str] = field(default_factory=list)  # new or changed topics that got a look back
    preview: bool = False  # a keyword ranking: the items are shown, nothing was saved or sent


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# Days a source's window reaches back before its last good run.
OVERLAP_DAYS = {ARXIV: ARXIV_OVERLAP_DAYS}


def window_start(
    cfg: Config, last_run: str | None, now: datetime, days: int | None, overlap_days: int = 0
) -> datetime:
    """Where a source's window begins: its last good run minus the source's overlap,
    capped by max_catchup_days."""
    if days is not None:
        return now - timedelta(days=days)
    earliest = now - timedelta(days=int(cfg.settings["max_catchup_days"]))
    previous = _parse_iso(last_run)
    if previous is None:
        return now - timedelta(days=int(cfg.settings["first_run_days"]))
    return max(previous - timedelta(days=overlap_days), earliest)


def topic_fingerprint(topic: Topic) -> str:
    """Changes when the text of a topic or its per source queries change."""
    payload = [topic.description.strip(), topic.keywords, topic.exclude, topic.options]
    if topic.watch:  # appended only when set, so topics without a watch list keep their marks
        payload.append(topic.watch)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _noop(_: str) -> None:
    return None


def scan(
    cfg: Config,
    *,
    http: Http | None = None,
    ollama: Ollama | None = None,
    now: datetime | None = None,
    progress: Progress = _noop,
    days: int | None = None,
    use_llm: bool = True,
) -> RunResult:
    now = now or datetime.now(timezone.utc)
    local_date = now.astimezone().date().isoformat()
    store = Store(cfg.state_dir)
    result = RunResult(date=local_date)

    own_http = http is None
    http = http or Http(github_token=cfg.github_token())
    own_ollama = ollama is None
    ollama = ollama or Ollama(
        cfg.settings["ollama_url"], cfg.settings["embed_model"], cfg.settings["chat_model"]
    )
    model = str(cfg.settings["embed_model"])
    try:
        llm_ready = use_llm and ollama.ping()
        args = (cfg, store, http, ollama, now, local_date, progress, days, use_llm, llm_ready, result)
        if not llm_ready:  # a keyword ranking is a preview: it writes nothing, so it needs no lock
            with CandidateStore.open(cfg.state_dir, embed_model=model, now=now, write=False) as candidates:
                return _scan(candidates, *args)
        with scan_lock(cfg.state_dir):
            # The first scan of this version builds the store from seen.json and vectors.json.
            with CandidateStore.open(cfg.state_dir, embed_model=model, now=now) as candidates:
                return _scan(candidates, *args)
    finally:
        if own_http:
            http.close()
        if own_ollama:
            ollama.close()


def _scan(
    candidates: CandidateStore,
    cfg: Config,
    store: Store,
    http: Http,
    ollama: Ollama,
    now: datetime,
    local_date: str,
    progress: Progress,
    days: int | None,
    use_llm: bool,
    llm_ready: bool,
    result: RunResult,
) -> RunResult:
    last_runs = store.last_runs()
    model = str(cfg.settings["embed_model"])
    record = ScanRecord(now=now, day=local_date, embed_model=model)

    # Votes pressed on Telegram buttons since the last scan count in this scan's ranking.
    if cfg.channel_enabled("telegram"):
        _, vote_warning = collect_telegram_votes(cfg, candidates, now)
        if vote_warning:
            result.warnings.append(vote_warning)

    # Topics that are new or were edited since the last scan get a one time look back, because
    # the window of a source only reaches back to the last scan. The first scan after the
    # upgrade to this behaviour just records the topics, so nothing is fetched twice.
    marks = store.topic_marks()
    current = {t.name: topic_fingerprint(t) for t in cfg.topics}
    look_back = int(cfg.settings["first_run_days"])
    explicit_short = days is not None and days < look_back
    pending = [] if marks is None else [t for t in cfg.topics if marks.get(t.name) != current[t.name]]

    # 1. Fetch -------------------------------------------------------------------
    # One item per key: the same paper from arXiv and from Hugging Face is one item.
    advanced: dict[str, str] = {}
    windows: dict[str, datetime] = {}
    for name, fetch in SOURCES.items():
        if not cfg.source_enabled(name):
            continue
        progress(f"Reading {name}")
        windows[name] = window_start(cfg, last_runs.get(name), now, days, OVERLAP_DAYS.get(name, 0))
        ctx = FetchContext(config=cfg, http=http, since=windows[name], until=now)
        try:
            for item in fetch(ctx):
                record.add(item)
        except SourceError as exc:
            result.skipped.append(SkippedSource(name, str(exc)))
            continue
        result.warnings.extend(ctx.warnings)
        if not ctx.warnings:  # a source with failed requests keeps its old window
            advanced[name] = now.isoformat()

    only_for_new: set[str] = set()
    looked_back = False
    if pending and not explicit_short:
        reached = False
        back_cfg = replace(cfg, topics=pending)
        back_start = now - timedelta(days=look_back)
        failed = False
        for name, fetch in SOURCES.items():
            if name not in windows or back_start >= windows[name]:
                continue  # disabled, or the normal window already reaches that far
            progress(f"Looking back {look_back} days for new topics: {name}")
            reached = True
            ctx = FetchContext(config=back_cfg, http=http, since=back_start, until=windows[name])
            try:
                for item in fetch(ctx):
                    if _fold_into_watched(record, item):
                        continue  # the feed already has this scan's item; this entry is an earlier one
                    if record.add(item):
                        only_for_new.add(item.key)
            except SourceError:
                failed = True
                continue
            if ctx.warnings:
                failed = True
        looked_back = not failed
        if failed:
            result.warnings.append("Could not look back for new topics on every source; trying again at the next scan.")
        elif reached:
            result.backfilled = [t.name for t in pending]
    result.fetched = record.fetched = len(record.items)
    record.windows = {name: [start.isoformat(), now.isoformat()] for name, start in windows.items()}

    # Items shown, dropped or gone never compete again; held ones only at their recheck.
    known = candidates.known(record.items)
    fresh: dict[str, Item] = {}
    for key, item in list(record.items.items()):
        row = known.get(key)
        if row is not None and row.url != item.url:
            item = record.items[key] = replace(item, url=row.url)  # the link shown is the first one stored
        if row is None or row.state == "seen":
            fresh[key] = item

    # Entries of a watched feed are shown without a bar: they skip ranking and selection.
    watched = [item for item in fresh.values() if item.meta.get("watch_topic")]
    for item in watched:
        del fresh[item.key]

    # Repositories held back a week ago are asked about again; one that is not thin joins the
    # candidates of this scan and competes for every topic.
    trust_on = bool(cfg.settings.get("trust_check", True))
    cleared: set[str] = set()
    if cfg.source_enabled("github") and llm_ready:  # a preview skips the held repositories
        for item in _recheck_held(http, candidates, record, local_date, trust_on, result, progress):
            record.add(item)
            fresh[item.key] = record.items[item.key]
            cleared.add(item.key)

    # 2. Rank --------------------------------------------------------------------
    # An item competes for a topic once per version of the topic's text and of the embedding
    # model, so an item fetched again is not scored again and an edited topic gets one look.
    likes, dislikes = candidates.vote_vectors(model)
    done = candidates.scored(fresh, model)
    pending_names = {t.name for t in pending}

    def competes(item: Item, topic: Topic) -> bool:
        key = item.key
        if key in cleared:
            return True
        if key in only_for_new and topic.name not in pending_names:
            return False  # found only by the look back, so it answers the new topics alone
        return (key, topic.name, current[topic.name]) not in done

    competing = [i for i in fresh.values() if any(competes(i, t) for t in cfg.topics)]
    stored_vectors = candidates.vectors([i.key for i in competing], model) if llm_ready else {}

    # Each topic's bar and cap: its own value, else the one under settings. A keyword preview
    # lowers every bar to 0.5, because one keyword hit is the lowest score in that mode.
    def topic_bars(keywords: bool) -> dict[str, float]:
        bars = {t.name: float(cfg.settings["min_score"] if t.min_score is None else t.min_score) for t in cfg.topics}
        return {name: min(bar, 0.5) for name, bar in bars.items()} if keywords else bars

    caps = {
        t.name: int(cfg.settings["max_per_topic"] if t.max_per_topic is None else t.max_per_topic) for t in cfg.topics
    }
    bars = topic_bars(not llm_ready)

    def rank_all(embed) -> Ranking:
        return rank(
            competing,
            cfg.topics,
            embed=embed,
            global_exclude=cfg.exclude,
            likes=likes if embed else None,
            dislikes=dislikes if embed else None,
            feedback_weight=float(cfg.settings["feedback_weight"]),
            competes=competes,
            known=stored_vectors if embed else None,
            bars=bars,
        )

    ranking = Ranking([], "embeddings" if llm_ready else "keywords")
    if competing:
        progress(f"Scoring {len(competing)} items")
        try:
            ranking = rank_all(ollama.embed if llm_ready else None)
        except OllamaError as exc:
            result.warnings.append(f"Embedding failed ({exc}); used keyword matching instead.")
            llm_ready = False
            bars = topic_bars(True)
            ranking = rank_all(None)
    result.ranking = ranking.mode
    scored = ranking.scored
    if not llm_ready and use_llm:
        result.warnings.append(
            f"Ollama is not reachable at {cfg.settings['ollama_url']}. Items were matched by keywords and described without a model."
        )

    def hold(entry: Scored) -> str | None:
        item = entry.item
        if not trust_on or item.source != "github" or item.key in cleared:
            return None
        meta = item.meta
        return thin_reason(http, meta.get("size_kb"), meta.get("stars"), meta.get("owner"), record.followers)

    chosen = select(
        scored,
        bars=bars,
        max_items=int(cfg.settings["max_items"]),
        caps=caps,
        hold=hold,
    )
    result.explain = scored[:30]
    # Watched entries come first and take no place under max_items or max_per_topic.
    by_name = {t.name: t for t in cfg.topics}
    watch_entries = [
        Scored(item, by_name[item.meta["watch_topic"]], 0.0, selected=True, outcome="watch")
        for item in watched
        if item.meta["watch_topic"] in by_name
    ]
    for entry in watch_entries:
        record.decide(entry.item.key, entry.topic.name, 0.0, "watch")
    shown_entries = watch_entries + chosen
    # A rechecked repository that was not chosen is an ordinary candidate again; the choice was
    # made like any other.
    for key in cleared:
        record.set_state(key, "seen")
    held_until = (date_type.fromisoformat(local_date) + timedelta(days=HOLD_DAYS)).isoformat()
    for entry in scored:
        record.decide(entry.item.key, entry.topic.name, entry.score, entry.outcome)
        if entry.held:
            record.set_state(entry.item.key, "held", topic=entry.topic.name, held_until=held_until)
            result.warnings.append(f"Held back for a week: {entry.item.title} ({entry.held}).")
    for key, why in ranking.left_out.items():
        record.decide(key, "", 0.0, why)
    for key, measures in ranking.measures.items():
        for topic_name, similarity, hits in measures:
            record.scores.append((key, topic_name, current[topic_name], similarity, hits))
    record.vectors.update(ranking.vectors)

    if result.ranking == "keywords":
        # A preview: the items are chosen and described as always, but nothing is saved or sent.
        result.preview = True
        for entry in shown_entries:
            item = entry.item
            what, why, verdict = _note_without_model(entry)
            result.new_reports.append(
                Report(
                    date=local_date,
                    topic=entry.topic.name,
                    score=round(entry.score, 4),
                    source=item.source,
                    kind=item.kind,
                    title=item.title,
                    url=item.url,
                    what=what,
                    why=why,
                    verdict=verdict,
                    meta=item.meta,
                    ranking="keywords",
                )
            )
        result.all_reports = list(result.new_reports)
        progress(f"Done: {len(result.new_reports)} items in a keyword preview")
        return result

    # 3. Describe the chosen items ------------------------------------------------------
    language = str(cfg.settings["summary_language"])
    for number, entry in enumerate(shown_entries, start=1):
        item = entry.item
        progress(f"Writing note {number}/{len(shown_entries)}: {item.title}")
        source_text = item.text
        if item.source == "github" and item.meta.get("full_name"):
            readme = fetch_readme(http, item.meta["full_name"])
            source_text = f"{item.text}\n\n{readme}".strip() if readme else item.text
        if entry.outcome == "watch" and len(item.text.strip()) < WATCH_MIN_TEXT:
            what, why, verdict = watch_note(item)
        elif llm_ready:
            what, why, verdict = summarize(ollama, item, entry.topic, source_text, language)
        else:
            what, why, verdict = fallback_note(item, entry.topic)
        result.new_reports.append(
            Report(
                date=local_date,
                topic=entry.topic.name,
                score=round(entry.score, 4),
                source=item.source,
                kind=item.kind,
                title=item.title,
                url=item.url,
                what=what,
                why=why,
                verdict=verdict,
                meta=item.meta,
                ranking=result.ranking,
            )
        )
        stars = item.meta.get("stars")
        record.set_state(
            item.key, "shown", topic=entry.topic.name, shown_stars=stars if isinstance(stars, int) else None
        )

    # 4. Repositories reported earlier that gained many stars -----------------------------------
    if cfg.source_enabled("github"):
        _add_rising(cfg, http, candidates, record, now, local_date, result, progress)

    # 4b. Follower counts of thin looking repositories that were not chosen -------------
    # `select` asks about a repository only when it would take a place. Replay needs the count
    # of the others in the label band too; their outcome does not change.
    if trust_on:
        _ask_unchosen_owners(http, scored, record)

    # 5. Save --------------------------------------------------------------------
    record.chat_model = str(getattr(ollama, "chat_model", "") or cfg.settings.get("chat_model") or "")
    candidates.save(record)
    day = store.merge_day(
        local_date, result.new_reports, result.skipped, result.warnings, result.ranking, at=now.isoformat()
    )
    result.all_reports = day["reports"]
    last_runs.update(advanced)
    store.save_last_runs(last_runs)
    covered = [] if (pending and (explicit_short or not looked_back)) else pending
    covered_names = {t.name for t in covered}
    store.save_topic_marks(
        {t.name: current[t.name] for t in cfg.topics if marks is None or t not in pending or t.name in covered_names}
    )
    # The note covers every scan of the day, so it takes the notes of earlier scans too.
    result.note_path = write_note(cfg, local_date, day["reports"], day["skipped"], day["warnings"], day["ranking"])

    # 6. Digest file and messages. Nothing here can fail the scan. ------------------------
    try:
        if cfg.delivery["html_file"]:
            result.digest_path = write_digest_file(cfg, local_date, store=store)
        if any(cfg.channel_enabled(c) for c in ("email", "telegram")):
            progress("Sending the digest")
            result.deliveries = deliver_pending(cfg, store, local_date)
    except OSError as exc:
        result.warnings.append(f"Could not write the digest file ({type(exc).__name__}).")
    progress(f"Done: {len(result.new_reports)} new items")
    return result


def _ask_unchosen_owners(http: Http, scored: list[Scored], record: ScanRecord) -> None:
    """Ask the follower count of the owner of each GitHub repository that selection stopped (by the
    bar or a cap), is inside the label band of its topic and looks thin, when the scan has no count
    for that owner yet. At most MAX_UNCHOSEN_OWNERS owners, in the order of `scored`. The count goes
    into the scan's observation row; outcome, state and warnings do not change. A failed request
    leaves the count empty."""
    asked = 0
    for entry in scored:
        if asked >= MAX_UNCHOSEN_OWNERS:
            return
        item = entry.item
        if item.source != "github" or entry.outcome not in ("day_full", "topic_full", "below_bar"):
            continue
        if entry.score < entry.bar - LABEL_BAND:
            continue
        owner = item.meta.get("owner")
        if not owner or owner.lower() in record.followers:
            continue
        if not looks_thin(item.meta.get("size_kb"), item.meta.get("stars")):
            continue
        owner_followers(http, owner, record.followers)
        asked += 1


def _note_without_model(entry: Scored) -> tuple[str, str, str]:
    if entry.outcome == "watch" and len(entry.item.text.strip()) < WATCH_MIN_TEXT:
        return watch_note(entry.item)
    return fallback_note(entry.item, entry.topic)


def _fold_into_watched(record: ScanRecord, item: Item) -> bool:
    """A watched feed gives one item per scan. When the look back for an edited topic finds an
    older entry of a feed the scan already has an item for, count it there. True when it was counted."""
    feed = item.meta.get("watch_topic") and item.meta.get("feed")
    if not feed:
        return False
    for kept in record.items.values():
        if kept.source == item.source and kept.meta.get("watch_topic") and kept.meta.get("feed") == feed:
            fold_earlier(kept, item)
            return True
    return False


def _add_rising(
    cfg: Config,
    http: Http,
    candidates: CandidateStore,
    record: ScanRecord,
    now: datetime,
    local_date: str,
    result: RunResult,
    progress: Progress,
) -> None:
    horizon = (now - timedelta(days=int(cfg.settings["rising_days"]))).astimezone().date().isoformat()
    tracked: dict[str, StoredItem] = {}
    for row in candidates.shown_repos(since=horizon, today=local_date):
        tracked.setdefault(row.full_name, row)
    if not tracked:
        return
    progress("Checking star counts of earlier items")
    check = check_repos(http, list(tracked))
    for full_name in check.gone:
        row = tracked[full_name]
        record.set_state(row.key, "gone")
        result.warnings.append(f"{full_name}, shown on {_day_month(row.state_on)}, has been removed from GitHub since.")
    for full_name, repo in check.current.items():
        row = tracked[full_name]
        record.observe(row.key, repo)
        before = row.shown_stars or 0
        stars_now = repo["stars"]
        if stars_now >= before * 1.5 and stars_now - before >= 50:
            record.update(row.key, shown_stars=stars_now)
            result.new_reports.append(
                Report(
                    date=local_date,
                    topic=row.topic or cfg.topics[0].name,  # the topic it was first reported under
                    score=0.0,
                    source="github",
                    kind="repo",
                    title=full_name,
                    url=row.url,
                    verdict="watch",
                    meta={
                        "stars_before": before,
                        "stars_now": stars_now,
                        "license": repo["license"],
                        "language": repo["language"],
                    },
                    rising=True,
                )
            )


def _recheck_held(
    http: Http,
    candidates: CandidateStore,
    record: ScanRecord,
    local_date: str,
    trust_on: bool,
    result: RunResult,
    progress: Progress,
) -> list[Item]:
    """Ask GitHub again about repositories held for HOLD_DAYS. Removed ones are marked gone, ones
    still thin are dropped for good, and the rest come back as candidates."""
    due = candidates.held_due(local_date)
    if not due:
        return []
    progress("Checking repositories held back earlier")
    back: list[Item] = []
    for row in due:
        name = row.full_name
        try:
            raw = fetch_repo(http, name)
        except SourceError as exc:
            if exc.status in GITHUB_GONE:
                record.set_state(row.key, "gone")
                result.warnings.append(
                    f"{name}, held back on {_day_month(row.state_on)}, has been removed from GitHub since."
                )
                continue
            if exc.status == 403:
                continue  # blocked for now; asked again at the next scan
            break  # rate limit, network or server error: the rest wait for the next scan
        item = parse_repo(raw) if raw else None
        if item is None:
            continue
        meta = item.meta
        record.observe(row.key, meta)
        reason = (
            thin_reason(http, meta.get("size_kb"), meta.get("stars"), meta.get("owner"), record.followers)
            if trust_on
            else None
        )
        if reason is None:
            if item.key != row.key:  # renamed since; the old name is not asked about again
                record.set_state(row.key, "seen")
            back.append(item)
        else:
            record.set_state(row.key, "dropped")
    return back


def _day_month(iso_date: str) -> str:
    """2026-11-04 as "4 November"."""
    try:
        day = date_type.fromisoformat(iso_date)
    except ValueError:
        return iso_date or "an earlier day"
    return f"{day.day} {MONTHS[day.month - 1]}"
