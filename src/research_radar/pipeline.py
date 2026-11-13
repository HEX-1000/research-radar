"""One scan, from the last successful run up to now."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .config import Config, Topic
from .deliver import DeliveryResult, deliver_pending, write_digest_file
from .digest import MONTHS
from .http import Http, SourceError
from .llm import Ollama, OllamaError
from .models import Item, Report, SkippedSource
from .notes import write_note
from .ranker import Scored, rank, select
from .sources import SOURCES, FetchContext
from .sources.github import HOLD_DAYS, check_repos, fetch_readme, fetch_repo, parse_repo, thin_reason
from .sources.github import GONE as GITHUB_GONE
from .store import Store
from .summarize import fallback_note, summarize

Progress = Callable[[str], None]


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


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def window_start(cfg: Config, last_run: str | None, now: datetime, days: int | None) -> datetime:
    """Where a source's window begins: its last good run, capped by max_catchup_days."""
    if days is not None:
        return now - timedelta(days=days)
    earliest = now - timedelta(days=int(cfg.settings["max_catchup_days"]))
    previous = _parse_iso(last_run)
    if previous is None:
        return now - timedelta(days=int(cfg.settings["first_run_days"]))
    return max(previous, earliest)


def topic_fingerprint(topic: Topic) -> str:
    """Changes when the text of a topic or its per source queries change."""
    payload = [topic.description.strip(), topic.keywords, topic.exclude, topic.options]
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
    try:
        return _scan(cfg, store, http, ollama, now, local_date, progress, days, use_llm, result)
    finally:
        if own_http:
            http.close()
        if own_ollama:
            ollama.close()


def _scan(
    cfg: Config,
    store: Store,
    http: Http,
    ollama: Ollama,
    now: datetime,
    local_date: str,
    progress: Progress,
    days: int | None,
    use_llm: bool,
    result: RunResult,
) -> RunResult:
    last_runs = store.last_runs()
    seen = store.seen()

    # Topics that are new or were edited since the last scan get a one time look back, because
    # the window of a source only reaches back to the last scan. The first scan after the
    # upgrade to this behaviour just records the topics, so nothing is fetched twice.
    marks = store.topic_marks()
    current = {t.name: topic_fingerprint(t) for t in cfg.topics}
    look_back = int(cfg.settings["first_run_days"])
    explicit_short = days is not None and days < look_back
    pending = [] if marks is None else [t for t in cfg.topics if marks.get(t.name) != current[t.name]]

    # 1. Fetch -------------------------------------------------------------------
    fetched: dict[str, Item] = {}
    advanced: dict[str, str] = {}
    windows: dict[str, datetime] = {}
    for name, fetch in SOURCES.items():
        if not cfg.source_enabled(name):
            continue
        progress(f"Reading {name}")
        windows[name] = window_start(cfg, last_runs.get(name), now, days)
        ctx = FetchContext(config=cfg, http=http, since=windows[name], until=now)
        try:
            for item in fetch(ctx):
                fetched.setdefault(item.url, item)
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
                    if item.url not in fetched:
                        fetched[item.url] = item
                        only_for_new.add(item.url)
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
    result.fetched = len(fetched)

    fresh = [item for url, item in fetched.items() if url not in seen]

    # Repositories held back a week ago are asked about again; one that is not thin joins the
    # candidates of this scan.
    trust_on = bool(cfg.settings.get("trust_check", True))
    followers: dict[str, int | None] = {}
    cleared: dict[str, str] = {}  # url of a rechecked repository: the held date it had
    if cfg.source_enabled("github"):
        for item, held_on in _recheck_held(cfg, http, seen, local_date, trust_on, followers, result, progress):
            fresh.append(item)
            cleared[item.url] = held_on

    # 2. Rank --------------------------------------------------------------------
    llm_ready = use_llm and ollama.ping()
    likes, dislikes = store.vote_vectors()
    scored: list[Scored] = []

    def rank_all(embed) -> tuple[list[Scored], str]:
        # Items found only by the look back are compared with the new topics alone, so the
        # look back cannot bring up older leftovers for topics that have not changed.
        kept = [i for i in fresh if i.url not in only_for_new]
        extra = [i for i in fresh if i.url in only_for_new]
        options = dict(
            embed=embed,
            global_exclude=cfg.exclude,
            likes=likes if embed else None,
            dislikes=dislikes if embed else None,
            feedback_weight=float(cfg.settings["feedback_weight"]),
        )
        ranked, mode = rank(kept, cfg.topics, **options)
        if extra:
            more, _ = rank(extra, pending, **options)
            ranked = sorted(ranked + more, key=lambda s: s.score, reverse=True)
        return ranked, mode

    if fresh:
        progress(f"Scoring {len(fresh)} items")
        try:
            scored, result.ranking = rank_all(ollama.embed if llm_ready else None)
        except OllamaError as exc:
            result.warnings.append(f"Embedding failed ({exc}); used keyword matching instead.")
            llm_ready = False
            scored, result.ranking = rank_all(None)
    else:
        result.ranking = "embeddings" if llm_ready else "keywords"
    if not llm_ready and use_llm:
        result.warnings.append(
            f"Ollama is not reachable at {cfg.settings['ollama_url']}. Items were matched by keywords and described without a model."
        )

    min_score = float(cfg.settings["min_score"])
    if result.ranking == "keywords":
        min_score = min(min_score, 0.5)  # one keyword hit is the lowest score in this mode
    def hold(entry: Scored) -> str | None:
        item = entry.item
        if not trust_on or item.source != "github" or item.url in cleared:
            return None
        meta = item.meta
        return thin_reason(http, meta.get("size_kb"), meta.get("stars"), meta.get("owner"), followers)

    chosen = select(
        scored,
        min_score=min_score,
        max_items=int(cfg.settings["max_items"]),
        max_per_topic=int(cfg.settings["max_per_topic"]),
        hold=hold,
    )
    result.explain = scored[:30]
    for entry in scored:
        if entry.held:
            item = entry.item
            seen[item.url] = {
                "date": local_date,
                "title": item.title,
                "source": item.source,
                "full_name": item.meta.get("full_name"),
                "stars": item.meta.get("stars"),
                "held": local_date,
                "held_reason": entry.held,
            }
            result.warnings.append(f"Held back for a week: {item.title} ({entry.held}).")
    # A rechecked repository that was not chosen leaves the list; the choice was made like any other.
    for url in cleared:
        if url in seen and seen[url].get("held"):
            del seen[url]

    # 3. Describe the chosen items ------------------------------------------------------
    language = str(cfg.settings["summary_language"])
    vectors = store.vectors()
    for number, entry in enumerate(chosen, start=1):
        item = entry.item
        progress(f"Writing note {number}/{len(chosen)}: {item.title}")
        source_text = item.text
        if item.source == "github" and item.meta.get("full_name"):
            readme = fetch_readme(http, item.meta["full_name"])
            source_text = f"{item.text}\n\n{readme}".strip() if readme else item.text
        if llm_ready:
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
        seen[item.url] = {
            "date": local_date,
            "title": item.title,
            "source": item.source,
            "full_name": item.meta.get("full_name"),
            "stars": item.meta.get("stars"),
        }
        if entry.vector:
            vectors[item.url] = entry.vector

    # 4. Repositories reported earlier that gained many stars -----------------------------------
    if cfg.source_enabled("github"):
        _add_rising(cfg, http, seen, now, local_date, result, progress)

    # 5. Save --------------------------------------------------------------------
    store.save_seen(seen)
    store.save_vectors(vectors)
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


def _add_rising(
    cfg: Config,
    http: Http,
    seen: dict,
    now: datetime,
    local_date: str,
    result: RunResult,
    progress: Progress,
) -> None:
    horizon = (now - timedelta(days=int(cfg.settings["rising_days"]))).astimezone().date().isoformat()
    tracked = {
        meta["full_name"]: url
        for url, meta in seen.items()
        if meta.get("source") == "github"
        and meta.get("full_name")
        and isinstance(meta.get("stars"), int)
        and not meta.get("gone")
        and not meta.get("held")
        and not meta.get("dropped")
        and meta.get("date", "") >= horizon
        and meta.get("date") != local_date
    }
    if not tracked:
        return
    progress("Checking star counts of earlier items")
    check = check_repos(http, list(tracked))
    for full_name in check.gone:
        url = tracked[full_name]
        seen[url]["gone"] = local_date
        result.warnings.append(
            f"{full_name}, shown on {_day_month(seen[url].get('date', ''))}, has been removed from GitHub since."
        )
    for full_name, repo in check.current.items():
        url = tracked[full_name]
        before = seen[url]["stars"]
        stars_now = repo["stars"]
        if stars_now >= before * 1.5 and stars_now - before >= 50:
            seen[url]["stars"] = stars_now
            result.new_reports.append(
                Report(
                    date=local_date,
                    topic=_topic_of(url, result, cfg),
                    score=0.0,
                    source="github",
                    kind="repo",
                    title=full_name,
                    url=url,
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
    cfg: Config,
    http: Http,
    seen: dict,
    local_date: str,
    trust_on: bool,
    followers: dict[str, int | None],
    result: RunResult,
    progress: Progress,
) -> list[tuple[Item, str]]:
    """Ask GitHub again about repositories held for HOLD_DAYS. Removed ones are marked gone, ones
    still thin are dropped for good, and the rest come back as candidates with the date they were held."""
    today = date_type.fromisoformat(local_date)
    due: list[str] = []
    for url, meta in seen.items():
        held = meta.get("held")
        if not held or not meta.get("full_name"):
            continue
        try:
            if date_type.fromisoformat(held) + timedelta(days=HOLD_DAYS) <= today:
                due.append(url)
        except ValueError:
            due.append(url)  # an unreadable date must not hold a repository forever
    if not due:
        return []
    progress("Checking repositories held back earlier")
    back: list[tuple[Item, str]] = []
    for url in due:
        meta = seen[url]
        name = meta["full_name"]
        try:
            raw = fetch_repo(http, name)
        except SourceError as exc:
            if exc.status in GITHUB_GONE:
                meta.pop("held", None)
                meta["gone"] = local_date
                result.warnings.append(
                    f"{name}, held back on {_day_month(meta.get('date', ''))}, has been removed from GitHub since."
                )
                continue
            if exc.status == 403:
                continue  # blocked for now; asked again at the next scan
            break  # rate limit, network or server error: the rest wait for the next scan
        item = parse_repo(raw) if raw else None
        if item is None:
            continue
        item_meta = item.meta
        reason = (
            thin_reason(http, item_meta.get("size_kb"), item_meta.get("stars"), item_meta.get("owner"), followers)
            if trust_on
            else None
        )
        if reason is None:
            back.append((item, meta["held"]))
        else:
            meta.pop("held", None)
            meta["dropped"] = local_date
    return back


def _day_month(iso_date: str) -> str:
    """2026-11-04 as "4 November"."""
    try:
        day = date_type.fromisoformat(iso_date)
    except ValueError:
        return iso_date or "an earlier day"
    return f"{day.day} {MONTHS[day.month - 1]}"


def _topic_of(url: str, result: RunResult, cfg: Config) -> str:
    """Rising items go under the topic they were first reported under, when we can tell."""
    store = Store(cfg.state_dir)
    for date in reversed(store.report_dates()):
        for report in store.load_day(date)["reports"]:
            if report.url == url and not report.rising:
                return report.topic
    return cfg.topics[0].name
