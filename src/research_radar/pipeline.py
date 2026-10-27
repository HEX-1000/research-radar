"""One scan, from the last successful run up to now."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .config import Config
from .http import Http, SourceError
from .llm import Ollama, OllamaError
from .models import Item, Report, SkippedSource
from .notes import write_note
from .ranker import Scored, rank, select
from .sources import SOURCES, FetchContext
from .sources.github import fetch_readme, star_counts
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
    fetched: int = 0
    explain: list[Scored] = field(default_factory=list)


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

    # 1. Fetch -------------------------------------------------------------------
    fetched: dict[str, Item] = {}
    advanced: dict[str, str] = {}
    for name, fetch in SOURCES.items():
        if not cfg.source_enabled(name):
            continue
        progress(f"Reading {name}")
        ctx = FetchContext(
            config=cfg,
            http=http,
            since=window_start(cfg, last_runs.get(name), now, days),
            until=now,
        )
        try:
            for item in fetch(ctx):
                fetched.setdefault(item.url, item)
        except SourceError as exc:
            result.skipped.append(SkippedSource(name, str(exc)))
            continue
        result.warnings.extend(ctx.warnings)
        if not ctx.warnings:  # a source with failed requests keeps its old window
            advanced[name] = now.isoformat()
    result.fetched = len(fetched)

    fresh = [item for url, item in fetched.items() if url not in seen]

    # 2. Rank --------------------------------------------------------------------
    llm_ready = use_llm and ollama.ping()
    likes, dislikes = store.vote_vectors()
    scored: list[Scored] = []
    if fresh:
        progress(f"Scoring {len(fresh)} items")
        embed = ollama.embed if llm_ready else None
        try:
            scored, result.ranking = rank(
                fresh,
                cfg.topics,
                embed=embed,
                global_exclude=cfg.exclude,
                likes=likes,
                dislikes=dislikes,
                feedback_weight=float(cfg.settings["feedback_weight"]),
            )
        except OllamaError as exc:
            result.warnings.append(f"Embedding failed ({exc}); used keyword matching instead.")
            llm_ready = False
            scored, result.ranking = rank(fresh, cfg.topics, embed=None, global_exclude=cfg.exclude)
    else:
        result.ranking = "embeddings" if llm_ready else "keywords"
    if not llm_ready and use_llm:
        result.warnings.append(
            f"Ollama is not reachable at {cfg.settings['ollama_url']}. Items were matched by keywords and described without a model."
        )

    min_score = float(cfg.settings["min_score"])
    if result.ranking == "keywords":
        min_score = min(min_score, 0.5)  # one keyword hit is the lowest score in this mode
    chosen = select(
        scored,
        min_score=min_score,
        max_items=int(cfg.settings["max_items"]),
        max_per_topic=int(cfg.settings["max_per_topic"]),
    )
    result.explain = scored[:30]

    # 3. Describe the chosen items ------------------------------------------------------
    language = str(cfg.settings["summary_language"])
    vectors = store.vectors()
    for number, entry in enumerate(chosen, start=1):
        item = entry.item
        progress(f"Writing note {number}/{len(chosen)}: {item.title}")
        source_text = item.text
        if item.source == "github" and item.meta.get("full_name"):
            source_text = fetch_readme(http, item.meta["full_name"]) or item.text
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
    result.all_reports = store.merge_day(
        local_date, result.new_reports, result.skipped, result.warnings, result.ranking
    )
    last_runs.update(advanced)
    store.save_last_runs(last_runs)
    result.note_path = write_note(
        cfg, local_date, result.all_reports, result.skipped, result.warnings, result.ranking
    )
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
        and meta.get("date", "") >= horizon
        and meta.get("date") != local_date
    }
    if not tracked:
        return
    progress("Checking star counts of earlier items")
    current = star_counts(http, list(tracked))
    for full_name, stars_now in current.items():
        url = tracked[full_name]
        before = seen[url]["stars"]
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
                    meta={"stars_before": before, "stars_now": stars_now},
                    rising=True,
                )
            )


def _topic_of(url: str, result: RunResult, cfg: Config) -> str:
    """Rising items go under the topic they were first reported under, when we can tell."""
    store = Store(cfg.state_dir)
    for date in reversed(store.report_dates()):
        for report in store.load_day(date)["reports"]:
            if report.url == url and not report.rising:
                return report.topic
    return cfg.topics[0].name
