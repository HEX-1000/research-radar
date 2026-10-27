"""The daily markdown note. It is rebuilt from the day's saved reports."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .models import Report, SkippedSource


def _count(value: object) -> str | None:
    return f"{value:,}" if isinstance(value, int) else None


def meta_line(report: Report) -> str:
    meta = report.meta
    parts: list[str] = []
    if report.kind == "repo":
        stars = _count(meta.get("stars"))
        if stars:
            parts.append(f"{stars} stars")
        parts.append(f"license {meta['license']}" if meta.get("license") else "license not stated")
        if meta.get("language"):
            parts.append(str(meta["language"]))
    elif report.kind == "paper":
        if meta.get("arxiv_id"):
            parts.append(f"arXiv:{meta['arxiv_id']}")
        upvotes = _count(meta.get("upvotes"))
        if upvotes:
            parts.append(f"{upvotes} upvotes")
    elif report.kind == "model":
        if meta.get("task"):
            parts.append(str(meta["task"]))
        likes = _count(meta.get("likes"))
        if likes:
            parts.append(f"{likes} likes")
        downloads = _count(meta.get("downloads"))
        if downloads:
            parts.append(f"{downloads} downloads")
    return ", ".join(parts)


def report_line(report: Report) -> str:
    head = f"- **{report.title}** ({report.url}): "
    if report.rising:
        before = _count(report.meta.get("stars_before")) or "?"
        now = _count(report.meta.get("stars_now")) or "?"
        return f"{head}rising, from {before} to {now} stars since it was first reported."
    sentences = [s for s in (report.what, report.why) if s]
    body = " ".join(s if s.endswith((".", "!", "?")) else s + "." for s in sentences)
    line = f"{head}{body} Recommendation: {report.verdict}."
    meta = meta_line(report)
    return f"{line} ({meta})" if meta else line


def render_note(
    date: str,
    reports: list[Report],
    topic_order: list[str],
    skipped: list[SkippedSource],
    warnings: list[str],
    ranking: str,
) -> str:
    lines = [f"# Radar {date}", ""]
    by_topic: dict[str, list[Report]] = {}
    for report in reports:
        by_topic.setdefault(report.topic, []).append(report)
    names = list(topic_order) + [t for t in by_topic if t not in topic_order]
    for name in names:
        lines.append(f"## {name}")
        entries = sorted(by_topic.get(name, []), key=lambda r: (r.rising, -r.score))
        if entries:
            lines.extend(report_line(r) for r in entries)
        else:
            lines.append("Nothing new today.")
        lines.append("")
    lines.append("## Skipped sources")
    if skipped:
        lines.extend(f"- {s.source}: {s.reason}" for s in skipped)
    else:
        lines.append("- none")
    lines.append("")
    notes = list(warnings)
    if ranking == "keywords":
        notes.insert(0, "Ranked by keyword matching because the embedding model was not available.")
    if notes:
        lines.append("## Notes")
        lines.extend(f"- {n}" for n in notes)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def note_path(cfg: Config, date: str) -> Path:
    year, month, _ = date.split("-")
    return cfg.notes_dir / year / month / f"{date}.md"


def write_note(
    cfg: Config,
    date: str,
    reports: list[Report],
    skipped: list[SkippedSource],
    warnings: list[str],
    ranking: str,
) -> Path:
    path = note_path(cfg, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_note(date, reports, [t.name for t in cfg.topics], skipped, warnings, ranking)
    path.write_text(text, encoding="utf-8")
    return path
