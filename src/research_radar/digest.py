"""The daily digest: the day's reports laid out to be read in a mail or a chat message.

The markdown note is the archive. The digest is the version meant for a person
who has half a minute: items grouped by topic, the ones worth acting on first,
the ones the model marked "skip" folded into one short list at the end.

Titles, links and descriptions come from outside sources, so every piece of text
is escaped before it goes into HTML, and only http and https links are made clickable.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import date as date_type
from urllib.parse import urlparse

from .models import Report, SkippedSource, plain_title
from .notes import meta_line

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

VERDICT_ORDER = {"install": 0, "read": 1, "watch": 2, "skip": 3}
VERDICT_LABEL = {"install": "TRY", "read": "READ", "watch": "WATCH", "skip": "SKIP"}
KIND_LABEL = {"repo": "repository", "paper": "paper", "model": "model"}

REPO_URL = "https://github.com/HEX-1000/research-radar"
TELEGRAM_LIMIT = 4000  # the API allows 4096 characters per message


_QUERY_WARNING = re.compile(r"^(?P<source>[\w-]+): (?:query|topic) '(?P<what>[^']+)': (?P<why>.+)$")


def condense_warnings(warnings: list[str]) -> list[str]:
    """Five searches that failed for the same reason are one line, not five."""
    groups: dict[tuple[str, str], list[str]] = {}
    others: list[str] = []
    for warning in warnings:
        match = _QUERY_WARNING.match(warning)
        if match:
            groups.setdefault((match["source"], match["why"]), []).append(match["what"])
        else:
            others.append(warning)
    lines = []
    for (source, why), names in groups.items():
        noun = "search" if len(names) == 1 else "searches"
        lines.append(f"{source}: {len(names)} {noun} failed ({', '.join(names)}): {why}")
    return lines + others


def safe_url(url: str) -> str:
    parts = urlparse(url or "")
    return url if parts.scheme in ("http", "https") and parts.netloc else ""


def long_date(value: str) -> str:
    try:
        day = date_type.fromisoformat(value)
    except ValueError:
        return value
    return f"{WEEKDAYS[day.weekday()]}, {day.day} {MONTHS[day.month - 1]} {day.year}"


def _sentence(text: str) -> str:
    text = " ".join((text or "").split())
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


@dataclass
class Digest:
    date: str
    reports: list[Report]
    topic_order: list[str] = field(default_factory=list)
    skipped: list[SkippedSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ranking: str = "embeddings"

    # selection and order ----------------------------------------------------------

    @property
    def main(self) -> list[Report]:
        return [r for r in self.reports if r.rising or r.verdict != "skip"]

    @property
    def also_matched(self) -> list[Report]:
        return [r for r in self.reports if not r.rising and r.verdict == "skip"]

    @property
    def worth_your_time(self) -> int:
        return sum(1 for r in self.reports if not r.rising and r.verdict in ("install", "read"))

    def groups(self) -> list[tuple[str, list[Report]]]:
        by_topic: dict[str, list[Report]] = {}
        for report in self.main:
            by_topic.setdefault(report.topic, []).append(report)
        names = [n for n in self.topic_order if n in by_topic] + [n for n in by_topic if n not in self.topic_order]
        groups = []
        for name in names:
            entries = sorted(by_topic[name], key=lambda r: (VERDICT_ORDER.get(r.verdict, 2), r.rising, -r.score))
            groups.append((name, entries))
        return groups

    # wording ------------------------------------------------------------------------

    def summary_line(self) -> str:
        count = len(self.reports)
        if count == 0:
            return "Nothing new today."
        parts = [f"{count} new item{'s' if count != 1 else ''}"]
        if self.worth_your_time:
            parts.append(f"{self.worth_your_time} worth your time")
        topics = {r.topic for r in self.reports}
        parts.append(f"{len(topics)} topic{'s' if len(topics) != 1 else ''}")
        return " · ".join(parts)

    def subject(self) -> str:
        count = len(self.reports)
        if count == 0:
            return f"Research Radar {self.date}: nothing new"
        text = f"Research Radar {self.date}: {count} new item{'s' if count != 1 else ''}"
        if self.worth_your_time:
            text += f", {self.worth_your_time} worth your time"
        return text

    def footer_notes(self) -> list[str]:
        notes = [f"Source {s.source} was skipped: {s.reason}" for s in self.skipped]
        notes.extend(condense_warnings(self.warnings))
        if self.ranking == "keywords":
            notes.append("Ranked by keyword matching because the embedding model was not available.")
        return notes


def _detail_line(report: Report) -> str:
    if report.rising:
        before = report.meta.get("stars_before", "?")
        now = report.meta.get("stars_now", "?")
        return f"Rising: from {before} to {now} stars since it was first reported."
    return ""


def _kind_line(report: Report) -> str:
    kind = KIND_LABEL.get(report.kind, report.kind)
    return f"{kind} · {report.source}"


# plain text -----------------------------------------------------------------------------


def render_text(digest: Digest) -> str:
    lines = [f"RESEARCH RADAR, {long_date(digest.date)}", digest.summary_line(), ""]
    for name, entries in digest.groups():
        lines.append(f"== {name} ({len(entries)}) ==")
        lines.append("")
        for report in entries:
            lines.append(f"[{VERDICT_LABEL.get(report.verdict, report.verdict.upper())}] {plain_title(report.title)}")
            detail = _detail_line(report)
            if detail:
                lines.append(detail)
            else:
                if report.what:
                    lines.append(_sentence(report.what))
                if report.why:
                    lines.append("Why: " + _sentence(report.why))
            meta = meta_line(report)
            if meta:
                lines.append(meta)
            link = safe_url(report.url)
            if link:
                lines.append(link)
            lines.append("")
    also = digest.also_matched
    if also:
        lines.append("Also matched, probably not for you:")
        for report in also:
            link = safe_url(report.url)
            lines.append(f"- {plain_title(report.title)}" + (f" {link}" if link else ""))
        lines.append("")
    notes = digest.footer_notes()
    if notes:
        lines.append("Notes:")
        lines.extend(f"- {n}" for n in notes)
        lines.append("")
    lines.append("Generated on your computer by Research Radar.")
    return "\n".join(lines).rstrip() + "\n"


# Telegram -------------------------------------------------------------------------------


def _tg(text: str) -> str:
    return html.escape(text or "", quote=False)


def _telegram_item(report: Report) -> str:
    title = _tg(plain_title(report.title))
    link = safe_url(report.url)
    head = f'<a href="{html.escape(link, quote=True)}">{title}</a>' if link else title
    lines = [f"<b>{VERDICT_LABEL.get(report.verdict, report.verdict.upper())}</b> · {head}"]
    detail = _detail_line(report)
    if detail:
        lines.append(_tg(detail))
    else:
        if report.what:
            lines.append(_tg(_sentence(report.what)))
        if report.why:
            lines.append("<i>" + _tg(_sentence(report.why)) + "</i>")
    meta = meta_line(report)
    if meta:
        lines.append(_tg(meta))
    return "\n".join(lines)


def render_telegram(digest: Digest) -> list[str]:
    """Telegram messages in HTML mode, split at item boundaries to stay under the length limit."""
    blocks = [f"<b>Research Radar</b> · {_tg(long_date(digest.date))}\n{_tg(digest.summary_line())}"]
    for name, entries in digest.groups():
        blocks.append(f"<b>{_tg(name)}</b> ({len(entries)})")
        blocks.extend(_telegram_item(r) for r in entries)
    also = digest.also_matched
    if also:
        rows = []
        for report in also:
            link = safe_url(report.url)
            title = _tg(plain_title(report.title))
            rows.append(f'- <a href="{html.escape(link, quote=True)}">{title}</a>' if link else f"- {title}")
        blocks.append("Also matched, probably not for you:\n" + "\n".join(rows))
    notes = digest.footer_notes()
    if notes:
        # <code> keeps Telegram from turning host names such as api.github.com into links
        blocks.append("Notes:\n" + "\n".join("- <code>" + _tg(n) + "</code>" for n in notes))

    messages: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > TELEGRAM_LIMIT:
            block = block[: TELEGRAM_LIMIT - 20].rsplit("\n", 1)[0] + "\n(cut)"
        if current and len(current) + 2 + len(block) > TELEGRAM_LIMIT:
            messages.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        messages.append(current)
    return messages


# HTML -----------------------------------------------------------------------------------

# Inline styles carry the light theme because many mail clients ignore <style>.
# The classes let clients that support prefers-color-scheme switch to the dark theme.
CSS = """
@media (prefers-color-scheme: dark) {
  .page { background: #111418 !important; }
  .card { background: #1b2027 !important; border-color: #2c343d !important; }
  .text { color: #e6e9ed !important; }
  .muted { color: #9aa5b1 !important; }
  .link { color: #7db7f5 !important; }
  .accent { color: #FFAA33 !important; }
  .b-install { background: #1d3a2a !important; color: #8fd6a9 !important; }
  .b-read { background: #1c3250 !important; color: #9cc3f2 !important; }
  .b-watch { background: #3d3014 !important; color: #f0c978 !important; }
  .b-skip { background: #2a3036 !important; color: #b0bcc6 !important; }
}
@media (max-width: 480px) { .wrap { padding: 12px 8px !important; } .pad { padding: 14px !important; } }
"""

FONT = "-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
BADGE_STYLE = {
    "install": "background:#e3f4ea;color:#17613a;",
    "read": "background:#e4eefb;color:#1a4f94;",
    "watch": "background:#fdf3dc;color:#7a5200;",
    "skip": "background:#eceff1;color:#455a64;",
}


def _e(text: str) -> str:
    return html.escape(text or "", quote=True)


def _html_card(report: Report) -> str:
    verdict = report.verdict if report.verdict in BADGE_STYLE else "watch"
    label = "RISING" if report.rising else VERDICT_LABEL[verdict]
    title = _e(plain_title(report.title))
    link = safe_url(report.url)
    title_html = (
        f'<a class="link" href="{_e(link)}" style="color:#0b5cad;text-decoration:none;">{title}</a>' if link else title
    )
    parts = [
        f'<span class="b-{verdict}" style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-size:11px;font-weight:700;letter-spacing:.5px;{BADGE_STYLE[verdict]}">{label}</span>'
        f'<span class="muted" style="font-size:12px;color:#616e7c;margin-left:8px;">{_e(_kind_line(report))}</span>',
        f'<div class="text" style="font-size:17px;font-weight:600;line-height:1.35;margin:8px 0 6px;color:#1f2933;">{title_html}</div>',
    ]
    detail = _detail_line(report)
    if detail:
        parts.append(f'<div class="text" style="font-size:14px;line-height:1.5;color:#1f2933;">{_e(detail)}</div>')
    else:
        if report.what:
            parts.append(
                f'<div class="text" style="font-size:14px;line-height:1.5;color:#1f2933;margin-bottom:6px;">{_e(_sentence(report.what))}</div>'
            )
        if report.why:
            parts.append(
                f'<div class="muted" style="font-size:14px;line-height:1.5;color:#616e7c;">{_e(_sentence(report.why))}</div>'
            )
    meta = meta_line(report)
    if meta:
        parts.append(f'<div class="muted" style="font-size:12px;color:#616e7c;margin-top:8px;">{_e(meta)}</div>')
    return (
        '<tr><td style="padding:0 0 12px;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" class="card" '
        'style="background:#ffffff;border:1px solid #e3e6ea;border-radius:8px;">'
        f'<tr><td class="pad" style="padding:16px 18px;font-family:{FONT};">' + "".join(parts) + "</td></tr></table></td></tr>"
    )


def render_html(digest: Digest) -> str:
    rows: list[str] = []
    for name, entries in digest.groups():
        rows.append(
            f'<tr><td class="text" style="padding:14px 2px 8px;font-family:{FONT};font-size:13px;font-weight:700;'
            f'letter-spacing:.6px;text-transform:uppercase;color:#1f2933;">{_e(name)} '
            f'<span class="muted" style="font-weight:400;color:#616e7c;">({len(entries)})</span></td></tr>'
        )
        rows.extend(_html_card(r) for r in entries)
    if not digest.reports:
        rows.append(
            f'<tr><td class="text" style="padding:24px 2px;font-family:{FONT};font-size:15px;color:#1f2933;">'
            "Nothing new matched your interests today.</td></tr>"
        )
    also = digest.also_matched
    if also:
        items = []
        for report in also:
            link = safe_url(report.url)
            title = _e(plain_title(report.title))
            items.append(
                f'<li style="margin:0 0 4px;">'
                + (f'<a class="link" href="{_e(link)}" style="color:#0b5cad;text-decoration:none;">{title}</a>' if link else title)
                + "</li>"
            )
        rows.append(
            f'<tr><td class="muted" style="padding:14px 2px 4px;font-family:{FONT};font-size:13px;color:#616e7c;">'
            f'Also matched, probably not for you:<ul style="margin:6px 0 0;padding-left:18px;">{"".join(items)}</ul></td></tr>'
        )
    notes = digest.footer_notes()
    if notes:
        rows.append(
            f'<tr><td class="muted" style="padding:14px 2px 4px;font-family:{FONT};font-size:12px;color:#616e7c;">'
            "Notes:<ul style=\"margin:4px 0 0;padding-left:18px;\">"
            + "".join(f"<li>{_e(n)}</li>" for n in notes)
            + "</ul></td></tr>"
        )
    preheader = digest.summary_line()
    if digest.groups():
        first = digest.groups()[0][1][0]
        preheader += " · " + plain_title(first.title)
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark"><meta name="supported-color-schemes" content="light dark">'
        f"<title>{_e(digest.subject())}</title><style>{CSS}</style></head>\n"
        '<body class="page" style="margin:0;padding:0;background:#f4f5f7;">'
        f'<div style="display:none;max-height:0;overflow:hidden;opacity:0;">{_e(preheader)}</div>'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" class="page" style="background:#f4f5f7;">'
        '<tr><td align="center" class="wrap" style="padding:24px 12px;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:640px;">'
        f'<tr><td style="padding:14px 2px 6px;font-family:{FONT};border-top:4px solid #FF8C00;">'
        '<div class="accent" style="font-size:12px;font-weight:700;letter-spacing:1px;color:#A85200;">\u250c RESEARCH RADAR</div>'
        f'<div class="text" style="font-size:24px;font-weight:700;line-height:1.3;color:#1f2933;margin-top:4px;">{_e(long_date(digest.date))}</div>'
        f'<div class="muted" style="font-size:14px;color:#616e7c;margin-top:4px;">{_e(digest.summary_line())}</div>'
        "</td></tr>"
        + "".join(rows)
        + f'<tr><td class="muted" style="padding:20px 2px 0;font-family:{FONT};font-size:12px;color:#616e7c;">'
        f'Generated on your computer by <a class="link" href="{REPO_URL}" style="color:#0b5cad;text-decoration:none;">Research Radar</a>.'
        "</td></tr></table></td></tr></table></body></html>\n"
    )
