"""The terminal interface: today's items, a detail pane, votes, and a scan key."""

from __future__ import annotations

import webbrowser
from datetime import date as date_type
from datetime import datetime
from typing import Callable

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Static

from .config import Config
from .deliver import describe, write_digest_file
from .models import Report, plain_title
from .notes import meta_line
from .pipeline import RunResult, scan
from .store import Store

VOTE_MARK = {1: "+", -1: "-", 0: " "}
VERDICT_STYLE = {"install": "bold #8BC48A", "read": "bold #FF8C00", "watch": "#FFAA33", "skip": "dim"}
TOPIC_WIDTH = 18


def short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "\u2026"

# Near-black ground with amber accents. Choose another theme with settings.theme.
HEX_THEME = Theme(
    name="hex-amber",
    primary="#FF8C00",
    secondary="#FFAA33",
    accent="#FFAA33",
    foreground="#EDE6DA",
    background="#0A0806",
    surface="#13100C",
    panel="#1B150F",
    success="#8BC48A",
    warning="#FFAA33",
    error="#E5604D",
    dark=True,
)

RADAR_ART = r"""
        .  -  ~  -  .
     .'       |       '.
   .'         |         '.
  /           |     o     \
 |     .      |            |
 |------------+------------|
 |            |  o         |
  \           |           /
   '.         |    .    .'
     '.       |       .'
        ' -  . _ .  - '
""".strip("\n")


def needs_catch_up(cfg: Config, store: Store, today: date_type | None = None) -> bool:
    """True when any enabled source has not had a good scan today."""
    today = today or datetime.now().astimezone().date()
    runs = store.last_runs()
    for name in cfg.sources:
        if not cfg.source_enabled(name):
            continue
        stamp = runs.get(name)
        if not stamp:
            return True
        try:
            last = datetime.fromisoformat(stamp).astimezone().date()
        except ValueError:
            return True
        if last < today:
            return True
    return False


class RadarApp(App):
    TITLE = "Research Radar"
    CSS = """
    #topbar { height: 2; border-bottom: solid $primary; padding: 0 1; }
    #brand { width: 1fr; color: $primary; text-style: bold; height: 1; }
    #daymeta { width: auto; color: $text-muted; height: 1; }
    #body { height: 1fr; }
    #list { width: 55%; overflow-x: hidden; }
    #detail-box { width: 45%; border-left: solid $primary; padding: 0 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    """
    BINDINGS = [
        Binding("s", "scan", "Scan now"),
        Binding("plus", "vote(1)", "+"),
        Binding("minus", "vote(-1)", "-"),
        Binding("o", "open_link", "Open"),
        Binding("d", "open_digest", "Digest page"),
        Binding("left_square_bracket", "day(-1)", "Older day"),
        Binding("right_square_bracket", "day(1)", "Newer day"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, cfg: Config, scan_fn: Callable[..., RunResult] = scan, auto_scan: bool | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.store = Store(cfg.state_dir)
        self.scan_fn = scan_fn
        self.auto_scan = cfg.settings.get("auto_scan_on_launch", True) if auto_scan is None else auto_scan
        self.dates: list[str] = []
        self.day_index = 0
        self.reports: dict[str, Report] = {}
        self.scanning = False

    # layout ---------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("\u250c RESEARCH RADAR", id="brand")
            yield Static("", id="daymeta")
        with Horizontal(id="body"):
            yield DataTable(id="list", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="detail-box"):
                yield Static("", id="detail")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(HEX_THEME)
        try:
            self.theme = str(self.cfg.settings.get("theme") or "hex-amber")
        except Exception:  # an unknown theme name must not stop the app
            self.theme = "hex-amber"
        table = self.query_one("#list", DataTable)
        table.add_column("", width=1)
        table.add_column("Verdict", width=7)
        table.add_column("Topic", width=TOPIC_WIDTH)
        table.add_column("Item")
        self.refresh_dates(jump_to_latest=True)
        self.render_day()
        if self.auto_scan and needs_catch_up(self.cfg, self.store):
            self.action_scan()
        else:
            self.set_status("Press s to scan.")

    # data -----------------------------------------------------------------------

    def refresh_dates(self, jump_to_latest: bool = False) -> None:
        today = datetime.now().astimezone().date().isoformat()
        dates = sorted(set(self.store.report_dates()) | {today})
        self.dates = dates
        if jump_to_latest or self.day_index >= len(dates):
            self.day_index = len(dates) - 1

    @property
    def current_date(self) -> str:
        return self.dates[self.day_index]

    def render_day(self) -> None:
        day = self.store.load_day(self.current_date)
        reports: list[Report] = sorted(day["reports"], key=lambda r: (r.topic, r.rising, -r.score))
        self.reports = {r.url: r for r in reports}
        self.sub_title = self.current_date
        count = len(reports)
        self.query_one("#daymeta", Static).update(
            f"{self.current_date}  \u00b7  {count} item{'s' if count != 1 else ''} \u2518"
        )
        table = self.query_one("#list", DataTable)
        table.clear()
        for report in reports:
            title = f"{plain_title(report.title)} (rising)" if report.rising else plain_title(report.title)
            verdict = Text("rising" if report.rising else report.verdict, style=VERDICT_STYLE.get(report.verdict, ""))
            table.add_row(VOTE_MARK[report.vote], verdict, short(report.topic, TOPIC_WIDTH), title, key=report.url)
        if reports:
            self.show_detail(reports[0])
        else:
            art = Text(RADAR_ART + "\n\n", style="#B35900")
            art.append("Nothing here yet. Press s to scan.", style="bold")
            self.query_one("#detail", Static).update(art)

    def show_detail(self, report: Report) -> None:
        text = Text()
        text.append(plain_title(report.title) + "\n", style="bold")
        text.append(report.url + "\n\n", style="underline")
        text.append(f"{report.topic}  |  {report.source}  |  score {report.score:.2f}\n\n", style="dim")
        if report.rising:
            text.append(f"Rising: {report.meta.get('stars_before')} to {report.meta.get('stars_now')} stars.\n\n")
        else:
            if report.what:
                text.append(report.what + "\n\n")
            if report.why:
                text.append(report.why + "\n\n")
            text.append(f"Recommendation: {report.verdict}\n", style="bold")
        meta = meta_line(report)
        if meta:
            text.append(meta + "\n", style="dim")
        if report.vote:
            text.append("\nYour vote: " + ("+" if report.vote > 0 else "-"), style="bold")
        self.query_one("#detail", Static).update(text)

    def selected(self) -> Report | None:
        table = self.query_one("#list", DataTable)
        if table.row_count == 0:
            return None
        try:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:
            return None
        return self.reports.get(key)

    def set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    # events ---------------------------------------------------------------------

    @on(DataTable.RowHighlighted)
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        report = self.reports.get(event.row_key.value or "")
        if report:
            self.show_detail(report)

    # actions --------------------------------------------------------------------

    def action_day(self, step: int) -> None:
        new_index = self.day_index + step
        if 0 <= new_index < len(self.dates):
            self.day_index = new_index
            self.render_day()

    def action_open_link(self) -> None:
        report = self.selected()
        if report:
            webbrowser.open(report.url)

    def action_open_digest(self) -> None:
        path = write_digest_file(self.cfg, self.current_date, store=self.store)
        webbrowser.open(path.as_uri())
        self.set_status(f"Opened the digest page for {self.current_date}.")

    def action_vote(self, vote: int) -> None:
        report = self.selected()
        if report is None:
            return
        self.store.set_vote(report.date, report.url, vote)
        table = self.query_one("#list", DataTable)
        row = table.cursor_row
        self.render_day()
        if table.row_count:
            table.move_cursor(row=min(row, table.row_count - 1))

    def action_scan(self) -> None:
        if self.scanning:
            self.set_status("A scan is already running.")
            return
        self.scanning = True
        self.set_status("Scanning...")
        self._scan_worker()

    @work(thread=True, exclusive=True, group="scan")
    def _scan_worker(self) -> None:
        try:
            result = self.scan_fn(self.cfg, progress=lambda m: self.call_from_thread(self.set_status, m))
        except Exception as exc:  # the UI must survive any failure in a scan
            self.call_from_thread(self._scan_failed, f"{type(exc).__name__}: {exc}")
        else:
            self.call_from_thread(self._scan_done, result)

    def _scan_failed(self, message: str) -> None:
        self.scanning = False
        self.set_status(f"Scan failed: {message}")

    def _scan_done(self, result: RunResult) -> None:
        self.scanning = False
        self.refresh_dates(jump_to_latest=True)
        self.render_day()
        message = f"Scan done: {len(result.new_reports)} new items."
        if result.skipped:
            message += " Skipped: " + ", ".join(s.source for s in result.skipped) + "."
        for line in describe(result.deliveries):
            message += f" {line}."
        self.set_status(message)
