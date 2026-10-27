"""The terminal interface: today's items, a detail pane, votes, and a scan key."""

from __future__ import annotations

import subprocess
import sys
import webbrowser
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Callable

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Static

from . import __version__
from .config import Config, ConfigError, load_config, uses_example_topics
from .deliver import describe, write_digest_file
from .llm import Ollama, ollama_advice
from .models import Report, plain_title
from .notes import meta_line
from .pipeline import RunResult, scan
from .store import Store

VOTE_MARK = {1: "+", -1: "-", 0: " "}
VERDICT_STYLE = {"install": "bold #8BC48A", "read": "bold #FF8C00", "watch": "#FFAA33", "skip": "dim"}
TOPIC_WIDTH = 18

AMBER = "#FF8C00"
AMBER_LIGHT = "#FFAA33"
GREEN = "#8BC48A"
RED = "#E5604D"
MUTED = "#8C8273"

# The dot next to "ollama" in the top bar. None means the check has not answered yet.
OLLAMA_DOT = {
    None: ("○", MUTED),
    "ok": ("●", GREEN),
    "no-chat": ("●", AMBER_LIGHT),
    "no-embed": ("●", AMBER_LIGHT),
    "down": ("●", RED),
}

CHECKING = "Checking Ollama..."

EXAMPLE_NOTICE = (
    "These are the two example topics from radar init. Press e to describe your own interests "
    "in interests.yaml, save the file, then press s to scan."
)

# Shown where the items would be, so a first-time user sees what to do without the README.
FIRST_KEYS = [("s", "scan now"), ("e", "edit your interests"), ("f1", "all keys"), ("q", "quit")]


def short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"

# Near-black ground with amber accents. Choose another theme with settings.theme.
HEX_THEME = Theme(
    name="hex-amber",
    primary=AMBER,
    secondary=AMBER_LIGHT,
    accent=AMBER_LIGHT,
    foreground="#EDE6DA",
    background="#0A0806",
    surface="#13100C",
    panel="#1B150F",
    success=GREEN,
    warning=AMBER_LIGHT,
    error=RED,
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

# Each line fits the help box of an 80-column terminal.
KEYS = [
    ("s", "scan now"),
    ("+ / -", "useful / not useful (press again to clear)"),
    ("o", "open the link in your browser"),
    ("d", "open the digest page of this day"),
    ("[ / ]", "older day / newer day"),
    ("e", "edit interests.yaml; save it, then press s"),
    ("f1 or ?", "this sheet"),
    ("q", "quit"),
]


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


def default_probe(cfg: Config) -> list[str]:
    """The models Ollama has. Raises OllamaError when Ollama cannot be reached."""
    s = cfg.settings
    client = Ollama(s["ollama_url"], s["embed_model"], s["chat_model"], timeout=5.0)
    try:
        return client.models()
    finally:
        client.close()


def open_in_editor(path: Path) -> None:
    """Open a text file in a separate editor window; the interface keeps running."""
    if sys.platform == "win32":
        subprocess.Popen(["notepad.exe", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-t", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def empty_state(example_topics: bool) -> Text:
    # The words come first: on a small window the picture is what gets cut off.
    text = Text("Nothing here yet.\n\n", style="bold")
    if example_topics:
        text.append(
            "You are using the example topics. Press e to describe your own interests first.\n\n",
            style=AMBER_LIGHT,
        )
    for key, label in FIRST_KEYS:
        text.append(f"{key:<6}", style=f"bold {AMBER}")
        text.append(label + "\n", style="dim")
    text.append("\n" + RADAR_ART, style="#B35900")
    return text


def help_text(cfg: Config) -> Text:
    text = Text()
    text.append(f"Research Radar {__version__}", style=f"bold {AMBER}")
    text.append("   esc closes this sheet\n\n", style="dim")
    for key, label in KEYS:
        text.append(f"  {key:<9}", style=f"bold {AMBER}")
        text.append(label + "\n")
    text.append("\nThe dot next to ollama: ", style="dim")
    text.append("green", style=GREEN)
    text.append(" ready, ", style="dim")
    text.append("amber", style=AMBER_LIGHT)
    text.append(" a model is missing, ", style="dim")
    text.append("red", style=RED)
    text.append(" Ollama is not reachable.\n\n", style="dim")
    text.append("interests  ", style="dim")
    text.append(str(cfg.home / "interests.yaml") + "\n")
    text.append("notes      ", style="dim")
    text.append(str(cfg.notes_dir))
    return text


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,f1,question_mark,q", "app.pop_screen", "close")]
    CSS = """
    HelpScreen { align: center middle; }
    #help { width: 90; max-width: 95%; height: auto; max-height: 90%; border: round $primary; background: $surface; padding: 1 2; }
    #help > Static { width: 100%; height: auto; }
    """

    def __init__(self, body: Text) -> None:
        super().__init__()
        self.body = body

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help"):
            yield Static(self.body)


VOTE = Binding.Group("vote")
DAY = Binding.Group("day")


class RadarApp(App):
    TITLE = "Research Radar"
    ENABLE_COMMAND_PALETTE = False  # nothing to search here, and the footer needs the room
    CSS = """
    #topbar { height: 2; border-bottom: solid $primary; padding: 0 1; }
    #brand { width: 1fr; color: $primary; text-style: bold; height: 1; }
    #ollama { width: auto; height: 1; margin-right: 3; }
    #daymeta { width: auto; color: $text-muted; height: 1; }
    #body { height: 1fr; }
    #list { width: 55%; overflow-x: hidden; }
    #detail-box { width: 45%; border-left: solid $primary; padding: 0 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    """
    BINDINGS = [
        Binding("s", "scan", "scan"),
        Binding("plus", "vote(1)", "useful", group=VOTE),
        Binding("minus", "vote(-1)", "not useful", group=VOTE),
        Binding("o", "open_link", "open"),
        Binding("d", "open_digest", "digest"),
        Binding("left_square_bracket", "day(-1)", "older day", group=DAY),
        Binding("right_square_bracket", "day(1)", "newer day", group=DAY),
        Binding("e", "edit_interests", "interests"),
        Binding("f1", "help", "keys"),
        Binding("question_mark", "help", "keys", show=False),
        Binding("q", "quit", "quit"),
    ]

    def __init__(
        self,
        cfg: Config,
        scan_fn: Callable[..., RunResult] = scan,
        auto_scan: bool | None = None,
        *,
        probe: Callable[[Config], list[str]] | None = None,
        opener: Callable[[Path], None] | None = None,
        notes: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.store = Store(cfg.state_dir)
        self.scan_fn = scan_fn
        self.auto_scan = cfg.settings.get("auto_scan_on_launch", True) if auto_scan is None else auto_scan
        self.probe = probe or default_probe
        self.opener = opener or open_in_editor
        self.startup_notes = list(notes or [])
        self.dates: list[str] = []
        self.day_index = 0
        self.reports: dict[str, Report] = {}
        self.scanning = False
        self.ollama_state: str | None = None
        self.status_text = ""
        self.example_topics = uses_example_topics(cfg)
        self._auto_scan_pending = False

    # layout ---------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("┌ RESEARCH RADAR", id="brand")
            yield Static("", id="ollama")
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
        self.refresh_ollama_dot()
        for text, severity in self.startup_notes:
            self.notify(text, severity=severity, timeout=20 if "\n" in text else 10)
        if self.example_topics:
            self.notify(EXAMPLE_NOTICE, severity="warning", timeout=20)
        # The catch-up scan waits for the Ollama check: without Ollama it would spend today's
        # window on keyword matching. With the example topics it would scan someone else's interests.
        self._auto_scan_pending = bool(
            self.auto_scan and not self.example_topics and needs_catch_up(self.cfg, self.store)
        )
        self.set_status(CHECKING)
        self.run_worker(self._check_ollama, thread=True, group="ping")

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
            f"{self.current_date}  ·  {count} item{'s' if count != 1 else ''} ┘"
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
            self.query_one("#detail", Static).update(empty_state(self.example_topics))

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
        self.status_text = message
        self.query_one("#status", Static).update(message)

    def idle_status(self) -> str:
        if self.example_topics:
            return "Example topics in use: press e to write your own, save the file, then press s."
        if self.ollama_state == "down":
            return "Ollama is not reachable (see the notice). Press s to scan with keyword matching."
        if self.ollama_state == "no-embed":
            return "The embedding model is missing (see the notice). Press s to scan with keyword matching."
        return "Press s to scan."

    # Ollama ---------------------------------------------------------------------

    def refresh_ollama_dot(self) -> None:
        mark, color = OLLAMA_DOT.get(self.ollama_state, OLLAMA_DOT[None])
        dot = Text(mark + " ", style=color)
        dot.append("ollama", style=MUTED)
        self.query_one("#ollama", Static).update(dot)

    def _check_ollama(self, quiet: bool = False) -> None:
        """Runs in a thread: ask Ollama for its models. With quiet=True only the dot changes."""
        try:
            names = self.probe(self.cfg)
        except Exception:  # not reachable, or an answer we cannot read
            names = None
        self.call_from_thread(self._ollama_result, names, quiet)

    def _recheck_ollama(self) -> None:
        self.run_worker(lambda: self._check_ollama(quiet=True), thread=True, group="ping")

    def _ollama_result(self, names: list[str] | None, quiet: bool = False) -> None:
        state, messages = ollama_advice(self.cfg.settings, names)
        self.ollama_state = state
        self.refresh_ollama_dot()
        if quiet:
            return
        for text, severity in messages:
            self.notify(text, severity=severity, timeout=15)
        if self._auto_scan_pending:
            self._auto_scan_pending = False
            if state in ("ok", "no-chat"):
                self.action_scan()
                return
        if not self.scanning and self.status_text == CHECKING:  # do not hide a newer message
            self.set_status(self.idle_status())

    # events ---------------------------------------------------------------------

    @on(DataTable.RowHighlighted)
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        report = self.reports.get(event.row_key.value or "")
        if report:
            self.show_detail(report)

    # actions --------------------------------------------------------------------

    def action_help(self) -> None:
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen(help_text(self.cfg)))

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

    def action_edit_interests(self) -> None:
        path = self.cfg.home / "interests.yaml"
        if not path.exists():
            self.set_status(f"There is no {path}. Run radar init first.")
            return
        try:
            self.opener(path)
        except OSError as exc:
            self.set_status(f"Could not open an editor ({type(exc).__name__}). The file is {path}")
            return
        self.set_status("Opened interests.yaml. Save it, then press s to scan with your changes.")

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

    def _reload_config(self) -> bool:
        """Read interests.yaml again, so edits made while the app is open count. False on an error."""
        if not (self.cfg.home / "interests.yaml").exists():
            return True  # a configuration built in code (tests); nothing to reload
        try:
            self.cfg = load_config(self.cfg.home)
        except ConfigError as exc:
            self.notify(f"interests.yaml has a problem, so nothing was scanned. {exc}", severity="error", timeout=15)
            self.set_status("Fix interests.yaml, save it, then press s.")
            return False
        self.example_topics = uses_example_topics(self.cfg)
        return True

    def action_scan(self) -> None:
        self._auto_scan_pending = False  # a scan started by hand replaces the one on launch
        if self.scanning:
            self.set_status("A scan is already running.")
            return
        if not self._reload_config():
            return
        if self.ollama_state != "ok":
            self._recheck_ollama()  # Ollama may have been started since the last check
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
        self._recheck_ollama()

    def _scan_done(self, result: RunResult) -> None:
        self.scanning = False
        self.refresh_dates(jump_to_latest=True)
        self.render_day()
        message = f"Scan done: {len(result.new_reports)} new items."
        if result.ranking == "keywords":
            message += " Matched by keywords, without Ollama."
        if result.backfilled:
            message += " Looked back for: " + ", ".join(result.backfilled) + "."
        if result.skipped:
            message += " Skipped: " + ", ".join(s.source for s in result.skipped) + "."
        for line in describe(result.deliveries):
            message += f" {line}."
        self.set_status(message)
        if self.ollama_state != "ok":
            self._recheck_ollama()
