"""Command line: `radar` opens the TUI, the other commands work without it."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from . import __version__
from .config import Config, ConfigError, init_home, load_config, resolve_home
from .deliver import DeliveryError, describe
from .http import Http, SourceError
from .llm import Ollama, OllamaError, ollama_advice

# Set for the copy started right after an update, so it shows a notice and does not check again.
UPDATED_ENV = "RADAR_UPDATED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radar",
        description="Local-first research radar. Run without a command to open the terminal interface.",
    )
    parser.add_argument("--home", help="data folder (default: $RADAR_HOME or ~/.research-radar)")
    parser.add_argument("--version", action="version", version=f"research-radar {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("tui", help="open the terminal interface (default)")
    sub.add_parser("init", help="create the data folder and a starter interests.yaml")

    scan_p = sub.add_parser("scan", help="scan now without the interface (for schedulers)")
    scan_p.add_argument("--days", type=int, help="look back this many days instead of since the last scan")
    scan_p.add_argument("--no-llm", action="store_true", help="skip Ollama: keyword matching and plain descriptions")
    scan_p.add_argument("--explain", action="store_true", help="print the scores of the top 30 candidates")

    send_p = sub.add_parser("send", help="send the digest by the channels enabled in interests.yaml")
    send_p.add_argument("--date", help="send a whole past day again (YYYY-MM-DD)")
    send_p.add_argument("--dry-run", action="store_true", help="write the digest page and show what would be sent, send nothing")
    send_p.add_argument("--test", action="store_true", help="send a short test message through each enabled channel")

    sub.add_parser("shortcut", help="put a Research Radar icon on the Windows desktop")
    sub.add_parser("telegram-chat", help="find your Telegram chat id (write to your bot first)")
    doctor_p = sub.add_parser("doctor", help="check Ollama, every source and the delivery channels on this machine")
    doctor_p.add_argument("--local", action="store_true", help="check only Ollama and its models, nothing on the internet")
    sub.add_parser("update", help="install a newer version from GitHub now (the interface also checks when it opens)")
    return parser


def _load(home_arg: str | None) -> Config:
    return load_config(resolve_home(home_arg))


def cmd_init(home_arg: str | None) -> int:
    home = resolve_home(home_arg)
    target = init_home(home)
    print(f"Data folder: {home}")
    print(f"Edit your interests here: {target}")
    print("Then install the local models (once):")
    print("  ollama pull nomic-embed-text")
    print("  ollama pull <a chat model you like>")
    print("Check everything with `radar doctor`, then run `radar`.")
    return 0


def cmd_scan(cfg: Config, days: int | None, no_llm: bool, explain: bool) -> int:
    from .pipeline import scan

    result = scan(cfg, days=days, use_llm=not no_llm, progress=lambda m: print(m, flush=True))
    print()
    print(f"{len(result.new_reports)} new items ({result.fetched} fetched). Note: {result.note_path}")
    for skipped in result.skipped:
        print(f"skipped {skipped.source}: {skipped.reason}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    if result.digest_path:
        print(f"Digest page: {result.digest_path}")
    for line in describe(result.deliveries):
        print(line)
    if explain:
        print()
        print("score  hits  chosen  topic / item")
        for entry in result.explain:
            mark = "yes" if entry.selected else "no"
            print(f"{entry.score:5.2f}  {entry.keyword_hits:4d}  {mark:>6}  {entry.topic.name} / {entry.item.title[:70]}")
    return 0


def cmd_send(cfg: Config, date: str | None, dry_run: bool, test: bool) -> int:
    from .deliver import deliver_pending, send_test, write_digest_file
    from .store import Store

    if not any(cfg.channel_enabled(c) for c in ("email", "telegram")):
        print("No delivery channel is enabled. Turn one on in the delivery section of interests.yaml.")
        return 1
    if test:
        results = send_test(cfg)
    else:
        store = Store(cfg.state_dir)
        today = datetime.now().astimezone().date().isoformat()
        if dry_run:
            print(f"Digest page: {write_digest_file(cfg, date or today, store=store)}")
        results = deliver_pending(cfg, store, today, only_date=date, dry_run=dry_run)
    for line in describe(results):
        print(line)
    return 0 if all(r.ok for r in results) else 1


def cmd_shortcut(home_arg: str | None) -> int:
    from .shortcut import ShortcutError, create_shortcut

    home = resolve_home(home_arg)
    try:
        path = create_shortcut(home)
    except ShortcutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Created {path}")
    print("Double-click it to open Research Radar. To keep it on the taskbar, right-click it and choose Pin to taskbar.")
    return 0


def cmd_telegram_chat(cfg: Config) -> int:
    from .deliver import find_telegram_chats

    try:
        chats = find_telegram_chats(cfg)
    except DeliveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not chats:
        print("No messages found. Open your bot in Telegram, press Start, send it any text, then run this again.")
        return 1
    for chat_id, name in chats:
        print(f"{chat_id}  {name}")
    print("Put the number in delivery.telegram.chat_id in interests.yaml.")
    return 0


def _probe(label: str, fn) -> bool:
    try:
        detail = fn()
    except (SourceError, OllamaError, DeliveryError) as exc:
        print(f"FAIL  {label}: {exc}")
        return False
    except Exception as exc:  # a changed response shape is a failure worth reporting, not a crash
        print(f"FAIL  {label}: unexpected response ({type(exc).__name__}: {exc})")
        return False
    print(f"ok    {label}: {detail}")
    return True


def cmd_doctor(cfg: Config, local_only: bool = False) -> int:
    from .sources import arxiv, forgejo, github, huggingface

    results: list[bool] = []
    s = cfg.settings
    ollama = Ollama(s["ollama_url"], s["embed_model"], s["chat_model"])

    def check_ollama() -> str:
        try:
            names = ollama.models()
        except OllamaError:
            names = None
        _, advice = ollama_advice(s, names, retry_hint="run radar doctor again")
        if advice:
            raise OllamaError("\n      ".join(text for text, _ in advice))
        chat = ollama.resolve_chat_model()
        vector = ollama.embed(["test"], "query")[0]
        return f"embedding model ok ({len(vector)} dimensions), chat model '{chat}'"

    results.append(_probe(f"ollama at {s['ollama_url']}", check_ollama))
    ollama.close()
    if local_only:
        return 0 if all(results) else 1

    http = Http(github_token=cfg.github_token())
    now = datetime.now(timezone.utc)
    long_ago = now - timedelta(days=3650)
    try:
        if cfg.source_enabled("github"):
            def probe_github() -> str:
                data = http.get_json(
                    f"{github.API}/search/repositories",
                    params={"q": "stars:>1000", "per_page": 1},
                    headers=github.HEADERS,
                )
                repos = [r for r in (github.parse_repo(x) for x in data.get("items", [])) if r]
                if not repos:
                    raise SourceError("search worked but no repository could be read")
                rate = http.get_json(f"{github.API}/rate_limit", headers=github.HEADERS)
                search = rate.get("resources", {}).get("search", {})
                token = "token set" if http.github_token else "no token, set GITHUB_TOKEN for higher limits"
                return f"read {repos[0].title}; search requests left {search.get('remaining')}/{search.get('limit')} ({token})"

            results.append(_probe("github", probe_github))
        if cfg.source_enabled("arxiv"):
            def probe_arxiv() -> str:
                response = http.get(
                    arxiv.API,
                    params={"search_query": "cat:cs.IR", "max_results": 1, "sortBy": "submittedDate"},
                    bucket="arxiv",
                    min_interval=3.2,
                )
                items = arxiv.parse_feed(response.text)
                if not items:
                    raise SourceError("feed parsed but had no entries")
                return f"read {items[0].url}"

            results.append(_probe("arxiv", probe_arxiv))
        if cfg.source_enabled("huggingface-papers"):
            def probe_papers() -> str:
                items = huggingface.parse_papers(http.get_json(f"{huggingface.API}/daily_papers", params={"limit": 3}))
                if not items:
                    raise SourceError("response was a list but no paper could be read")
                return f"read {items[0].url}"

            results.append(_probe("huggingface-papers", probe_papers))
        if cfg.source_enabled("huggingface-models"):
            def probe_models() -> str:
                items = huggingface.parse_models(
                    http.get_json(
                        f"{huggingface.API}/models",
                        params={"sort": "trendingScore", "direction": "-1", "limit": 3},
                    )
                )
                if not items:
                    raise SourceError("response was a list but no model could be read")
                return f"read {items[0].url}"

            results.append(_probe("huggingface-models", probe_models))
        if cfg.source_enabled("forgejo"):
            for instance in cfg.sources["forgejo"].get("instances", []):
                def probe_forgejo(instance=instance) -> str:
                    data = http.get_json(
                        f"https://{instance}/api/v1/repos/search", params={"q": "python", "limit": 3}
                    )
                    items = forgejo.parse_repos(data, instance, long_ago)
                    if not items:
                        raise SourceError("search worked but no repository could be read")
                    return f"read {items[0].url}"

                results.append(_probe(f"forgejo ({instance})", probe_forgejo))
    finally:
        http.close()
    from .deliver import check_email, check_telegram

    if cfg.channel_enabled("email"):
        results.append(_probe("email", lambda: check_email(cfg)))
    if cfg.channel_enabled("telegram"):
        results.append(_probe("telegram", lambda: check_telegram(cfg)))
    return 0 if all(results) else 1


def cmd_update() -> int:
    from .update import check_and_update
    from .update import describe as describe_update

    result = check_and_update()
    print(describe_update(result))
    return 0 if result.status in ("current", "updated", "offline") else 1


def auto_update(argv: list[str]) -> tuple[int | None, list[tuple[str, str]]]:
    """Look for a newer version before the interface opens (see update.py).

    Returns an exit code when an update was installed and the new copy already ran in this
    process's place, else None, plus notices for the interface. Never raises."""
    try:
        from .update import check_and_update

        result = check_and_update()
    except Exception:
        return None, []
    if result.status == "updated":
        label = f"{result.old} to {result.new}"
        print(f"Research Radar: updated {label}. Starting the new version...", flush=True)
        env = dict(os.environ, **{UPDATED_ENV: label})
        return subprocess.call([sys.executable, "-m", "research_radar", *argv], env=env), []
    if result.status == "available":
        return None, [(f"Research Radar {result.new} is available. Update with: {result.message}", "information")]
    if result.status in ("failed", "skipped") and result.message:
        return None, [(f"Update check: {result.message}. Still on {result.old}.", "warning")]
    return None, []


def cmd_tui(cfg: Config, notes: list[tuple[str, str]] | None = None) -> int:
    from .tui import RadarApp

    RadarApp(cfg, notes=notes).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    command = args.command or "tui"
    try:
        if command == "init":
            return cmd_init(args.home)
        if command == "shortcut":
            return cmd_shortcut(args.home)
        if command == "update":
            return cmd_update()
        cfg = _load(args.home)
        if command == "scan":
            return cmd_scan(cfg, args.days, args.no_llm, args.explain)
        if command == "doctor":
            return cmd_doctor(cfg, local_only=args.local)
        if command == "send":
            return cmd_send(cfg, args.date, args.dry_run, args.test)
        if command == "telegram-chat":
            return cmd_telegram_chat(cfg)
        notes: list[tuple[str, str]] = []
        updated = os.environ.pop(UPDATED_ENV, "")
        if updated:  # this is the copy started right after an update
            notes.append((f"Research Radar updated ({updated}).", "information"))
        elif cfg.settings.get("auto_update", True):
            code, notes = auto_update(argv)
            if code is not None:
                return code
        return cmd_tui(cfg, notes)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
