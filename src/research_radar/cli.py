"""Command line: `radar` opens the TUI, the other commands work without it."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from . import __version__
from .config import Config, ConfigError, init_home, load_config, resolve_home
from .http import Http, SourceError
from .llm import Ollama, OllamaError


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

    sub.add_parser("doctor", help="check Ollama and every source on this machine")
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
    if explain:
        print()
        print("score  hits  chosen  topic / item")
        for entry in result.explain:
            mark = "yes" if entry.selected else "no"
            print(f"{entry.score:5.2f}  {entry.keyword_hits:4d}  {mark:>6}  {entry.topic.name} / {entry.item.title[:70]}")
    return 0


def _probe(label: str, fn) -> bool:
    try:
        detail = fn()
    except (SourceError, OllamaError) as exc:
        print(f"FAIL  {label}: {exc}")
        return False
    except Exception as exc:  # a changed response shape is a failure worth reporting, not a crash
        print(f"FAIL  {label}: unexpected response ({type(exc).__name__}: {exc})")
        return False
    print(f"ok    {label}: {detail}")
    return True


def cmd_doctor(cfg: Config) -> int:
    from .sources import arxiv, forgejo, github, huggingface

    results: list[bool] = []
    s = cfg.settings
    ollama = Ollama(s["ollama_url"], s["embed_model"], s["chat_model"])

    def check_ollama() -> str:
        names = ollama.models()
        if not any(n == s["embed_model"] or n.startswith(s["embed_model"] + ":") for n in names):
            raise OllamaError(f"embedding model '{s['embed_model']}' is not installed (ollama pull {s['embed_model']})")
        chat = ollama.resolve_chat_model()
        vector = ollama.embed(["test"], "query")[0]
        return f"embedding model ok ({len(vector)} dimensions), chat model '{chat}'"

    results.append(_probe(f"ollama at {s['ollama_url']}", check_ollama))
    ollama.close()

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
    return 0 if all(results) else 1


def cmd_tui(cfg: Config) -> int:
    from .tui import RadarApp

    RadarApp(cfg).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "tui"
    try:
        if command == "init":
            return cmd_init(args.home)
        cfg = _load(args.home)
        if command == "scan":
            return cmd_scan(cfg, args.days, args.no_llm, args.explain)
        if command == "doctor":
            return cmd_doctor(cfg)
        return cmd_tui(cfg)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
