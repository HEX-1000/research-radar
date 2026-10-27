"""What a new user sees: the empty screen, the Ollama and model notices, the example topics, the keys."""

from __future__ import annotations

import httpx
import pytest
import yaml
from textual.widgets import DataTable, Static
from textual.widgets._footer import FooterKey

from research_radar import cli, llm, tui
from research_radar.config import example_text, parse_config, uses_example_topics
from research_radar.llm import Ollama, OllamaError, has_model, ollama_advice
from research_radar.pipeline import RunResult

OWN_TOPICS = [
    {"name": "Teaching videos", "description": "Code-driven animation for computer science lessons."},
    {"name": "Terrain", "description": "Elevation models and satellite imagery of hills and valleys."},
]
EMBED = "nomic-embed-text:latest"


def config(tmp_path, own: bool = True, write: bool = False, **settings):
    data = yaml.safe_load(example_text())
    data["settings"].update(settings)
    if own:
        data["topics"] = OWN_TOPICS
    home = tmp_path / "home"
    if write:
        home.mkdir(parents=True, exist_ok=True)
        (home / "interests.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return parse_config(data, home)


def has(*names):
    return lambda cfg: list(names)


def down(cfg):
    raise OllamaError("connection refused")


def notices(app):
    return [(n.severity, n.message) for n in app._notifications]


def status(app) -> str:
    return str(app.query_one("#status", Static).render())


class ScanSpy:
    def __init__(self, ranking: str = "embeddings") -> None:
        self.configs = []
        self.ranking = ranking

    def __call__(self, cfg, progress=lambda m: None, **kwargs):
        self.configs.append(cfg)
        return RunResult(date="2026-10-27", ranking=self.ranking)


@pytest.fixture(autouse=True)
def ollama_is_installed(monkeypatch):
    monkeypatch.setattr(llm, "ollama_installed", lambda: True)


# --------------------------------------------------------------------------- the advice itself


def test_model_names_follow_ollama_tags():
    assert has_model(["nomic-embed-text:latest"], "nomic-embed-text")
    assert has_model(["qwen3.5:9b"], "qwen3.5:9b")
    assert not has_model(["qwen3.5:4b"], "qwen3.5:9b")
    assert not has_model(["nomic-embed-text:v1.5"], "nomic-embed-text")  # Ollama would look for :latest
    assert not has_model(["x:latest"], "")


def test_advice_names_the_exact_command():
    settings = {"ollama_url": "http://localhost:11434", "embed_model": "nomic-embed-text", "chat_model": ""}
    assert ollama_advice(settings, [EMBED, "llama3:8b"]) == ("ok", [])
    state, messages = ollama_advice(settings, ["llama3:8b"])
    assert state == "no-embed" and messages[0][0].endswith("ollama pull nomic-embed-text")
    state, messages = ollama_advice(settings, [EMBED])
    assert state == "no-chat" and "ollama pull " + llm.SUGGESTED_CHAT_MODEL in messages[0][0]
    state, messages = ollama_advice({**settings, "chat_model": "qwen3.5:9b"}, [EMBED, "qwen3.5:4b"])
    assert state == "no-chat"
    assert "Set chat_model to qwen3.5:4b" in messages[0][0] and messages[0][0].endswith("ollama pull qwen3.5:9b")
    state, messages = ollama_advice(settings, None, installed=True)
    assert state == "down" and "not reachable at http://localhost:11434" in messages[0][0]
    state, messages = ollama_advice(settings, None, installed=False)
    assert state == "down" and "https://ollama.com/download" in messages[0][0]


def test_a_remote_ollama_is_never_called_not_installed(monkeypatch):
    monkeypatch.setattr(llm, "ollama_installed", lambda: False)
    settings = {"ollama_url": "http://192.168.1.20:11434", "embed_model": "x", "chat_model": ""}
    assert "not reachable" in ollama_advice(settings, None)[1][0][0]


# --------------------------------------------------------------------------- the interface


async def test_empty_screen_lists_the_first_keys(tmp_path):
    app = tui.RadarApp(config(tmp_path), auto_scan=False, probe=has(EMBED, "llama3:8b"))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        text = str(app.query_one("#detail", Static).render())
        for key, label in tui.FIRST_KEYS:
            assert label in text
        assert "example topics" not in text


async def test_example_topics_get_a_notice_and_no_scan_on_launch(tmp_path):
    spy = ScanSpy()
    cfg = config(tmp_path, own=False)
    assert uses_example_topics(cfg)
    app = tui.RadarApp(cfg, scan_fn=spy, auto_scan=True, probe=has(EMBED, "llama3:8b"))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        assert spy.configs == []
        assert any("example topics" in m and "press e" in m.lower() for _, m in notices(app))
        assert "press e" in status(app)
        assert "example topics" in str(app.query_one("#detail", Static).render())


async def test_with_own_topics_and_every_model_the_catch_up_scan_starts_quietly(tmp_path):
    spy = ScanSpy()
    app = tui.RadarApp(config(tmp_path), scan_fn=spy, auto_scan=True, probe=has(EMBED, "llama3:8b"))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        await app.workers.wait_for_complete()
        assert len(spy.configs) == 1
        assert notices(app) == []
        assert app.ollama_state == "ok"


async def test_ollama_down_gives_an_error_a_red_dot_and_no_scan_on_launch(tmp_path):
    spy = ScanSpy()
    app = tui.RadarApp(config(tmp_path), scan_fn=spy, auto_scan=True, probe=down)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        assert app.ollama_state == "down"
        errors = [m for sev, m in notices(app) if sev == "error"]
        assert len(errors) == 1 and "not reachable" in errors[0] and "press s" in errors[0]
        assert spy.configs == []
        assert "keyword" in status(app)
        assert tui.OLLAMA_DOT["down"][0] in str(app.query_one("#ollama", Static).render())


async def test_ollama_not_installed_points_to_the_download(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "ollama_installed", lambda: False)
    app = tui.RadarApp(config(tmp_path), auto_scan=False, probe=down)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        assert any("https://ollama.com/download" in m for _, m in notices(app))


async def test_a_missing_embedding_model_names_the_pull_command_and_waits(tmp_path):
    spy = ScanSpy()
    app = tui.RadarApp(config(tmp_path), scan_fn=spy, auto_scan=True, probe=has("llama3:8b"))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        warnings = [m for sev, m in notices(app) if sev == "warning"]
        assert len(warnings) == 1 and warnings[0].endswith("ollama pull nomic-embed-text")
        assert app.ollama_state == "no-embed" and spy.configs == []


async def test_a_missing_chat_model_warns_but_the_scan_still_runs(tmp_path):
    spy = ScanSpy()
    cfg = config(tmp_path, chat_model="qwen3.5:9b")
    app = tui.RadarApp(cfg, scan_fn=spy, auto_scan=True, probe=has(EMBED, "qwen3.5:4b"))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        await app.workers.wait_for_complete()
        warnings = [m for sev, m in notices(app) if sev == "warning"]
        assert len(warnings) == 1
        assert "ollama pull qwen3.5:9b" in warnings[0] and "qwen3.5:4b" in warnings[0]
        assert len(spy.configs) == 1


async def test_the_dot_turns_green_after_ollama_is_started_late_without_new_notices(tmp_path):
    probes = {"up": False}

    def probe(cfg):
        if not probes["up"]:
            raise OllamaError("refused")
        return [EMBED, "llama3:8b"]

    app = tui.RadarApp(config(tmp_path), scan_fn=ScanSpy(), auto_scan=False, probe=probe)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        assert app.ollama_state == "down"
        count = len(notices(app))
        probes["up"] = True
        await pilot.press("s")  # a scan checks Ollama again, quietly
        await app.workers.wait_for_complete()
        await pilot.pause(0.2)
        assert app.ollama_state == "ok"
        assert len(notices(app)) == count
        assert status(app).startswith("Scan done")


async def test_a_keyword_only_scan_says_so_in_the_status_line(tmp_path):
    app = tui.RadarApp(config(tmp_path), scan_fn=ScanSpy(ranking="keywords"), auto_scan=False, probe=down)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "keywords" in status(app)


async def test_f1_and_question_mark_open_the_key_sheet_and_q_quits(tmp_path):
    app = tui.RadarApp(config(tmp_path), auto_scan=False, probe=has(EMBED, "llama3:8b"))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("f1")
        assert isinstance(app.screen, tui.HelpScreen)
        await pilot.press("escape")
        assert not isinstance(app.screen, tui.HelpScreen)
        await pilot.press("question_mark")
        assert isinstance(app.screen, tui.HelpScreen)
        await pilot.press("escape")
        await pilot.press("q")
        await pilot.pause()
        assert app._exit


def test_every_key_line_of_the_sheet_fits_an_80_column_terminal():
    for key, label in tui.KEYS:
        assert len(f"  {key:<9}{label}") <= 80 - 10  # the box takes border and padding


async def test_the_footer_fits_80_columns_and_names_f1_and_quit(tmp_path):
    app = tui.RadarApp(config(tmp_path), auto_scan=False, probe=has(EMBED, "llama3:8b"))
    async with app.run_test(size=(80, 24)) as pilot:
        keys = []
        for _ in range(60):  # the footer fills in a moment after the screen opens, later on a slow machine
            await pilot.pause(0.05)
            keys = list(app.screen.query(FooterKey))
            if {"s", "e", "f1", "q"} <= {k.key for k in keys}:
                break
        shown = {k.key for k in keys}
        assert {"s", "e", "f1", "q"} <= shown
        assert "question_mark" not in shown
        assert all(k.region.right <= 80 and k.region.width > 0 for k in keys)


async def test_e_opens_interests_yaml_and_s_scans_with_the_saved_edit(tmp_path):
    cfg = config(tmp_path, own=False, write=True)
    opened = []
    spy = ScanSpy()
    app = tui.RadarApp(cfg, scan_fn=spy, auto_scan=False, probe=has(EMBED, "llama3:8b"), opener=opened.append)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app.example_topics
        await pilot.press("e")
        assert opened == [cfg.home / "interests.yaml"]
        assert "Save it, then press s" in status(app)
        # the user writes their own topics and saves, the app stays open
        data = yaml.safe_load((cfg.home / "interests.yaml").read_text(encoding="utf-8"))
        data["topics"] = OWN_TOPICS
        (cfg.home / "interests.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        await pilot.press("s")
        await app.workers.wait_for_complete()
        assert [t.name for t in spy.configs[0].topics] == ["Teaching videos", "Terrain"]
        assert not app.example_topics


async def test_a_broken_interests_file_stops_the_scan_with_an_error(tmp_path):
    cfg = config(tmp_path, write=True)
    spy = ScanSpy()
    app = tui.RadarApp(cfg, scan_fn=spy, auto_scan=False, probe=has(EMBED, "llama3:8b"))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        (cfg.home / "interests.yaml").write_text("topics: [\n", encoding="utf-8")
        await pilot.press("s")
        await pilot.pause()
        assert spy.configs == []
        assert any(sev == "error" and "interests.yaml" in m for sev, m in notices(app))


async def test_e_without_an_editor_shows_the_path(tmp_path):
    cfg = config(tmp_path, write=True)

    def no_editor(path):
        raise FileNotFoundError("xdg-open")

    app = tui.RadarApp(cfg, auto_scan=False, probe=has(EMBED, "llama3:8b"), opener=no_editor)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("e")
        assert str(cfg.home / "interests.yaml") in status(app)


async def test_update_notices_are_shown(tmp_path):
    notes = [("Research Radar updated (0.1.0 to 0.2.0).", "information")]
    app = tui.RadarApp(config(tmp_path), auto_scan=False, probe=has(EMBED, "llama3:8b"), notes=notes)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert ("information", notes[0][0]) in notices(app)
        assert app.query_one("#list", DataTable).row_count == 0


# --------------------------------------------------------------------------- radar doctor --local


def _ollama_with(names, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in names]})
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})
        return httpx.Response(404)

    monkeypatch.setattr(cli, "Ollama", lambda url, embed, chat: Ollama(url, embed, chat, transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(cli, "Http", lambda **kw: pytest.fail("--local must not touch the sources"))


def test_doctor_local_passes_when_every_model_is_there(tmp_path, monkeypatch, capsys):
    cli.main(["--home", str(tmp_path), "init"])
    capsys.readouterr()
    _ollama_with([EMBED, "llama3:8b"], monkeypatch)
    assert cli.main(["--home", str(tmp_path), "doctor", "--local"]) == 0
    assert capsys.readouterr().out.startswith("ok    ollama")


def test_doctor_local_names_the_missing_models(tmp_path, monkeypatch, capsys):
    cli.main(["--home", str(tmp_path), "init"])
    capsys.readouterr()
    _ollama_with([], monkeypatch)
    assert cli.main(["--home", str(tmp_path), "doctor", "--local"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL  ollama")
    assert "ollama pull nomic-embed-text" in out and "ollama pull " + llm.SUGGESTED_CHAT_MODEL in out
