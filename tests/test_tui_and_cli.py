from __future__ import annotations

from datetime import datetime, timezone

from research_radar import cli
from research_radar.config import init_home
from research_radar.models import Report
from research_radar.pipeline import RunResult
from research_radar.store import Store
from research_radar.tui import RadarApp, needs_catch_up
from textual.widgets import DataTable, Static


def seed_today(cfg):
    store = Store(cfg.state_dir)
    today = datetime.now().astimezone().date().isoformat()
    reports = [
        Report(date=today, topic="Local-first AI tools", score=0.8, source="github", kind="repo", title="alice/local-llm-kit",
               url="https://github.com/alice/local-llm-kit", what="Runs models locally.", why="Fits.", verdict="install",
               meta={"stars": 120, "license": "MIT", "language": "Python"}),
        Report(date=today, topic="Geospatial machine learning", score=0.7, source="arxiv", kind="paper", title="A segmentation paper",
               url="https://arxiv.org/abs/1", what="Segments images.", why="Fits.", verdict="read", meta={"arxiv_id": "1"}),
    ]
    store.save_day(today, reports, [], [], "embeddings")
    return today


async def test_tui_lists_items_votes_and_navigates(make_cfg):
    cfg = make_cfg(auto_scan_on_launch=False)
    seed_today(cfg)
    app = RadarApp(cfg, auto_scan=False)
    async with app.run_test() as pilot:
        table = app.query_one("#list", DataTable)
        assert table.row_count == 2
        await pilot.press("plus")
        await pilot.pause()
        first = app.selected()
        assert first is not None and first.vote == 1
        await pilot.press("minus")
        await pilot.pause()
        assert app.selected().vote == -1
        await pilot.press("down")
        await pilot.pause()
        detail = app.query_one("#detail", Static).render()
        assert "Recommendation" in str(detail)
        await pilot.press("left_square_bracket")  # no older day exists, must not crash
        await pilot.pause()
        assert table.row_count == 2


async def test_tui_scan_key_runs_the_scan_and_shows_the_result(make_cfg):
    cfg = make_cfg(auto_scan_on_launch=False)
    calls: list[str] = []

    def fake_scan(config, progress=lambda m: None, **kwargs):
        calls.append("scan")
        progress("Reading github")
        today = seed_today(config)
        return RunResult(date=today, new_reports=Store(config.state_dir).load_day(today)["reports"])

    app = RadarApp(cfg, scan_fn=fake_scan, auto_scan=False)
    async with app.run_test() as pilot:
        assert app.query_one("#list", DataTable).row_count == 0
        await pilot.press("s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert calls == ["scan"]
        assert app.query_one("#list", DataTable).row_count == 2
        assert not app.scanning


async def test_tui_survives_a_failing_scan(make_cfg):
    cfg = make_cfg()

    def broken(config, progress=lambda m: None, **kwargs):
        raise RuntimeError("boom")

    app = RadarApp(cfg, scan_fn=broken, auto_scan=False)
    async with app.run_test() as pilot:
        await pilot.press("s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "boom" in str(app.query_one("#status", Static).render())
        assert not app.scanning


def test_catch_up_is_needed_until_every_source_ran_today(make_cfg):
    cfg = make_cfg()
    store = Store(cfg.state_dir)
    assert needs_catch_up(cfg, store)
    now = datetime.now(timezone.utc).isoformat()
    store.save_last_runs({name: now for name in cfg.sources if cfg.source_enabled(name)})
    assert not needs_catch_up(cfg, store)
    store.save_last_runs({"github": now})
    assert needs_catch_up(cfg, store)


def test_cli_init_then_scan_without_config_errors(tmp_path, capsys):
    assert cli.main(["--home", str(tmp_path / "h"), "scan"]) == 2
    assert "radar init" in capsys.readouterr().err
    assert cli.main(["--home", str(tmp_path / "h"), "init"]) == 0
    assert (tmp_path / "h" / "interests.yaml").exists()
    init_home(tmp_path / "h")  # idempotent


async def test_tui_uses_the_amber_theme_and_survives_an_unknown_theme_name(make_cfg):
    app = RadarApp(make_cfg(auto_scan_on_launch=False), auto_scan=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.theme == "hex-amber"
    odd = RadarApp(make_cfg(auto_scan_on_launch=False, theme="no-such-theme"), auto_scan=False)
    async with odd.run_test() as pilot:
        await pilot.pause()
        assert odd.theme == "hex-amber"


async def test_tui_list_shows_the_verdict_before_the_long_columns(make_cfg):
    cfg = make_cfg(auto_scan_on_launch=False)
    seed_today(cfg)
    app = RadarApp(cfg, auto_scan=False)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#list", DataTable)
        labels = [str(c.label) for c in table.columns.values()]
        assert labels == ["", "Verdict", "Topic", "Item"]
