"""Step B: keyword previews, unknown keys as notices, and the scan lock."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from conftest import FakeOllama
from research_radar import cli, pipeline, schedule
from research_radar.config import example_text, parse_config
from research_radar.lock import LOCK_NAME, ScanLocked, scan_lock
from research_radar.llm import OllamaError
from research_radar.pipeline import scan

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def run(cfg, http, ollama=None, now=NOW, **kwargs):
    return scan(cfg, http=http, ollama=ollama or FakeOllama(), now=now, **kwargs)


def snapshot(home: Path) -> dict[str, str]:
    return {
        str(p.relative_to(home)): hashlib.sha1(p.read_bytes()).hexdigest()
        for p in sorted(home.rglob("*"))
        if p.is_file()
    }


class EmbedFails(FakeOllama):
    """Answers the ping but cannot embed: the embedding call fails during the scan."""

    def embed(self, texts, kind="document"):
        raise OllamaError("model unloaded")


class ChatFails(FakeOllama):
    def chat_json(self, system, user):
        raise OllamaError("chat down")


@pytest.fixture
def no_sending(monkeypatch):
    sent = []
    monkeypatch.setattr(pipeline, "deliver_pending", lambda *a, **k: sent.append(1) or [])
    return sent


# keyword previews ----------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["no_llm", "ollama_down", "embedding_fails"])
def test_a_keyword_scan_changes_no_file_and_sends_nothing(make_cfg, http_factory, no_sending, how):
    cfg = make_cfg()
    cfg.delivery["telegram"]["enabled"] = True
    earlier = run(cfg, http_factory(), now=NOW - timedelta(days=2))  # state, a note and a digest to protect
    assert earlier.new_reports and not earlier.preview
    no_sending.clear()
    before = snapshot(cfg.home)
    assert before

    if how == "no_llm":
        result = run(cfg, http_factory(), use_llm=False)
    elif how == "ollama_down":
        result = run(cfg, http_factory(), ollama=FakeOllama(up=False))
    else:
        result = run(cfg, http_factory(), ollama=EmbedFails())

    assert result.preview and result.ranking == "keywords"
    assert result.note_path is None and result.digest_path is None and result.deliveries == []
    assert snapshot(cfg.home) == before
    assert no_sending == []


def test_a_preview_lists_its_items(make_cfg, http_factory):
    result = run(make_cfg(), http_factory(), use_llm=False)
    assert result.preview and result.new_reports
    assert result.all_reports == result.new_reports


def test_a_preview_does_not_use_up_the_window(make_cfg, http_factory):
    cfg = make_cfg()
    preview = run(cfg, http_factory(), use_llm=False)
    real = run(cfg, http_factory())
    assert preview.new_reports and real.new_reports and not real.preview


def test_a_chat_failure_is_not_a_preview(make_cfg, http_factory):
    result = run(make_cfg(), http_factory(), ollama=ChatFails())
    assert not result.preview and result.ranking == "embeddings"
    assert result.note_path and result.note_path.exists()


def test_cli_scan_prints_the_preview_and_ends_with_the_line(make_cfg, http_factory, monkeypatch, capsys):
    cfg = make_cfg()
    monkeypatch.setattr(pipeline, "Http", lambda **k: http_factory())
    monkeypatch.setattr(pipeline, "Ollama", lambda *a, **k: FakeOllama(up=False))
    code = cli.cmd_scan(cfg, None, False, False)
    out = capsys.readouterr().out
    assert code == 0
    assert "alice/local-llm-kit" in out
    assert out.rstrip().endswith(cli.PREVIEW_END)
    assert "Note:" not in out


def test_a_scheduled_preview_exits_as_skipped(make_cfg, http_factory, monkeypatch, capsys):
    cfg = make_cfg()
    monkeypatch.setattr(pipeline, "Http", lambda **k: http_factory())
    monkeypatch.setattr(pipeline, "Ollama", lambda *a, **k: EmbedFails())
    assert cli.cmd_scan(cfg, None, False, False, scheduled=True) == schedule.EXIT_SKIPPED
    assert cli.PREVIEW_END in capsys.readouterr().out


# unknown keys --------------------------------------------------------------------------------


def _data():
    return yaml.safe_load(example_text())


def test_an_unknown_key_in_each_section_is_a_notice(tmp_path):
    data = _data()
    data["settings"]["min_scor"] = 0.4
    data["sources"]["nope"] = {"enabled": True}
    data["topics"][0]["keyword"] = ["x"]
    data["delivery"] = {"emial": {"enabled": True}, "email": {"enabled": False, "smtp_hots": "x"}}
    cfg = parse_config(data, tmp_path)
    text = "\n".join(cfg.notices)
    assert "unknown setting 'min_scor' was ignored (known: " in text
    assert "unknown source 'nope' was ignored (known: " in text
    assert f"unknown option 'keyword' in topic '{data['topics'][0]['name']}' was ignored" in text
    assert "unknown delivery option 'emial' was ignored" in text
    assert "unknown option 'smtp_hots' in delivery.email was ignored" in text
    assert len(cfg.notices) == 5
    assert all(n.startswith("interests.yaml: ") for n in cfg.notices)
    assert "min_scor" not in cfg.settings and "nope" not in cfg.sources and cfg.problem == ""


def test_the_scan_runs_with_unknown_keys(tmp_path, http_factory):
    data = _data()
    data["settings"].update({"min_score": 0.2, "min_scor": 0.9})
    cfg = parse_config(data, tmp_path / "home")
    result = run(cfg, http_factory())
    assert result.new_reports and cfg.notices


def test_a_file_with_only_unknown_keys_is_saved_as_the_last_working_copy(tmp_path):
    from research_radar.config import last_good_path, load_config

    home = tmp_path / "home"
    data = _data()
    data["settings"]["min_scor"] = 0.4
    home.mkdir()
    (home / "interests.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(home)
    assert cfg.problem == "" and len(cfg.notices) == 1
    assert last_good_path(home).exists()


def test_notices_reach_the_doctor_and_the_digest(tmp_path, capsys):
    from research_radar.deliver import build_digest
    from research_radar.store import Store

    data = _data()
    data["settings"]["min_scor"] = 0.4
    cfg = parse_config(data, tmp_path / "home")
    assert "min_scor" in cfg.alert
    assert "min_scor" in build_digest(cfg, Store(cfg.state_dir), "2026-10-27", []).alert
    cli.cmd_doctor(cfg, local_only=True)
    assert "WARN  interests.yaml: unknown setting 'min_scor'" in capsys.readouterr().out


def test_the_scan_command_prints_notices(tmp_path, http_factory, monkeypatch, capsys):
    data = _data()
    data["settings"].update({"min_score": 0.2, "min_scor": 0.9})
    cfg = parse_config(data, tmp_path / "home")
    monkeypatch.setattr(pipeline, "Http", lambda **k: http_factory())
    monkeypatch.setattr(pipeline, "Ollama", lambda *a, **k: FakeOllama())
    cli.cmd_scan(cfg, None, False, False)
    assert "notice: interests.yaml: unknown setting 'min_scor'" in capsys.readouterr().out


# the scan lock -------------------------------------------------------------------------------


def test_a_second_scan_stops_on_the_lock(make_cfg, http_factory):
    cfg = make_cfg()
    with scan_lock(cfg.state_dir):
        with pytest.raises(ScanLocked, match=r"^Another scan has been running since \d\d:\d\d\.$"):
            run(cfg, http_factory())
    assert not (cfg.state_dir / LOCK_NAME).exists()
    assert run(cfg, http_factory()).new_reports  # the lock is gone, so the next scan runs


def test_the_lock_is_removed_when_a_scan_fails(make_cfg, http_factory, monkeypatch):
    cfg = make_cfg()

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "write_note", boom)
    with pytest.raises(RuntimeError):
        run(cfg, http_factory())
    assert not (cfg.state_dir / LOCK_NAME).exists()


def test_a_stale_lock_is_taken_over(make_cfg, http_factory):
    cfg = make_cfg()
    cfg.state_dir.mkdir(parents=True)
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    (cfg.state_dir / LOCK_NAME).write_text(json.dumps({"pid": 1, "started": old.isoformat()}), encoding="utf-8")
    assert run(cfg, http_factory()).new_reports
    assert not (cfg.state_dir / LOCK_NAME).exists()


def test_a_lock_under_two_hours_old_is_respected(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    recent = datetime.now(timezone.utc) - timedelta(hours=1, minutes=59)
    (state / LOCK_NAME).write_text(json.dumps({"pid": 1, "started": recent.isoformat()}), encoding="utf-8")
    with pytest.raises(ScanLocked):
        with scan_lock(state):
            pass
    assert (state / LOCK_NAME).exists()  # not ours, so not removed


def test_a_preview_ignores_the_lock(make_cfg, http_factory):
    cfg = make_cfg()
    with scan_lock(cfg.state_dir):
        assert run(cfg, http_factory(), use_llm=False).preview


def test_the_cli_reports_a_locked_scan(make_cfg, http_factory, monkeypatch, capsys):
    cfg = make_cfg()
    monkeypatch.setattr(pipeline, "Http", lambda **k: http_factory())
    monkeypatch.setattr(pipeline, "Ollama", lambda *a, **k: FakeOllama())
    with scan_lock(cfg.state_dir):
        assert cli.cmd_scan(cfg, None, False, False) == schedule.EXIT_SKIPPED
    assert "Another scan has been running since" in capsys.readouterr().out
