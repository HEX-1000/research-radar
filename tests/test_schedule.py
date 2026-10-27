"""The daily scan: the Task Scheduler wrapper and the scan the task runs. No Windows needed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research_radar import cli, schedule
from research_radar.config import init_home
from research_radar.schedule import ScheduleError


def done(out: str = "", code: int = 0, err: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout=out, stderr=err)


class Recorder:
    def __init__(self, out: str = "ok", code: int = 0, err: str = "") -> None:
        self.calls: list[tuple[str, dict]] = []
        self.reply = done(out, code, err)

    def __call__(self, script: str, env: dict) -> subprocess.CompletedProcess:
        self.calls.append((script, env))
        return self.reply


# the time -----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [("19:45", "19:45"), ("7:05", "07:05"), (" 00:00 ", "00:00")])
def test_times_are_normalised(text, expected):
    assert schedule.parse_time(text) == expected


@pytest.mark.parametrize("text", ["", "25:00", "19:60", "7pm", "19.45", "1945"])
def test_bad_times_are_refused_with_an_example(text):
    with pytest.raises(ScheduleError, match="HH:MM"):
        schedule.parse_time(text)


# the task -----------------------------------------------------------------------

def test_create_passes_every_value_through_the_environment(tmp_path):
    home = tmp_path / "my data folder"
    run = Recorder("ok")
    assert schedule.create(home, "6:30", runner=run, is_windows=True) == "06:30"
    script, env = run.calls[0]
    assert env["RR_TIME"] == "06:30"
    assert env["RR_TASK"] == schedule.TASK_NAME
    assert env["RR_WORKDIR"] == str(home)
    assert '--home "' in env["RR_ARGS"] and env["RR_ARGS"].endswith("scan --scheduled")
    assert f'--home "{home}"' in env["RR_ARGS"]
    assert "StartWhenAvailable" in script and "-RunOnlyIfNetworkAvailable" in script
    assert str(home) not in script  # nothing from the user is pasted into the script text


def test_a_quote_in_a_path_cannot_end_the_quoted_argument_early():
    args = schedule.task_arguments(Path('C:/my "data" folder'))
    assert args.count('"') == 2 and 'my data folder' in args


def test_create_fails_with_the_scheduler_message(tmp_path):
    run = Recorder("", code=1, err="Access is denied.\nmore")
    with pytest.raises(ScheduleError, match="Access is denied"):
        schedule.create(tmp_path, runner=run, is_windows=True)


def test_create_needs_confirmation_from_the_script(tmp_path):
    with pytest.raises(ScheduleError, match="did not confirm"):
        schedule.create(tmp_path, runner=Recorder(""), is_windows=True)


def test_nothing_is_attempted_off_windows(tmp_path):
    run = Recorder()
    for call in (lambda: schedule.create(tmp_path, runner=run, is_windows=False),
                 lambda: schedule.remove(runner=run, is_windows=False),
                 lambda: schedule.status(runner=run, is_windows=False)):
        with pytest.raises(ScheduleError, match="only on Windows"):
            call()
    assert run.calls == []


def test_remove_reports_whether_there_was_a_task():
    assert schedule.remove(runner=Recorder("removed"), is_windows=True) is True
    assert schedule.remove(runner=Recorder("absent"), is_windows=True) is False


def test_status_of_a_task_that_ran():
    out = "present\nstate=Ready\ntime=19:45\nnext=2026-10-28 19:45\nlast=2026-10-27 19:45\nresult=0\n"
    task = schedule.status(runner=Recorder(out), is_windows=True)
    assert task.installed and task.time == "19:45" and task.last_result == 0
    lines = schedule.describe(task, Path("scan.log"))
    assert lines[0] == "A scan runs every day at 19:45 (Ready)."
    assert "Next run: 2026-10-28 19:45" in lines
    assert any("Last run: 2026-10-27 19:45, the last run finished" in line for line in lines)


def test_status_of_a_task_that_never_ran_and_of_none():
    out = "present\nstate=Ready\ntime=19:45\nnext=2026-10-27 19:45\nlast=\nresult=267011\n"
    task = schedule.status(runner=Recorder(out), is_windows=True)
    assert any("never" in line for line in schedule.describe(task, Path("x")))
    none = schedule.status(runner=Recorder("absent"), is_windows=True)
    assert not none.installed
    assert "radar schedule on" in schedule.describe(none, Path("x"))[0]


def test_exit_codes_are_explained():
    assert "Ollama" in schedule.result_text(schedule.EXIT_SKIPPED)
    assert "interests.yaml" in schedule.result_text(2)
    assert "code 77" in schedule.result_text(77)
    assert schedule.result_text(None) == "unknown"


def test_the_cron_line_for_other_systems(tmp_path):
    line = schedule.cron_line(tmp_path, "07:05")
    assert line.startswith("5 7 * * * ")
    assert line.endswith("scan --scheduled")


# the commands ---------------------------------------------------------------------

def test_schedule_on_needs_interests_yaml(tmp_path, capsys):
    assert cli.main(["--home", str(tmp_path / "empty"), "schedule", "on"]) == 1
    assert "radar init" in capsys.readouterr().err


def test_schedule_on_elsewhere_prints_the_cron_line(tmp_path, capsys, monkeypatch):
    home = tmp_path / "home"
    init_home(home)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert cli.main(["--home", str(home), "schedule", "on", "--at", "8:15"]) == 1
    out = capsys.readouterr().out
    assert "crontab" in out and "15 8 * * *" in out


def test_schedule_rejects_a_bad_time(tmp_path, capsys, monkeypatch):
    home = tmp_path / "home"
    init_home(home)
    monkeypatch.setattr(cli.sys, "platform", "win32")
    assert cli.main(["--home", str(home), "schedule", "on", "--at", "evening"]) == 1
    assert "HH:MM" in capsys.readouterr().err


# the scan the task runs ---------------------------------------------------------------

@pytest.fixture
def scheduled_home(tmp_path):
    home = tmp_path / "home"
    init_home(home)
    return home


def log_of(home: Path) -> str:
    return schedule.log_path(home).read_text(encoding="utf-8")


def test_scheduled_scan_writes_everything_to_the_log(scheduled_home, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ollama_names", lambda cfg: ["nomic-embed-text:latest", "qwen3.5:9b"])
    monkeypatch.setattr(cli, "cmd_scan", lambda cfg, days, no_llm, explain: print("3 new items") or 0)
    assert cli.main(["--home", str(scheduled_home), "scan", "--scheduled"]) == 0
    assert capsys.readouterr().out == ""  # pythonw has no console to print to
    text = log_of(scheduled_home)
    assert "3 new items" in text and text.rstrip().endswith("exit code 0")


def test_scheduled_scan_waits_for_ollama_then_gives_up_without_scanning(scheduled_home, monkeypatch):
    scanned = []
    waits = []
    monkeypatch.setattr(cli, "_ollama_names", lambda cfg: None)
    monkeypatch.setattr(cli, "cmd_scan", lambda *a: scanned.append(1) or 0)
    code = cli.cmd_scheduled_scan(str(scheduled_home), None, False, False, sleep=waits.append)
    assert code == schedule.EXIT_SKIPPED
    assert scanned == []
    assert len(waits) == cli.OLLAMA_TRIES - 1
    text = log_of(scheduled_home)
    assert "Ollama" in text and "No scan was run" in text


def test_scheduled_scan_goes_ahead_when_ollama_comes_up_late(scheduled_home, monkeypatch):
    answers = iter([None, ["nomic-embed-text:latest", "qwen3.5:9b"]])
    monkeypatch.setattr(cli, "_ollama_names", lambda cfg: next(answers))
    monkeypatch.setattr(cli, "cmd_scan", lambda *a: 0)
    assert cli.cmd_scheduled_scan(str(scheduled_home), None, False, False, sleep=lambda s: None) == 0


def test_scheduled_scan_skips_when_the_embedding_model_is_missing(scheduled_home, monkeypatch):
    monkeypatch.setattr(cli, "_ollama_names", lambda cfg: ["qwen3.5:9b"])
    monkeypatch.setattr(cli, "cmd_scan", lambda *a: pytest.fail("must not scan"))
    waits = []
    assert cli.cmd_scheduled_scan(str(scheduled_home), None, False, False, sleep=waits.append) == schedule.EXIT_SKIPPED
    assert waits == []  # a missing model does not fix itself, so there is nothing to wait for
    assert "ollama pull nomic-embed-text" in log_of(scheduled_home)


def test_no_llm_scans_do_not_need_ollama(scheduled_home, monkeypatch):
    monkeypatch.setattr(cli, "_ollama_names", lambda cfg: pytest.fail("must not ask Ollama"))
    monkeypatch.setattr(cli, "cmd_scan", lambda *a: 0)
    assert cli.cmd_scheduled_scan(str(scheduled_home), None, True, False) == 0


def test_a_broken_interests_file_and_a_crash_end_up_in_the_log(scheduled_home, monkeypatch):
    (scheduled_home / "interests.yaml").write_text("topics: [", encoding="utf-8")
    assert cli.cmd_scheduled_scan(str(scheduled_home), None, False, False) == 2
    assert "error:" in log_of(scheduled_home)

    (scheduled_home / "interests.yaml").unlink()
    init_home(scheduled_home)
    monkeypatch.setattr(cli, "_ollama_names", lambda cfg: ["nomic-embed-text:latest", "qwen3.5:9b"])

    def boom(*a):
        raise RuntimeError("source exploded")

    monkeypatch.setattr(cli, "cmd_scan", boom)
    assert cli.cmd_scheduled_scan(str(scheduled_home), None, False, False) == 1
    assert "RuntimeError: source exploded" in log_of(scheduled_home)


def test_the_log_is_moved_aside_when_it_gets_big(scheduled_home, monkeypatch):
    monkeypatch.setattr(cli, "_ollama_names", lambda cfg: ["nomic-embed-text:latest", "qwen3.5:9b"])
    monkeypatch.setattr(cli, "cmd_scan", lambda *a: 0)
    log = schedule.log_path(scheduled_home)
    log.parent.mkdir(parents=True)
    log.write_text("x" * (cli.LOG_LIMIT + 1), encoding="utf-8")
    cli.cmd_scheduled_scan(str(scheduled_home), None, False, False)
    assert log.with_suffix(".log.1").exists()
    assert "xxxx" not in log.read_text(encoding="utf-8")
