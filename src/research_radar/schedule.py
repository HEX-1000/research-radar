"""A daily scan without opening the interface, through the Windows Task Scheduler.

`radar schedule on` registers one task for the current user. It runs `radar scan --scheduled`
at the chosen time with pythonw.exe, so no console window opens, and it writes what it did to
<data home>/logs/scan.log. The task is set to start as soon as possible when the computer was
off or asleep at the chosen time, and to wait for a network connection. It runs while you are
logged on and needs no password and no administrator rights.

On Linux and macOS nothing is registered: the command prints the cron line to use.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

TASK_NAME = "Research Radar daily scan"
DESCRIPTION = "Runs a Research Radar scan and sends the digest. Added by `radar schedule on`."
DEFAULT_TIME = "19:45"
EXIT_SKIPPED = 3  # `radar scan --scheduled` found Ollama or the embedding model missing and did not scan

# Values reach PowerShell through the environment, never through the command text,
# so a quote or a space in a path cannot change what the script does.
CREATE_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$at = [datetime]::ParseExact($env:RR_TIME, 'HH:mm', [Globalization.CultureInfo]::InvariantCulture)
$action = New-ScheduledTaskAction -Execute $env:RR_EXE -Argument $env:RR_ARGS -WorkingDirectory $env:RR_WORKDIR
$trigger = New-ScheduledTaskTrigger -Daily -At $at
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RunOnlyIfNetworkAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName $env:RR_TASK -Action $action -Trigger $trigger -Settings $settings -Description $env:RR_DESCRIPTION -Force | Out-Null
Write-Output 'ok'
"""

REMOVE_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $env:RR_TASK -ErrorAction SilentlyContinue
if ($task) {
    Unregister-ScheduledTask -TaskName $env:RR_TASK -Confirm:$false
    Write-Output 'removed'
} else {
    Write-Output 'absent'
}
"""

STATUS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $env:RR_TASK -ErrorAction SilentlyContinue
if (-not $task) { Write-Output 'absent'; exit 0 }
$info = Get-ScheduledTaskInfo -TaskName $env:RR_TASK
function Stamp($value) {
    if ($value -and $value.Year -gt 2000) { return $value.ToString('yyyy-MM-dd HH:mm') }
    return ''
}
Write-Output 'present'
Write-Output ('state=' + $task.State)
Write-Output ('time=' + ([datetime]$task.Triggers[0].StartBoundary).ToString('HH:mm'))
Write-Output ('next=' + (Stamp $info.NextRunTime))
Write-Output ('last=' + (Stamp $info.LastRunTime))
Write-Output ('result=' + $info.LastTaskResult)
"""


class ScheduleError(Exception):
    """The task could not be created, removed or read."""


@dataclass
class TaskStatus:
    installed: bool
    state: str = ""
    time: str = ""
    next_run: str = ""
    last_run: str = ""
    last_result: int | None = None


Runner = Callable[[str, dict[str, str]], "subprocess.CompletedProcess[str]"]


def log_path(home: Path) -> Path:
    return home / "logs" / "scan.log"


def parse_time(text: str) -> str:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text.strip())
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise ScheduleError(f"'{text}' is not a time of day. Write it as HH:MM, for example 19:45.")
    return f"{int(match.group(1)):02d}:{match.group(2)}"


def background_python() -> str:
    """pythonw.exe starts without a console window. Elsewhere the normal interpreter."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    return str(pythonw if pythonw.exists() else sys.executable)


def task_arguments(home: Path) -> str:
    return f'-m research_radar --home "{str(home).replace(chr(34), "")}" scan --scheduled'


def cron_line(home: Path, at: str) -> str:
    hour, minute = at.split(":")
    return f'{int(minute)} {int(hour)} * * * "{sys.executable}" -m research_radar --home "{home}" scan --scheduled'


def result_text(code: int | None) -> str:
    if code is None:
        return "unknown"
    code &= 0xFFFFFFFF
    known = {
        0: "the last run finished",
        1: "the last run failed, see the log",
        2: "the last run stopped on an error in interests.yaml, see the log",
        EXIT_SKIPPED: "the last run did not scan because Ollama or the embedding model was not ready",
        0x41303: "it has not run yet",
        0x41301: "it is running now",
        0x41325: "the last run was skipped because another one was still running",
    }
    return known.get(code, f"the last run ended with code {code}")


def _run_powershell(script: str, env: dict[str, str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        env=dict(os.environ, **env),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _call(script: str, env: dict[str, str], runner: Runner | None) -> list[str]:
    try:
        done = (runner or _run_powershell)(script, env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ScheduleError(f"could not run PowerShell ({type(exc).__name__})") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        raise ScheduleError("the Task Scheduler refused: " + (detail[0][:300] if detail else f"exit code {done.returncode}"))
    return [line.strip() for line in (done.stdout or "").splitlines() if line.strip()]


def _require_windows(is_windows: bool | None) -> None:
    if (sys.platform == "win32") if is_windows is None else is_windows:
        return
    raise ScheduleError("The Task Scheduler exists only on Windows. On Linux or macOS use cron, see the line `radar schedule on` prints.")


def create(home: Path, at: str = DEFAULT_TIME, *, runner: Runner | None = None, is_windows: bool | None = None) -> str:
    """Register the daily task. Returns the time it was set to."""
    _require_windows(is_windows)
    at = parse_time(at)
    home.mkdir(parents=True, exist_ok=True)
    env = {
        "RR_TASK": TASK_NAME,
        "RR_TIME": at,
        "RR_EXE": background_python(),
        "RR_ARGS": task_arguments(home),
        "RR_WORKDIR": str(home),
        "RR_DESCRIPTION": DESCRIPTION,
    }
    lines = _call(CREATE_SCRIPT, env, runner)
    if "ok" not in lines:
        raise ScheduleError("the Task Scheduler did not confirm the task")
    return at


def remove(*, runner: Runner | None = None, is_windows: bool | None = None) -> bool:
    """Delete the task. Returns False when there was none."""
    _require_windows(is_windows)
    return "removed" in _call(REMOVE_SCRIPT, {"RR_TASK": TASK_NAME}, runner)


def status(*, runner: Runner | None = None, is_windows: bool | None = None) -> TaskStatus:
    _require_windows(is_windows)
    lines = _call(STATUS_SCRIPT, {"RR_TASK": TASK_NAME}, runner)
    if "present" not in lines:
        return TaskStatus(installed=False)
    fields = dict(line.split("=", 1) for line in lines if "=" in line)
    try:
        code: int | None = int(fields.get("result", ""))
    except ValueError:
        code = None
    return TaskStatus(
        installed=True,
        state=fields.get("state", ""),
        time=fields.get("time", ""),
        next_run=fields.get("next", ""),
        last_run=fields.get("last", ""),
        last_result=code,
    )


def describe(task: TaskStatus, log: Path) -> list[str]:
    if not task.installed:
        return ["No daily scan is set up. Run `radar schedule on` to add one."]
    lines = [f"A scan runs every day at {task.time or '?'} ({task.state or 'state unknown'})."]
    if task.next_run:
        lines.append(f"Next run: {task.next_run}")
    lines.append(f"Last run: {task.last_run}, {result_text(task.last_result)}." if task.last_run else f"Last run: never, {result_text(task.last_result)}.")
    lines.append(f"Log: {log}")
    return lines
