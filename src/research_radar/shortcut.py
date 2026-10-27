"""A desktop icon for Windows that opens the terminal interface with one click.

The shortcut starts Windows Terminal when it is installed (it draws the colors
and box characters best) and falls back to the plain console otherwise. It
carries the data folder in its arguments, so it works even when the
RADAR_HOME environment variable is not visible to the program that launches it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

NAME = "Research Radar"
DESCRIPTION = "Local research radar (Ollama)"

# Values reach PowerShell through the environment, never through the command text,
# so a quote or a space in a path cannot change what the script does.
SCRIPT = r"""
$desktop = $env:RR_DESKTOP
if (-not $desktop) { $desktop = [Environment]::GetFolderPath('Desktop') }
$path = Join-Path $desktop ($env:RR_NAME + '.lnk')
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($path)
$link.TargetPath = $env:RR_TARGET
$link.Arguments = $env:RR_ARGUMENTS
$link.WorkingDirectory = $env:RR_WORKDIR
$link.IconLocation = $env:RR_ICON
$link.Description = $env:RR_DESCRIPTION
$link.Save()
Write-Output $path
"""


class ShortcutError(Exception):
    """The shortcut could not be created."""


@dataclass
class Spec:
    target: str
    arguments: str
    workdir: str
    icon: str


def _quote(text: str) -> str:
    return '"' + text.replace('"', "") + '"'


def launcher_command() -> list[str]:
    """The command that starts the program from this Python environment."""
    exe = Path(sys.executable).parent / "radar.exe"
    if exe.exists():
        return [str(exe)]
    return [sys.executable, "-m", "research_radar"]


def windows_terminal() -> Path | None:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = Path(local) / "Microsoft" / "WindowsApps" / "wt.exe"
        if candidate.exists():
            return candidate
    found = shutil.which("wt.exe")
    return Path(found) if found else None


def build_spec(home: Path, command: list[str], icon: Path, terminal: Path | None) -> Spec:
    """What the shortcut runs. Pure, so it can be tested without Windows."""
    run = " ".join(_quote(part) for part in command) + f" --home {_quote(str(home))}"
    if terminal is not None:
        arguments = f'--title {_quote(NAME)} -d {_quote(str(home))} {run}'
        return Spec(str(terminal), arguments, str(home), str(icon))
    first, rest = command[0], command[1:]
    arguments = " ".join([*(_quote(p) for p in rest), "--home", _quote(str(home))])
    return Spec(first, arguments, str(home), str(icon))


def packaged_icon() -> Path:
    return Path(str(resources.files("research_radar").joinpath("assets", "radar.ico")))


def create_shortcut(home: Path, *, desktop: Path | None = None, is_windows: bool | None = None) -> Path:
    if is_windows is None:
        is_windows = sys.platform == "win32"
    if not is_windows:
        raise ShortcutError(
            "Desktop shortcuts are only supported on Windows. On Linux or macOS, add an alias such as "
            "alias radar-open='radar' to your shell profile."
        )
    home.mkdir(parents=True, exist_ok=True)
    icon = home / "radar.ico"  # a stable copy, so the icon survives moving or rebuilding the virtual environment
    shutil.copyfile(packaged_icon(), icon)
    spec = build_spec(home, launcher_command(), icon, windows_terminal())
    env = dict(os.environ)
    env.update(
        RR_NAME=NAME,
        RR_TARGET=spec.target,
        RR_ARGUMENTS=spec.arguments,
        RR_WORKDIR=spec.workdir,
        RR_ICON=spec.icon,
        RR_DESCRIPTION=DESCRIPTION,
        RR_DESKTOP=str(desktop) if desktop else "",
    )
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ShortcutError(f"could not run PowerShell ({type(exc).__name__})") from exc
    if done.returncode != 0:
        raise ShortcutError("PowerShell could not create the shortcut: " + (done.stderr or "").strip()[:300])
    return Path(done.stdout.strip().splitlines()[-1])
