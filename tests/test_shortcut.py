from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research_radar import cli, shortcut


HOME = Path("D:/Projects/Research Radar/data")
EXE = ["D:/Projects/Research Radar/research-radar/.venv/Scripts/radar.exe"]
ICON = HOME / "radar.ico"


def test_with_windows_terminal_the_shortcut_opens_a_titled_tab_in_the_data_folder():
    terminal = Path("C:/Users/x/AppData/Local/Microsoft/WindowsApps/wt.exe")
    spec = shortcut.build_spec(HOME, EXE, ICON, terminal)
    assert spec.target == str(terminal)
    assert spec.arguments.startswith('--title "Research Radar" -d "')
    assert f'"{EXE[0]}" --home "{HOME}"' in spec.arguments
    assert spec.workdir == str(HOME) and spec.icon == str(ICON)


def test_without_windows_terminal_it_starts_the_program_directly():
    spec = shortcut.build_spec(HOME, EXE, ICON, None)
    assert spec.target == EXE[0]
    assert spec.arguments == f'--home "{HOME}"'


def test_a_python_fallback_keeps_its_module_arguments():
    spec = shortcut.build_spec(HOME, ["C:/Python/python.exe", "-m", "research_radar"], ICON, None)
    assert spec.target == "C:/Python/python.exe"
    assert spec.arguments == f'"-m" "research_radar" --home "{HOME}"'


def test_quotes_in_a_path_cannot_break_out_of_the_arguments():
    spec = shortcut.build_spec(Path('D:/odd"name'), EXE, ICON, None)
    assert spec.arguments.count('"') == 2


def test_the_packaged_icon_is_a_multi_size_windows_icon():
    data = shortcut.packaged_icon().read_bytes()
    assert data[:4] == b"\x00\x00\x01\x00"
    assert int.from_bytes(data[4:6], "little") >= 4


def test_other_systems_get_a_clear_message(tmp_path):
    with pytest.raises(shortcut.ShortcutError, match="only supported on Windows"):
        shortcut.create_shortcut(tmp_path, is_windows=False)


def test_creating_the_shortcut_passes_values_by_environment_not_by_command_text(tmp_path, monkeypatch):
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["env"] = cmd, kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout="C:\\Users\\x\\Desktop\\Research Radar.lnk\n", stderr="")

    monkeypatch.setattr(shortcut.subprocess, "run", fake_run)
    monkeypatch.setattr(shortcut, "windows_terminal", lambda: None)
    home = tmp_path / "data with space"
    result = shortcut.create_shortcut(home, desktop=tmp_path, is_windows=True)
    assert str(result).endswith("Research Radar.lnk")
    assert (home / "radar.ico").exists()  # a stable copy of the icon next to the data
    assert seen["cmd"][:3] == ["powershell", "-NoProfile", "-NonInteractive"]
    assert str(home) not in seen["cmd"][-1]  # the script text never contains the user's paths
    assert seen["env"]["RR_ICON"] == str(home / "radar.ico")
    assert seen["env"]["RR_NAME"] == "Research Radar"
    assert f'--home "{home}"' in seen["env"]["RR_ARGUMENTS"]


def test_a_powershell_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shortcut.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="access denied")
    )
    monkeypatch.setattr(shortcut, "windows_terminal", lambda: None)
    with pytest.raises(shortcut.ShortcutError, match="access denied"):
        shortcut.create_shortcut(tmp_path, is_windows=True)


def test_the_command_needs_no_interests_file_and_explains_itself_off_windows(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(shortcut.sys, "platform", "linux")  # on a Windows runner it would really create one
    assert cli.main(["--home", str(tmp_path / "nothing-here"), "shortcut"]) == 1
    assert "only supported on Windows" in capsys.readouterr().err
