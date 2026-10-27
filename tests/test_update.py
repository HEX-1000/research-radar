"""The launch-time update: ZIP and git paths, user files, dependencies, plain pip installs."""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from research_radar import cli, update as up

INIT_OLD = '"""Research Radar."""\n\n__version__ = "0.2.0"\n'
INIT_NEW = '"""Research Radar."""\n\n__version__ = "0.3.0"\n'
PYPROJECT = '''[project]
name = "research-radar"
dynamic = ["version"]
dependencies = [
    "httpx>=0.27",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8"]
'''
PYPROJECT_MORE = PYPROJECT.replace('"pyyaml>=6.0",', '"pyyaml>=6.0",\n    "textual[syntax]>=9.0",  # new\n')
GOOD_PY = "VALUE = 'new'\n"


def make_root(root: Path) -> Path:
    """A source checkout as `pip install -e .` leaves it, with personal files next to it."""
    files = {
        "pyproject.toml": PYPROJECT,
        "README.md": "old readme\n",
        up.VERSION_PATH: INIT_OLD,
        "src/research_radar/cli.py": "VALUE = 'old'\n",
        "interests.yaml": "mine",
        "notes/2026/10/2026-10-27.md": "my note",
        "state/seen.json": "{}",
        ".env": "TOKEN=secret",
        ".venv/pyvenv.cfg": "home = x",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class Remote:
    """A fake GitHub: the version file, pyproject.toml and the repository ZIP, as file:// URLs."""

    def __init__(self, base: Path, init_text: str = INIT_NEW, pyproject: str = PYPROJECT, **extra: str | bytes) -> None:
        base.mkdir(parents=True, exist_ok=True)
        self.version_file = base / "remote_init.py"
        self.pyproject_file = base / "remote_pyproject.toml"
        self.zip_file = base / "main.zip"
        self.version_file.write_text(init_text, encoding="utf-8")
        self.pyproject_file.write_text(pyproject, encoding="utf-8")
        files: dict[str, str | bytes] = {
            "pyproject.toml": pyproject,
            "README.md": "new readme\n",
            up.VERSION_PATH: init_text,
            "src/research_radar/cli.py": GOOD_PY,
            "src/research_radar/newmod.py": "X = 1\n",
            "interests.yaml": "must not overwrite",
            "notes/x.md": "must not appear",
            ".env": "must not overwrite",
        }
        files.update(extra)
        with zipfile.ZipFile(self.zip_file, "w") as zf:
            for rel, content in files.items():
                zf.writestr(f"research-radar-main/{rel}", content)

    @property
    def urls(self) -> dict[str, str]:
        return {
            "version_url": self.version_file.as_uri(),
            "pyproject_url": self.pyproject_file.as_uri(),
            "zip_url": self.zip_file.as_uri(),
        }


class FakePip:
    def __init__(self, fail: bool = False, watch: Path | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail
        self.watch = watch
        self.seen_then: str | None = None

    def __call__(self, requirements: list[str], root: Path) -> None:
        self.calls.append(list(requirements))
        if self.watch is not None:
            self.seen_then = self.watch.read_text(encoding="utf-8")
        if self.fail:
            raise up.UpdateError("pip could not install the new dependencies (offline)")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return make_root(tmp_path / "app")


def run(root: Path, remote: Remote, pip: FakePip | None = None) -> up.UpdateResult:
    return up.check_and_update(root, pip=pip or FakePip(), **remote.urls)


# --------------------------------------------------------------------------- ZIP path


def test_zip_update_replaces_code_and_keeps_user_files(root, tmp_path):
    result = run(root, Remote(tmp_path / "remote"))
    assert (result.status, result.old, result.new) == ("updated", "0.2.0", "0.3.0"), result.message
    assert up.read_local_version(root) == "0.3.0"
    assert (root / "src/research_radar/cli.py").read_text() == GOOD_PY
    assert (root / "src/research_radar/newmod.py").read_text() == "X = 1\n"
    assert (root / "README.md").read_text() == "new readme\n"
    # personal files next to the code are untouched, and nothing personal is added
    assert (root / "interests.yaml").read_text() == "mine"
    assert (root / ".env").read_text() == "TOKEN=secret"
    assert (root / "notes/2026/10/2026-10-27.md").read_text() == "my note"
    assert not (root / "notes/x.md").exists()
    # replaced files were backed up, and no temporary files are left
    assert (root / up.BACKUP_DIR / "src/research_radar/cli.py").read_text() == "VALUE = 'old'\n"
    assert (root / up.BACKUP_DIR / up.VERSION_PATH).read_text() == INIT_OLD
    assert list(root.rglob("*.new")) == []


def test_same_version_changes_nothing(root, tmp_path):
    result = run(root, Remote(tmp_path / "remote", init_text=INIT_OLD))
    assert result.status == "current"
    assert (root / "src/research_radar/cli.py").read_text() == "VALUE = 'old'\n"


def test_local_newer_than_github_changes_nothing(root, tmp_path):
    (root / up.VERSION_PATH).write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    result = run(root, Remote(tmp_path / "remote"))
    assert result.status == "current"
    assert up.read_local_version(root) == "1.0.0"


def test_unreachable_github_is_offline(root, tmp_path):
    missing = (tmp_path / "nope").as_uri()
    result = up.check_and_update(root, version_url=missing, pyproject_url=missing, zip_url=missing)
    assert result.status == "offline"
    assert up.read_local_version(root) == "0.2.0"


def test_a_page_without_a_version_fails_quietly(root, tmp_path):
    remote = Remote(tmp_path / "remote", init_text="<html>404</html>")
    result = run(root, remote)
    assert result.status == "failed"
    assert up.read_local_version(root) == "0.2.0"


def test_syntax_error_in_the_download_writes_nothing(root, tmp_path):
    remote = Remote(tmp_path / "remote", **{"src/research_radar/cli.py": "def broken(:\n"})
    result = run(root, remote)
    assert result.status == "failed"
    assert "does not compile" in result.message
    assert up.read_local_version(root) == "0.2.0"
    assert (root / "README.md").read_text() == "old readme\n"
    assert not (root / "src/research_radar/newmod.py").exists()


def test_zip_slip_is_rejected(root, tmp_path):
    remote = Remote(tmp_path / "remote")
    with zipfile.ZipFile(remote.zip_file, "a") as zf:
        zf.writestr("research-radar-main/../escaped.py", "x = 1\n")
    result = run(root, remote)
    assert result.status == "failed"
    assert not (tmp_path / "escaped.py").exists()
    assert up.read_local_version(root) == "0.2.0"


def test_zip_without_the_package_is_rejected(root, tmp_path):
    remote = Remote(tmp_path / "remote")
    with zipfile.ZipFile(remote.zip_file, "w") as zf:
        zf.writestr("research-radar-main/README.md", "only a readme")
    result = run(root, remote)
    assert result.status == "failed"
    assert (root / "src/research_radar/cli.py").read_text() == "VALUE = 'old'\n"


def test_a_download_that_is_not_a_zip_is_rejected(root, tmp_path):
    remote = Remote(tmp_path / "remote")
    remote.zip_file.write_bytes(b"this is not a zip")
    assert run(root, remote).status == "failed"


def test_a_line_ending_difference_is_not_rewritten(root, tmp_path):
    (root / "README.md").write_bytes(b"new readme\r\n")
    run(root, Remote(tmp_path / "remote"))
    assert (root / "README.md").read_bytes() == b"new readme\r\n"
    assert not (root / up.BACKUP_DIR / "README.md").exists()


def test_new_dependencies_are_installed_before_any_code_changes(root, tmp_path):
    pip = FakePip(watch=root / up.VERSION_PATH)
    result = run(root, Remote(tmp_path / "remote", pyproject=PYPROJECT_MORE), pip)
    assert result.status == "updated", result.message
    assert pip.calls == [["httpx>=0.27", "pyyaml>=6.0", "textual[syntax]>=9.0"]]
    assert pip.seen_then == INIT_OLD  # pip ran while the old version was still in place


def test_unchanged_dependencies_do_not_run_pip(root, tmp_path):
    pip = FakePip()
    run(root, Remote(tmp_path / "remote"), pip)
    assert pip.calls == []


def test_a_pip_failure_changes_nothing(root, tmp_path):
    result = run(root, Remote(tmp_path / "remote", pyproject=PYPROJECT_MORE), FakePip(fail=True))
    assert result.status == "failed"
    assert "pip" in result.message
    assert up.read_local_version(root) == "0.2.0"
    assert (root / "src/research_radar/cli.py").read_text() == "VALUE = 'old'\n"


# --------------------------------------------------------------------------- plain pip install


def test_a_plain_pip_install_is_only_told_the_command(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "installed_version", lambda: "0.2.0")
    remote = Remote(tmp_path / "remote")
    result = up.check_and_update(None, detect=False, **remote.urls)
    assert (result.status, result.new) == ("available", "0.3.0")
    assert "pip install --upgrade" in result.message and up.ZIP_URL in result.message
    same = Remote(tmp_path / "same", init_text=INIT_OLD)
    assert up.check_and_update(None, detect=False, **same.urls).status == "current"


def test_source_root_finds_this_checkout_and_not_site_packages(tmp_path):
    here = up.source_root()
    assert here is not None and (here / "pyproject.toml").exists()
    fake = tmp_path / "lib" / "site-packages" / "research_radar" / "update.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    assert up.source_root(fake) is None


# --------------------------------------------------------------------------- small parts


def test_is_protected():
    for rel in ("interests.yaml", "notes/2026/a.md", "state/seen.json", "digests/x.html", ".env", ".env.local",
                "feedback-2026.json", "me.local.yaml", ".venv/Scripts/python.exe", ".git/config",
                "src/research_radar/__pycache__/a.pyc", "src/research_radar.egg-info/PKG-INFO"):
        assert up.is_protected(rel), rel
    for rel in ("README.md", "pyproject.toml", up.VERSION_PATH, "src/research_radar/store.py",
                "src/research_radar/notes.py", "src/research_radar/interests.example.yaml", "assets/banner.png",
                "install.ps1", "tests/test_update.py"):
        assert not up.is_protected(rel), rel


def test_versions():
    assert up.parse_version("1.10.0") == (1, 10, 0)
    assert up.parse_version("v1.0") is None and up.parse_version("") is None
    assert up.is_newer("1.10.0", "1.9.0") and up.is_newer("0.2.1", "0")
    assert not up.is_newer("1.0.0", "1.0.0")
    assert up.version_in(INIT_NEW) == "0.3.0"
    assert up.version_in("nothing here") is None


def test_dependencies_with_and_without_tomllib():
    expected = ["httpx>=0.27", "pyyaml>=6.0", "textual[syntax]>=9.0"]
    assert up.dependencies(PYPROJECT_MORE) == expected
    assert up._dependencies_without_tomllib(PYPROJECT_MORE) == expected
    assert up._dependencies_without_tomllib(PYPROJECT) == ["httpx>=0.27", "pyyaml>=6.0"]
    assert up._dependencies_without_tomllib('[project]\nname = "x"\n') == []
    real = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert up._dependencies_without_tomllib(real) == up.dependencies(real)


def test_origin_must_be_the_official_repository_exactly():
    for url in ("https://github.com/HEX-1000/research-radar", "https://github.com/hex-1000/research-radar.git",
                "git@github.com:HEX-1000/research-radar.git", "/tmp/x/HEX-1000/research-radar.git/",
                "C:\\Users\\me\\HEX-1000\\research-radar.git"):
        assert up.origin_is(url, "HEX-1000/research-radar"), url
    for url in ("https://github.com/HEX-1000/research-radar-dev", "https://github.com/someone/research-radar",
                "https://github.com/xHEX-1000/research-radar"):
        assert not up.origin_is(url, "HEX-1000/research-radar"), url


# --------------------------------------------------------------------------- git path


def git(cwd: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": os.environ["PATH"], "HOME": str(cwd), "GIT_CONFIG_NOSYSTEM": "1",
    }
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return out.stdout


@pytest.fixture
def clone(tmp_path: Path):
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    for rel, text in {"pyproject.toml": PYPROJECT, up.VERSION_PATH: INIT_OLD,
                      "src/research_radar/cli.py": "VALUE = 1\n"}.items():
        (seed / rel).parent.mkdir(parents=True, exist_ok=True)
        (seed / rel).write_text(text, encoding="utf-8")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "one")
    origin = tmp_path / "HEX-1000" / "research-radar.git"
    origin.parent.mkdir()
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(origin))
    app = tmp_path / "app"
    git(tmp_path, "clone", "-q", str(origin), str(app))
    (seed / up.VERSION_PATH).write_text(INIT_NEW, encoding="utf-8")
    (seed / "src/research_radar/cli.py").write_text("VALUE = 2\n", encoding="utf-8")
    git(seed, "commit", "-q", "-am", "two")
    git(seed, "push", "-q", str(origin), "main")
    urls = {
        "version_url": (seed / up.VERSION_PATH).as_uri(),
        "pyproject_url": (seed / "pyproject.toml").as_uri(),
        "zip_url": (tmp_path / "unused.zip").as_uri(),
    }
    return seed, app, urls


def test_a_clean_clone_is_pulled(clone):
    seed, app, urls = clone
    result = up.check_and_update(app, pip=FakePip(), **urls)
    assert (result.status, result.new) == ("updated", "0.3.0"), result.message
    assert (app / "src/research_radar/cli.py").read_text() == "VALUE = 2\n"


def test_local_code_edits_block_the_pull(clone):
    seed, app, urls = clone
    (app / "src/research_radar/cli.py").write_text("VALUE = 'mine'\n", encoding="utf-8")
    result = up.check_and_update(app, pip=FakePip(), **urls)
    assert result.status == "skipped"
    assert (app / "src/research_radar/cli.py").read_text() == "VALUE = 'mine'\n"


def test_untracked_personal_files_do_not_block_the_pull(clone):
    seed, app, urls = clone
    (app / "interests.yaml").write_text("mine", encoding="utf-8")
    result = up.check_and_update(app, pip=FakePip(), **urls)
    assert result.status == "updated", result.message
    assert (app / "interests.yaml").read_text() == "mine"


def test_a_clone_of_another_repository_is_left_alone(clone):
    seed, app, urls = clone
    result = up.check_and_update(app, repo="someone/research-radar", pip=FakePip(), **urls)
    assert result.status == "skipped"
    assert (app / "src/research_radar/cli.py").read_text() == "VALUE = 1\n"


def test_git_path_installs_new_dependencies_before_pulling(clone):
    seed, app, urls = clone
    (seed / "pyproject.toml").write_text(PYPROJECT_MORE, encoding="utf-8")
    git(seed, "commit", "-q", "-am", "three")
    git(seed, "push", "-q", str(app.parent / "HEX-1000" / "research-radar.git"), "main")
    pip = FakePip(watch=app / "src/research_radar/cli.py")
    result = up.check_and_update(app, pip=pip, **urls)
    assert result.status == "updated", result.message
    assert pip.calls == [["httpx>=0.27", "pyyaml>=6.0", "textual[syntax]>=9.0"]]
    assert pip.seen_then == "VALUE = 1\n"  # the pull came after pip


def test_without_git_on_path_a_clone_is_not_touched(clone, monkeypatch):
    seed, app, urls = clone
    monkeypatch.setattr(up.shutil, "which", lambda name: None)
    result = up.check_and_update(app, pip=FakePip(), **urls)
    assert result.status == "skipped" and "git" in result.message
    assert (app / "src/research_radar/cli.py").read_text() == "VALUE = 1\n"


# --------------------------------------------------------------------------- command line


def test_radar_update_prints_one_line(monkeypatch, capsys):
    monkeypatch.setattr(up, "check_and_update", lambda: up.UpdateResult("current", "0.2.0", "0.2.0"))
    assert cli.main(["update"]) == 0
    assert "0.2.0 is up to date" in capsys.readouterr().out
    monkeypatch.setattr(up, "check_and_update", lambda: up.UpdateResult("failed", "0.2.0", "0.3.0", "boom"))
    assert cli.main(["update"]) == 1


def test_an_installed_update_starts_the_new_copy_with_the_same_arguments(monkeypatch):
    seen: dict = {}

    def fake_call(cmd, env):
        seen["cmd"], seen["env"] = cmd, env
        return 7

    monkeypatch.setattr(up, "check_and_update", lambda: up.UpdateResult("updated", "0.2.0", "0.3.0", "git pull"))
    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    code, notes = cli.auto_update(["--home", "D:/data"])
    assert code == 7 and notes == []
    assert seen["cmd"][1:] == ["-m", "research_radar", "--home", "D:/data"]
    assert seen["env"][cli.UPDATED_ENV] == "0.2.0 to 0.3.0"


def test_update_problems_become_notices_and_the_app_still_starts(monkeypatch):
    monkeypatch.setattr(up, "check_and_update", lambda: up.UpdateResult("skipped", "0.2.0", "", "tracked files have local edits"))
    code, notes = cli.auto_update([])
    assert code is None and notes == [("Update check: tracked files have local edits. Still on 0.2.0.", "warning")]
    monkeypatch.setattr(up, "check_and_update", lambda: up.UpdateResult("available", "0.2.0", "0.3.0", "pip install --upgrade x"))
    code, notes = cli.auto_update([])
    assert code is None and "0.3.0 is available" in notes[0][0]

    def broken():
        raise RuntimeError("unexpected")

    monkeypatch.setattr(up, "check_and_update", broken)
    assert cli.auto_update([]) == (None, [])


def _tui_spy(monkeypatch):
    opened: list = []
    monkeypatch.setattr(cli, "cmd_tui", lambda cfg, notes=None: opened.append(notes) or 0)
    return opened


def test_the_copy_started_after_an_update_says_so_and_does_not_check_again(tmp_path, monkeypatch):
    cli.main(["--home", str(tmp_path), "init"])
    opened = _tui_spy(monkeypatch)
    monkeypatch.setattr(cli, "auto_update", lambda argv: pytest.fail("must not check again"))
    monkeypatch.setattr(up, "source_root", lambda *a: None)  # a pip install has no CHANGELOG.md
    monkeypatch.setenv(cli.UPDATED_ENV, "0.2.0 to 0.3.0")
    assert cli.main(["--home", str(tmp_path)]) == 0
    assert opened == [[("Research Radar updated (0.2.0 to 0.3.0).", "information")]]
    assert cli.UPDATED_ENV not in os.environ  # not passed on to anything started later


CHANGELOG_TEXT = """# Changelog

## 0.3.0
- Third thing.
- Fourth thing.

## 0.2.0
- Second thing.

## 0.1.0
- First thing.
"""


def test_changes_between_two_versions():
    assert up.changes_between(CHANGELOG_TEXT, "0.2.0", "0.3.0") == ["Third thing.", "Fourth thing."]
    assert up.changes_between(CHANGELOG_TEXT, "0.1.0", "0.3.0") == [
        "0.3.0: Third thing.", "0.3.0: Fourth thing.", "0.2.0: Second thing.",
    ]
    assert up.changes_between(CHANGELOG_TEXT, "0.3.0", "0.3.0") == []  # nothing newer than where you are
    assert up.changes_between(CHANGELOG_TEXT, "0.1.0", "0.2.0") == ["Second thing."]  # not past the new version


def test_a_long_changelog_is_cut_with_a_pointer_to_the_file():
    text = "## 0.9.0\n" + "".join(f"- line {n}\n" for n in range(8))
    lines = up.changes_between(text, "0.8.0", "0.9.0", limit=3)
    assert lines == ["line 0", "line 1", "line 2", "and 5 more in CHANGELOG.md"]


def test_whats_new_never_fails(tmp_path):
    assert up.whats_new(None, "0.1.0", "0.2.0") == []
    assert up.whats_new(tmp_path, "0.1.0", "0.2.0") == []  # no CHANGELOG.md there
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG_TEXT, encoding="utf-8")
    assert up.whats_new(tmp_path, "0.2.0", "0.3.0") == ["Third thing.", "Fourth thing."]


def test_the_notice_after_an_update_lists_what_changed(tmp_path, monkeypatch):
    cli.main(["--home", str(tmp_path / "home"), "init"])
    root = tmp_path / "root"
    root.mkdir()
    (root / "CHANGELOG.md").write_text(CHANGELOG_TEXT, encoding="utf-8")
    opened = _tui_spy(monkeypatch)
    monkeypatch.setattr(up, "source_root", lambda *a: root)
    monkeypatch.setenv(cli.UPDATED_ENV, "0.2.0 to 0.3.0")
    cli.main(["--home", str(tmp_path / "home")])
    text, severity = opened[0][0]
    assert text == "Research Radar updated (0.2.0 to 0.3.0). What's new:\n- Third thing.\n- Fourth thing."
    assert severity == "information"


def test_radar_update_prints_what_changed(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    root.mkdir()
    (root / "CHANGELOG.md").write_text(CHANGELOG_TEXT, encoding="utf-8")
    monkeypatch.setattr(up, "source_root", lambda *a: root)
    monkeypatch.setattr(up, "check_and_update", lambda: up.UpdateResult("updated", "0.2.0", "0.3.0", "git pull"))
    assert cli.main(["update"]) == 0
    out = capsys.readouterr().out
    assert "What's new:" in out and "  - Third thing." in out


def test_the_real_changelog_covers_the_current_version():
    from research_radar import __version__

    text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    assert up.changes_between(text, "0.0.1", __version__, limit=100), "add an entry for this version to CHANGELOG.md"
    assert f"## {__version__}" in text


def test_auto_update_false_skips_the_check(tmp_path, monkeypatch):
    cli.main(["--home", str(tmp_path), "init"])
    path = tmp_path / "interests.yaml"
    text = path.read_text(encoding="utf-8")
    assert "  auto_update: true" in text
    path.write_text(text.replace("  auto_update: true", "  auto_update: false", 1), encoding="utf-8")
    opened = _tui_spy(monkeypatch)
    monkeypatch.setattr(cli, "auto_update", lambda argv: pytest.fail("auto_update is off"))
    assert cli.main(["--home", str(tmp_path)]) == 0
    assert opened == [[]]
