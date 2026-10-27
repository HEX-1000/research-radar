"""Repository rules: no em dash anywhere, and no personal data shipped by accident."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}


def tracked_text_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS):
            if path.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".ps1", ""}:
                yield path


def test_no_em_dash_characters():
    offenders = [str(p.relative_to(ROOT)) for p in tracked_text_files() if chr(0x2014) in p.read_text(encoding="utf-8", errors="ignore")]
    assert offenders == []


def test_gitignore_protects_personal_files():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("interests.yaml", "notes/", "state/", "logs/", ".env", ".update_backup/"):
        assert pattern in text.splitlines()


def test_installer_is_plain_ascii_for_windows_powershell():
    # Windows PowerShell 5.1 reads a file without a byte order mark in the local code page.
    assert (ROOT / "install.ps1").read_bytes().isascii()


def test_the_version_is_written_in_one_place_only():
    # The update check compares __version__ in src/research_radar/__init__.py with GitHub.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'path = "src/research_radar/__init__.py"' in pyproject
    assert "\nversion =" not in pyproject
