"""Repository rules: no em dash anywhere, and no personal data shipped by accident."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}


def tracked_text_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS):
            if path.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ""}:
                yield path


def test_no_em_dash_characters():
    offenders = [str(p.relative_to(ROOT)) for p in tracked_text_files() if chr(0x2014) in p.read_text(encoding="utf-8", errors="ignore")]
    assert offenders == []


def test_gitignore_protects_personal_files():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("interests.yaml", "notes/", "state/", ".env"):
        assert pattern in text.splitlines()
