"""Self-update for Research Radar. Standard library only.

When the terminal interface opens, check_and_update() reads one small file from the official
GitHub repository, src/research_radar/__init__.py, and compares its __version__ with yours.
If GitHub has a newer version, the new code is installed and the interface starts again from
the updated files.

Which copies update themselves
  - A git clone of the official repository, installed with `pip install -e .`:
    git pull --ff-only, and only when no tracked file has local edits.
  - A folder unpacked from the repository ZIP and installed with `pip install -e .` (what
    install.ps1 does): the new ZIP is downloaded, every Python file in it is syntax-checked,
    and only then are files written.
  - A plain `pip install` (the code lives in site-packages) is not changed. The interface
    shows the command that updates it.

What an update touches
  - Only files that come from the repository. Your data folder (interests.yaml, notes,
    digests, state) lives elsewhere and is never touched. .git, .venv and personal files at
    the top of the folder (if you keep your data inside the clone) are skipped as well.
  - Nothing is deleted. Files an update replaces are copied to .update_backup/ first.
  - When pyproject.toml lists new or changed dependencies, pip installs them before any code
    changes. If pip fails, nothing changes.

Nothing about you is sent: the requests are plain GETs to raw.githubusercontent.com and
github.com. Turn it off with `auto_update: false` under settings in interests.yaml.
Run it by hand with `radar update`.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Callable

REPO = "HEX-1000/research-radar"
BRANCH = "main"
VERSION_PATH = "src/research_radar/__init__.py"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
VERSION_URL = f"{RAW}/{VERSION_PATH}"
PYPROJECT_URL = f"{RAW}/pyproject.toml"
ZIP_URL = f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip"

BACKUP_DIR = ".update_backup"
MAX_ZIP_BYTES = 20 * 1024 * 1024
MAX_FILE_BYTES = 5 * 1024 * 1024

# Never written, wherever they appear in a path.
PROTECTED_PARTS = {".git", ".venv", "venv", "__pycache__", BACKUP_DIR, ".pytest_cache", "build", "dist"}
# Personal data that can sit at the top of the folder when someone keeps the data home in the clone.
PERSONAL_TOP = {"interests.yaml", "notes", "digests", "state", "logs", ".env"}
PERSONAL_TOP_PATTERNS = ("feedback*", ".env.*", "*.local.yaml")
REQUIRED_FILES = ("pyproject.toml", VERSION_PATH, "src/research_radar/cli.py")

VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)

Pip = Callable[[list[str], Path], None]


class UpdateError(Exception):
    """An expected failure with a message the user can read."""


@dataclass
class UpdateResult:
    # current | updated | available | offline | skipped | failed
    status: str
    old: str = ""
    new: str = ""
    message: str = ""


# --------------------------------------------------------------------------- versions


def parse_version(text: str) -> tuple[int, ...] | None:
    text = text.strip()
    if not re.fullmatch(r"\d+(\.\d+){0,3}", text):
        return None
    return tuple(int(p) for p in text.split("."))


def is_newer(remote: str, local: str) -> bool:
    r, loc = parse_version(remote), parse_version(local) or (0,)
    return r is not None and r > loc


def version_in(text: str) -> str | None:
    """The __version__ string in the text of an __init__.py, or None."""
    match = VERSION_RE.search(text)
    return match.group(1).strip() if match else None


def read_local_version(root: Path) -> str:
    try:
        return version_in((root / VERSION_PATH).read_text(encoding="utf-8")) or "0"
    except OSError:
        return "0"


def installed_version() -> str:
    from . import __version__

    return __version__


# --------------------------------------------------------------------------- where the code lives


def source_root(package_file: str | os.PathLike[str] | None = None) -> Path | None:
    """The folder of the source checkout this code runs from, or None for a plain pip install."""
    here = Path(package_file or __file__).resolve()
    if here.parent.name != "research_radar" or here.parents[1].name != "src":
        return None
    root = here.parents[2]
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    if not re.search(r'(?m)^name\s*=\s*"research-radar"\s*$', text):
        return None
    return root


def pip_upgrade_command() -> str:
    return f'"{sys.executable}" -m pip install --upgrade {ZIP_URL}'


# --------------------------------------------------------------------------- paths


def is_protected(rel: str) -> bool:
    """True for a path in the download that must never overwrite something of the user's."""
    parts = PurePosixPath(rel).parts
    if not parts:
        return True
    if any(p in PROTECTED_PARTS or p.endswith(".egg-info") for p in parts):
        return True
    top = parts[0]
    return top in PERSONAL_TOP or any(fnmatch(top, pattern) for pattern in PERSONAL_TOP_PATTERNS)


def _safe_relative(name: str) -> str:
    if "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise UpdateError(f"unsafe path in the download: {name}")
    parts = PurePosixPath(name).parts
    if not parts or any(p in ("..", "") for p in parts):
        raise UpdateError(f"unsafe path in the download: {name}")
    return "/".join(parts)


def _same(a: bytes, b: bytes) -> bool:
    """Equal, ignoring CRLF versus LF (Git on Windows may convert line endings)."""
    return a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")


# --------------------------------------------------------------------------- dependencies


def dependencies(pyproject_text: str) -> list[str] | None:
    """The [project] dependencies, or None when the file cannot be read."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        return _dependencies_without_tomllib(pyproject_text)
    try:
        data = tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError:
        return None
    deps = data.get("project", {}).get("dependencies", [])
    return [str(d) for d in deps] if isinstance(deps, list) else None


def _dependencies_without_tomllib(text: str) -> list[str] | None:
    """Enough of TOML for the dependencies array of this project's pyproject.toml."""
    project = re.search(r"(?m)^\[project\]\s*$", text)
    if not project:
        return None
    following = re.search(r"(?m)^\[", text[project.end():])
    section = text[project.end(): project.end() + following.start()] if following else text[project.end():]
    start = re.search(r"(?m)^dependencies\s*=\s*\[", section)
    if not start:
        return []
    out: list[str] = []
    i = start.end()
    while i < len(section):
        char = section[i]
        if char in "\"'":
            end = section.find(char, i + 1)
            if end < 0:
                return None
            out.append(section[i + 1: end])
            i = end + 1
        elif char == "#":
            newline = section.find("\n", i)
            i = len(section) if newline < 0 else newline
        elif char == "]":
            return out
        else:
            i += 1
    return None


def _changed_dependencies(old_text: str, new_text: str) -> list[str]:
    """The new dependency list when it differs from the old one, else an empty list."""
    new = dependencies(new_text)
    if new is None:
        raise UpdateError("pyproject.toml in the download cannot be read")
    old = dependencies(old_text)
    return new if new and new != old else []


def _pip_install(requirements: list[str], root: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *requirements],
        capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        last = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        # On Windows a package that is in use cannot be replaced. A fresh terminal can do it.
        raise UpdateError(
            f"pip could not install the new dependencies ({last[0]}). Close Research Radar and run: "
            f'"{sys.executable}" -m pip install -e "{root}"'
        )


# --------------------------------------------------------------------------- network


def _fetch(url: str, timeout: float, limit: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "research-radar-updater"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise ValueError("download is larger than expected")
    return data


# --------------------------------------------------------------------------- zip update


def _read_zip(data: bytes) -> dict[str, bytes]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UpdateError("the download is not a valid zip file") from exc
    files: dict[str, bytes] = {}
    with zf:
        members = [i for i in zf.infolist() if not i.filename.endswith("/")]
        tops = {i.filename.split("/", 1)[0] for i in members}
        if len(tops) != 1 or any("/" not in i.filename for i in members):
            raise UpdateError("the zip does not have the expected single top folder")
        top = tops.pop()
        for info in members:
            rel = _safe_relative(info.filename[len(top) + 1:])
            if info.file_size > MAX_FILE_BYTES:
                raise UpdateError(f"file too large in the download: {rel}")
            files[rel] = zf.read(info)
    return files


def _check_files(files: dict[str, bytes], old: str) -> str:
    for name in REQUIRED_FILES:
        if name not in files:
            raise UpdateError(f"the download has no {name}")
    new = version_in(files[VERSION_PATH].decode("utf-8", "replace")) or ""
    if not is_newer(new, old):
        raise UpdateError(f"the download is version {new or '?'}, not newer than {old}")
    for rel, data in files.items():
        if rel.endswith(".py"):
            try:
                compile(data, rel, "exec")
            except (SyntaxError, ValueError) as exc:
                raise UpdateError(f"{rel} in the download does not compile: {exc}") from exc
    return new


def _target_for(root: Path, rel: str) -> Path | None:
    """Where an update may write this file, or None to leave it alone."""
    target = root / rel
    resolved_root = root.resolve()
    if resolved_root != target.resolve() and resolved_root not in target.resolve().parents:
        raise UpdateError(f"unsafe path in the download: {rel}")
    return None if is_protected(rel) else target


def _update_zip(root: Path, zip_url: str, old: str, timeout: float, pip: Pip) -> UpdateResult:
    try:
        data = _fetch(zip_url, max(timeout, 30.0), MAX_ZIP_BYTES)
    except (OSError, ValueError) as exc:
        raise UpdateError(f"could not download the update: {exc}") from exc
    files = _read_zip(data)
    new = _check_files(files, old)

    changes: list[tuple[str, Path]] = []
    for rel in sorted(files):
        if rel == VERSION_PATH:
            continue  # written last, so a half-done update is retried on the next start
        target = _target_for(root, rel)
        if target is None:
            continue
        if target.exists() and _same(target.read_bytes(), files[rel]):
            continue
        changes.append((rel, target))

    old_pyproject = (root / "pyproject.toml").read_text(encoding="utf-8", errors="replace") if (root / "pyproject.toml").exists() else ""
    needed = _changed_dependencies(old_pyproject, files["pyproject.toml"].decode("utf-8", "replace"))
    if needed:
        pip(needed, root)

    backup = root / BACKUP_DIR
    shutil.rmtree(backup, ignore_errors=True)
    done: list[tuple[str, Path]] = []
    try:
        for rel, target in [*changes, (VERSION_PATH, root / VERSION_PATH)]:
            if target.exists():
                dest = backup / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, dest)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = target.with_name(target.name + ".new")
            tmp_file.write_bytes(files[rel])
            os.replace(tmp_file, target)
            done.append((rel, target))
    except OSError as exc:
        for rel, target in done:
            saved = backup / rel
            if saved.exists():
                shutil.copy2(saved, target)
            else:
                target.unlink(missing_ok=True)
        raise UpdateError(f"could not write the update, old files restored: {exc}") from exc
    return UpdateResult("updated", old, new, f"{len(changes) + 1} file(s) replaced")


# --------------------------------------------------------------------------- git update


def _git(root: Path, *args: str, timeout: float = 60) -> tuple[int, str]:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        timeout=timeout, stdin=subprocess.DEVNULL, env=env,
    )
    # rstrip only: the first column of `git status --porcelain` can be a space.
    return proc.returncode, (proc.stdout + proc.stderr).rstrip()


def origin_is(url: str, repo: str) -> bool:
    """True when a remote URL points at owner/name exactly (https, ssh or a local path)."""
    url = url.strip().replace("\\", "/")  # a local path on Windows uses backslashes
    return re.search(r"[:/]" + re.escape(repo) + r"(?:\.git)?/?$", url, re.IGNORECASE) is not None


def _update_git(root: Path, old: str, repo: str, branch: str, pyproject_url: str, timeout: float, pip: Pip) -> UpdateResult:
    if not shutil.which("git"):
        return UpdateResult("skipped", old, "", "git is not on PATH; run git pull yourself")
    rc, origin = _git(root, "remote", "get-url", "origin")
    if rc != 0 or not origin_is(origin, repo):
        return UpdateResult("skipped", old, "", "this clone does not point at the official repository")
    rc, current = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if rc != 0 or current != branch:
        return UpdateResult("skipped", old, "", f"not on branch {branch}")
    rc, status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if rc != 0 or status.strip():
        return UpdateResult("skipped", old, "", "tracked files have local edits; run git pull yourself")

    try:
        remote_pyproject = _fetch(pyproject_url, max(timeout, 10.0), 256 * 1024).decode("utf-8", "replace")
    except (OSError, ValueError) as exc:
        raise UpdateError(f"could not read pyproject.toml on GitHub: {exc}") from exc
    local_pyproject = (root / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
    needed = _changed_dependencies(local_pyproject, remote_pyproject)
    if needed:
        pip(needed, root)

    rc, out = _git(root, "pull", "--ff-only", "--quiet", "origin", branch, timeout=120)
    if rc != 0:
        last = out.splitlines()[-1:] or ["no output"]
        raise UpdateError(f"git pull failed: {last[0]}")
    new = read_local_version(root)
    return UpdateResult("updated" if is_newer(new, old) else "current", old, new, "git pull")


# --------------------------------------------------------------------------- entry point


def check_and_update(
    root: Path | None = None,
    *,
    detect: bool = True,
    version_url: str = VERSION_URL,
    pyproject_url: str = PYPROJECT_URL,
    zip_url: str = ZIP_URL,
    repo: str = REPO,
    branch: str = BRANCH,
    timeout: float = 4.0,
    pip: Pip | None = None,
) -> UpdateResult:
    """Compare the version on GitHub with this one and install a newer copy. Never raises.

    root is the source folder to update. With detect=True and no root, the folder this code
    runs from is used; a plain pip install has none and only learns whether an update exists.
    """
    if root is None and detect:
        root = source_root()
    root = Path(root).resolve() if root is not None else None
    old = read_local_version(root) if root is not None else installed_version()
    try:
        remote_text = _fetch(version_url, timeout, 64 * 1024).decode("utf-8", "replace")
    except (OSError, ValueError):
        return UpdateResult("offline", old)
    remote = version_in(remote_text) or ""
    if parse_version(remote) is None:
        return UpdateResult("failed", old, remote[:20], "the version on GitHub could not be read")
    if not is_newer(remote, old):
        return UpdateResult("current", old, remote)
    if root is None:
        return UpdateResult("available", old, remote, pip_upgrade_command())
    pip = pip or _pip_install
    try:
        if (root / ".git").exists():
            result = _update_git(root, old, repo, branch, pyproject_url, timeout, pip)
        else:
            result = _update_zip(root, zip_url, old, timeout, pip)
    except UpdateError as exc:
        return UpdateResult("failed", old, remote, str(exc))
    except Exception as exc:  # an update problem must never stop the app from starting
        return UpdateResult("failed", old, remote, f"{type(exc).__name__}: {exc}")
    return result


CHANGELOG = "CHANGELOG.md"
SECTION_RE = re.compile(r"^##\s+v?(\d+(?:\.\d+){0,3})\s*$")


def changes_between(text: str, old: str, new: str, limit: int = 5) -> list[str]:
    """The entries of CHANGELOG.md for versions after old, up to and including new.

    One version gives its lines as they are. Several give each line a "version: " prefix.
    More than limit lines are cut, with a pointer to the file."""
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        heading = SECTION_RE.match(line.strip())
        if heading:
            sections.append((heading.group(1), []))
        elif sections and line.lstrip().startswith("- "):
            sections[-1][1].append(line.lstrip()[2:].strip())
    wanted = [(v, items) for v, items in sections if is_newer(v, old) and not is_newer(v, new)]
    several = len(wanted) > 1
    lines = [f"{v}: {item}" if several else item for v, items in wanted for item in items]
    if len(lines) > limit:
        lines = lines[:limit] + [f"and {len(lines) - limit} more in {CHANGELOG}"]
    return lines


def whats_new(root: Path | None, old: str, new: str) -> list[str]:
    """What changed between two versions, from CHANGELOG.md in the source folder. Never raises."""
    if root is None:
        return []
    try:
        return changes_between((root / CHANGELOG).read_text(encoding="utf-8"), old, new)
    except (OSError, ValueError):
        return []


def describe(result: UpdateResult) -> str:
    """One line for the console."""
    if result.status == "current":
        return f"Research Radar {result.old} is up to date."
    if result.status == "updated":
        return f"Updated Research Radar from {result.old} to {result.new} ({result.message})."
    if result.status == "available":
        return f"Research Radar {result.new} is available (you have {result.old}). Update with: {result.message}"
    if result.status == "offline":
        return f"GitHub could not be reached. Still on {result.old}."
    return f"Update {result.status}: {result.message}. Still on {result.old}."
