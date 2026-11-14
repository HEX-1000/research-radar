"""The candidate store: every item a scan received, what selection did with it, and the numbers
that change. One SQLite file, state/radar.db, standard library only.

It replaces seen.json and vectors.json. The first scan that writes builds it from those two files
and renames them to *.migrated. Until that scan saves, the store lives in memory, so a scan that
fails or turns out to be a keyword preview leaves every file as it was.

Schema 2 adds the `label` and `vote` tables. Votes used to live in feedback.json; a scan, the
interface or any other writer that opens a schema 1 file upgrades it in place and retires that file.
"""

from __future__ import annotations

import json
import os
import sqlite3
from array import array
from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import Item, key_from_url
from .sources.github import HOLD_DAYS

DB_NAME = "radar.db"
SCHEMA_VERSION = 2
JSON_FILES = ("seen.json", "vectors.json", "feedback.json")
FEEDBACK_FILE = "feedback.json"
BUSY_TIMEOUT = 10  # seconds a writer waits for another one (the interface, `radar label`, a scan)
VECTOR_DAYS = 30  # vectors of items not shown and not voted are kept this long after first seen

SCHEMA_2_TABLES = """
CREATE TABLE label (
  key   TEXT NOT NULL REFERENCES item(key),
  topic TEXT NOT NULL,
  value TEXT NOT NULL,
  at    TEXT NOT NULL,
  PRIMARY KEY (key, topic)
);
CREATE TABLE vote (
  key   TEXT PRIMARY KEY REFERENCES item(key),
  value INTEGER NOT NULL,
  at    TEXT NOT NULL,
  via   TEXT NOT NULL
);
"""

SCHEMA = """
CREATE TABLE item (
  key         TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,
  url         TEXT NOT NULL,
  title       TEXT NOT NULL,
  text        TEXT NOT NULL,
  published   TEXT,
  meta        TEXT NOT NULL DEFAULT '{}',
  first_seen  TEXT NOT NULL,
  last_seen   TEXT NOT NULL,
  state       TEXT NOT NULL,
  state_on    TEXT NOT NULL,
  topic       TEXT,
  shown_stars INTEGER,
  held_until  TEXT
);
CREATE TABLE link (
  key    TEXT NOT NULL REFERENCES item(key),
  source TEXT NOT NULL,
  url    TEXT NOT NULL UNIQUE,
  PRIMARY KEY (key, source)
);
CREATE TABLE scan (
  id          INTEGER PRIMARY KEY,
  at          TEXT NOT NULL,
  version     TEXT NOT NULL,
  embed_model TEXT NOT NULL,
  chat_model  TEXT NOT NULL,
  windows     TEXT NOT NULL,
  fetched     INTEGER NOT NULL
);
CREATE TABLE score (
  key          TEXT NOT NULL REFERENCES item(key),
  topic        TEXT NOT NULL,
  topic_mark   TEXT NOT NULL,
  model        TEXT NOT NULL,
  similarity   REAL NOT NULL,
  keyword_hits INTEGER NOT NULL,
  scan         INTEGER NOT NULL REFERENCES scan(id),
  PRIMARY KEY (key, topic, topic_mark, model)
);
CREATE TABLE decision (
  scan    INTEGER NOT NULL REFERENCES scan(id),
  key     TEXT NOT NULL REFERENCES item(key),
  topic   TEXT NOT NULL,
  score   REAL NOT NULL,
  outcome TEXT NOT NULL,
  PRIMARY KEY (scan, key)
);
CREATE TABLE observation (
  key       TEXT NOT NULL REFERENCES item(key),
  at        TEXT NOT NULL,
  stars     INTEGER,
  forks     INTEGER,
  size_kb   INTEGER,
  followers INTEGER,
  PRIMARY KEY (key, at)
);
CREATE TABLE vector (
  key   TEXT NOT NULL REFERENCES item(key),
  model TEXT NOT NULL,
  data  BLOB NOT NULL,
  PRIMARY KEY (key, model)
);
""" + SCHEMA_2_TABLES

_KIND_BY_SOURCE = {
    "github": "repo",
    "forgejo": "repo",
    "arxiv": "paper",
    "huggingface-papers": "paper",
    "huggingface-models": "model",
}
_KIND_BY_PREFIX = {"github": "repo", "forgejo": "repo", "arxiv": "paper", "hf": "model"}
_ITEM_COLUMNS = "key, url, state, state_on, topic, shown_stars, held_until, meta"


class StoreError(Exception):
    """The candidate store could not be opened, created or saved. The message is fit to show."""


@dataclass
class StoredItem:
    key: str
    url: str
    state: str  # seen, shown, held, dropped or gone
    state_on: str  # local date the state was set
    topic: str | None
    shown_stars: int | None
    held_until: str | None
    meta: dict[str, Any]

    @property
    def full_name(self) -> str:
        """owner/name of a GitHub repository, as GitHub spelled it when known."""
        name = self.meta.get("full_name")
        return name if isinstance(name, str) and name else self.key.split(":", 1)[1]


def _number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass
class ScanRecord:
    """Everything one scan writes to the store. It is saved in one transaction at the end."""

    now: datetime
    day: str  # local date of the scan
    embed_model: str
    chat_model: str = ""
    windows: dict[str, list[str]] = field(default_factory=dict)
    fetched: int = 0
    items: dict[str, Item] = field(default_factory=dict)  # every item the scan received, by key
    links: set[tuple[str, str, str]] = field(default_factory=set)  # key, source, url
    scores: list[tuple[str, str, str, float, int]] = field(default_factory=list)  # key, topic, mark, similarity, hits
    decisions: dict[str, tuple[str, float, str]] = field(default_factory=dict)  # key: topic, score, outcome
    observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    followers: dict[str, int | None] = field(default_factory=dict)  # owner login in lower case, as asked
    vectors: dict[str, list[float]] = field(default_factory=dict)
    states: dict[str, dict[str, Any]] = field(default_factory=dict)  # key: item columns to set

    @property
    def at(self) -> str:
        return self.now.astimezone(timezone.utc).isoformat()

    def add(self, item: Item) -> bool:
        """Note an item a source returned. True when its key is new to this scan; a second
        copy (the same paper from another source) only adds its link."""
        key = item.key
        self.links.add((key, item.source, item.url))
        if item.source in ("github", "forgejo"):
            self.observe(key, item.meta)
        if key in self.items:
            return False
        self.items[key] = item
        return True

    def observe(self, key: str, numbers: dict[str, Any]) -> None:
        """Stars, forks and size from a search result or a repository lookup."""
        row = self.observations.setdefault(key, {})
        for name in ("stars", "forks", "size_kb"):
            value = _number(numbers.get(name))
            if value is not None:
                row[name] = value
        owner = numbers.get("owner")
        if isinstance(owner, str) and owner:
            row["owner"] = owner

    def set_state(self, key: str, state: str, **columns: Any) -> None:
        cols: dict[str, Any] = {"state": state, "state_on": self.day, "held_until": None}
        cols.update(columns)
        self.states[key] = {**self.states.get(key, {}), **cols}

    def update(self, key: str, **columns: Any) -> None:
        self.states[key] = {**self.states.get(key, {}), **columns}

    def decide(self, key: str, topic: str, score: float, outcome: str) -> None:
        self.decisions[key] = (topic, float(score), outcome)


def _pack(vector: Iterable[float]) -> bytes:
    return array("f", vector).tobytes()


def _unpack(data: bytes) -> list[float]:
    values = array("f")
    values.frombytes(data)
    return values.tolist()


def _connect(target: str, **kwargs: Any) -> sqlite3.Connection:
    conn = sqlite3.connect(target, isolation_level=None, timeout=BUSY_TIMEOUT, **kwargs)  # transactions by hand
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _plus_days(day: str, days: int, fallback: str) -> str:
    try:
        return (date_type.fromisoformat(day) + timedelta(days=days)).isoformat()
    except ValueError:
        return fallback  # an unreadable date must not hold a repository forever


class CandidateStore:
    """state/radar.db. Open it with `CandidateStore.open` and use it as a context manager."""

    def __init__(
        self, conn: sqlite3.Connection, path: Path, *, write: bool, create: bool, schema: int = SCHEMA_VERSION
    ) -> None:
        self.conn = conn
        self.schema = schema  # 1 only for a file opened read-only before it was upgraded
        self.path = path
        self.write = write
        self.create = create  # built in memory from the JSON files; written to `path` at the first save

    # opening -------------------------------------------------------------------------

    @classmethod
    def open(cls, state_dir: Path, *, embed_model: str, now: datetime, write: bool = True) -> "CandidateStore":
        """Open the store. Without a file yet, build it in memory from seen.json and vectors.json;
        the first save writes it. With write=False nothing is ever written."""
        path = state_dir / DB_NAME
        try:
            if path.exists():
                if write:
                    conn = _connect(str(path))
                else:
                    conn = _connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                version = _version(conn)
                if version > SCHEMA_VERSION:
                    conn.close()
                    raise StoreError(
                        f"{path} was made by a newer version of Research Radar (schema {version}). Run `radar update`."
                    )
                if version == SCHEMA_VERSION:
                    return cls(conn, path, write=write, create=False)
                if version == 1:
                    if write:
                        try:
                            _upgrade_1_to_2(conn, state_dir)
                        except BaseException:
                            conn.close()
                            raise
                        return cls(conn, path, write=True, create=False)
                    return cls(conn, path, write=False, create=False, schema=1)  # votes still in feedback.json
                conn.close()  # version 0, an empty file left by something else: build the store again
        except sqlite3.Error as exc:
            raise StoreError(f"Could not open {path}: {exc}.") from exc

        conn = _connect(":memory:")
        try:
            conn.execute("BEGIN")
            _build(conn, state_dir, embed_model, now)
            conn.execute("COMMIT")
        except Exception as exc:
            conn.close()
            raise StoreError(
                f"Could not create the store {path} from seen.json and vectors.json ({type(exc).__name__}: {exc}). "
                "Both files were left as they were and the scan stopped."
            ) from exc
        return cls(conn, path, write=write, create=True)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "CandidateStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # reading -------------------------------------------------------------------------

    def _rows_in(self, sql: str, keys: Iterable[str], before: tuple = ()) -> list[tuple]:
        """Run `sql`, whose "{}" is a list of placeholders, for the keys in chunks."""
        keys = list(dict.fromkeys(keys))
        rows: list[tuple] = []
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            rows.extend(self.conn.execute(sql.format(",".join("?" * len(chunk))), (*before, *chunk)).fetchall())
        return rows

    @staticmethod
    def _item(row: tuple) -> StoredItem:
        key, url, state, state_on, topic, shown_stars, held_until, meta = row
        try:
            data = json.loads(meta or "{}")
        except ValueError:
            data = {}
        return StoredItem(key, url, state, state_on, topic, shown_stars, held_until, data if isinstance(data, dict) else {})

    def known(self, keys: Iterable[str]) -> dict[str, StoredItem]:
        rows = self._rows_in(f"SELECT {_ITEM_COLUMNS} FROM item WHERE key IN ({{}})", keys)
        return {row[0]: self._item(row) for row in rows}

    def scored(self, keys: Iterable[str], model: str) -> set[tuple[str, str, str]]:
        """(key, topic, topic mark) of every topic version these items were scored against with `model`."""
        rows = self._rows_in("SELECT key, topic, topic_mark FROM score WHERE model = ? AND key IN ({})", keys, (model,))
        return {(k, t, m) for k, t, m in rows}

    def vectors(self, keys: Iterable[str], model: str) -> dict[str, list[float]]:
        rows = self._rows_in("SELECT key, data FROM vector WHERE model = ? AND key IN ({})", keys, (model,))
        return {key: _unpack(data) for key, data in rows}

    def key_for_url(self, url: str) -> str:
        row = self.conn.execute("SELECT key FROM link WHERE url = ?", (url,)).fetchone()
        return row[0] if row else key_from_url(url)

    def votes_by_key(self) -> dict[str, int]:
        """Every vote by item key."""
        if self.schema >= 2:
            return {key: value for key, value in self.conn.execute("SELECT key, value FROM vote")}
        # a schema 1 file opened for reading: its votes are still in feedback.json
        return {self.key_for_url(url): v for url, v in _read_feedback(self.path.parent).items() if v}

    def labels(self) -> list[tuple[str, str, str]]:
        """(key, topic, value) of every label; none before schema 2."""
        if self.schema < 2:
            return []
        return [tuple(row) for row in self.conn.execute("SELECT key, topic, value FROM label ORDER BY key, topic")]

    def vote_vectors(self, model: str) -> tuple[list[list[float]], list[list[float]]]:
        """Vectors of liked and disliked items, from the votes of the store."""
        keyed = self.votes_by_key()
        vectors = self.vectors(keyed, model)
        likes = [vectors[k] for k, v in keyed.items() if v > 0 and k in vectors]
        dislikes = [vectors[k] for k, v in keyed.items() if v < 0 and k in vectors]
        return likes, dislikes

    def held_due(self, today: str) -> list[StoredItem]:
        """GitHub repositories held back whose week is over."""
        rows = self.conn.execute(
            f"SELECT {_ITEM_COLUMNS} FROM item WHERE state = 'held' AND key LIKE 'github:%'"
            " AND (held_until IS NULL OR held_until <= ?) ORDER BY state_on, key",
            (today,),
        ).fetchall()
        return [self._item(r) for r in rows]

    def shown_repos(self, since: str, today: str) -> list[StoredItem]:
        """GitHub repositories shown from `since` up to the day before `today`, with their stars then."""
        rows = self.conn.execute(
            f"SELECT {_ITEM_COLUMNS} FROM item WHERE state = 'shown' AND key LIKE 'github:%'"
            " AND state_on >= ? AND state_on != ? AND shown_stars IS NOT NULL ORDER BY state_on, key",
            (since, today),
        ).fetchall()
        return [self._item(r) for r in rows]

    # writing -------------------------------------------------------------------------

    def save(self, record: ScanRecord) -> None:
        """Write a scan in one transaction. If anything fails, the store stays as it was."""
        if not self.write:
            raise StoreError("This store was opened for reading only.")
        conn = self.conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            _write(conn, record)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        if self.create:
            self._persist()

    def set_labels(self, key: str, labels: Iterable[tuple[str, str]], now: datetime | None = None) -> None:
        """Write the (topic, value) labels of one item in a transaction of their own. A label written
        again replaces the old one."""
        if not self.write or self.schema < 2:
            raise StoreError("This store was opened for reading only.")
        at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        conn = self.conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            for topic, value in labels:
                conn.execute("INSERT OR REPLACE INTO label (key, topic, value, at) VALUES (?, ?, ?, ?)", (key, topic, value, at))
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise StoreError(f"Could not write the label to {self.path}: {exc}.") from exc
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def _persist(self) -> None:
        """Write the store that was built in memory to its file, then retire the JSON files."""
        tmp = self.path.with_name(DB_NAME + ".tmp")
        try:
            tmp.unlink(missing_ok=True)
            target = sqlite3.connect(str(tmp))
            try:
                self.conn.backup(target)
            finally:
                target.close()
            os.replace(tmp, self.path)
        except (OSError, sqlite3.Error) as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise StoreError(
                f"Could not create the store {self.path} ({type(exc).__name__}: {exc}). "
                "seen.json and vectors.json were left as they were."
            ) from exc
        self.create = False
        for name in JSON_FILES:
            old = self.path.parent / name
            try:
                if old.exists():
                    old.replace(old.with_name(name + ".migrated"))
            except OSError:
                pass  # the store is in place; a stale copy only matters to an older version


def _write(conn: sqlite3.Connection, record: ScanRecord) -> None:
    from . import __version__

    at = record.at
    scan_id = conn.execute(
        "INSERT INTO scan (at, version, embed_model, chat_model, windows, fetched) VALUES (?, ?, ?, ?, ?, ?)",
        (at, __version__, record.embed_model, record.chat_model, json.dumps(record.windows), record.fetched),
    ).lastrowid
    conn.executemany(
        "INSERT INTO item (key, kind, url, title, text, published, meta, first_seen, last_seen, state, state_on)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'seen', ?)"
        " ON CONFLICT(key) DO UPDATE SET title = excluded.title, text = excluded.text,"
        " published = COALESCE(excluded.published, item.published), meta = excluded.meta,"
        " last_seen = excluded.last_seen",
        [
            (
                key,
                item.kind,
                item.url,
                item.title,
                item.text or "",
                item.published,
                json.dumps(item.meta, ensure_ascii=False, default=str),
                at,
                at,
                record.day,
            )
            for key, item in record.items.items()
        ],
    )
    conn.executemany("INSERT OR IGNORE INTO link (key, source, url) VALUES (?, ?, ?)", sorted(record.links))
    conn.executemany(
        "INSERT OR REPLACE INTO score (key, topic, topic_mark, model, similarity, keyword_hits, scan)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(k, t, m, record.embed_model, float(s), int(h), scan_id) for k, t, m, s, h in record.scores],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO decision (scan, key, topic, score, outcome) VALUES (?, ?, ?, ?, ?)",
        [(scan_id, key, topic, score, outcome) for key, (topic, score, outcome) in record.decisions.items()],
    )
    observations = []
    for key, row in record.observations.items():
        owner = row.get("owner")
        followers = record.followers.get(owner.lower()) if owner else None
        numbers = (row.get("stars"), row.get("forks"), row.get("size_kb"), followers)
        if any(n is not None for n in numbers):
            observations.append((key, at, *numbers))
    conn.executemany(
        "INSERT OR REPLACE INTO observation (key, at, stars, forks, size_kb, followers) VALUES (?, ?, ?, ?, ?, ?)",
        observations,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO vector (key, model, data) VALUES (?, ?, ?)",
        [(key, record.embed_model, _pack(vector)) for key, vector in record.vectors.items()],
    )
    for key, columns in record.states.items():
        names = ", ".join(f"{name} = ?" for name in columns)
        conn.execute(f"UPDATE item SET {names} WHERE key = ?", (*columns.values(), key))

    # After a few days only shown, voted and labelled items need their vector.
    cutoff = (record.now - timedelta(days=VECTOR_DAYS)).astimezone(timezone.utc).isoformat()
    conn.execute(
        "DELETE FROM vector WHERE key IN (SELECT item.key FROM item WHERE item.state != 'shown'"
        " AND item.first_seen < ? AND item.key NOT IN (SELECT key FROM vote)"
        " AND item.key NOT IN (SELECT key FROM label))",
        (cutoff,),
    )


# votes ------------------------------------------------------------------------------------------


def _read_feedback(state_dir: Path) -> dict[str, int]:
    """feedback.json of an older version: vote by URL. A damaged file is kept aside, as Store does."""
    path = state_dir / FEEDBACK_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        try:
            path.replace(path.with_suffix(path.suffix + ".broken"))
        except OSError:
            pass
        return {}
    if not isinstance(data, dict):
        return {}
    return {url: v for url, v in data.items() if isinstance(url, str) and v in (1, -1)}


def _write_feedback(state_dir: Path, votes: dict[str, int]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / FEEDBACK_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(votes, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _key_for_link(conn: sqlite3.Connection, url: str) -> str | None:
    row = conn.execute("SELECT key FROM link WHERE url = ?", (url,)).fetchone() or conn.execute(
        "SELECT key FROM item WHERE url = ?", (url,)
    ).fetchone()
    return row[0] if row else None


def _import_votes(conn: sqlite3.Connection, state_dir: Path) -> bool:
    """Copy feedback.json into `vote`, inside the caller's transaction. A vote whose URL has no item
    has no vector and had no effect, so it is dropped. False when there is no file to retire."""
    path = state_dir / FEEDBACK_FILE
    if not path.exists():
        return False
    votes = _read_feedback(state_dir)
    if not path.exists():  # a damaged file was set aside
        return False
    try:
        at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        at = datetime.now(timezone.utc).isoformat()
    for url, value in votes.items():
        key = _key_for_link(conn, url)
        if key:
            conn.execute("INSERT OR REPLACE INTO vote (key, value, at, via) VALUES (?, ?, ?, 'tui')", (key, int(value), at))
    return True


def _retire_feedback(state_dir: Path) -> None:
    old = state_dir / FEEDBACK_FILE
    try:
        if old.exists():
            old.replace(old.with_name(FEEDBACK_FILE + ".migrated"))
    except OSError:
        pass  # the votes are in the store; a stale copy only matters to an older version


def _upgrade_1_to_2(conn: sqlite3.Connection, state_dir: Path) -> None:
    """Add `label` and `vote` and import feedback.json, in one transaction. The version is read
    again inside it, so two writers opening the same file upgrade it once."""
    imported = False
    try:
        conn.execute("BEGIN IMMEDIATE")
        if _version(conn) == 1:
            for statement in SCHEMA_2_TABLES.split(";"):
                if statement.strip():
                    conn.execute(statement)
            imported = _import_votes(conn, state_dir)
            conn.execute("PRAGMA user_version = 2")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    if imported:
        _retire_feedback(state_dir)


def read_votes(state_dir: Path) -> dict[str, int]:
    """Every vote by URL (every URL of a voted item). From `vote` at schema 2, else from feedback.json."""
    path = state_dir / DB_NAME
    if path.exists():
        try:
            conn = _connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                if _version(conn) >= 2:
                    votes: dict[str, int] = {}
                    for url, value in conn.execute(
                        "SELECT link.url, vote.value FROM vote JOIN link ON link.key = vote.key"
                        " UNION SELECT item.url, vote.value FROM vote JOIN item ON item.key = vote.key"
                    ):
                        votes[url] = value
                    return votes
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise StoreError(f"Could not read the votes in {path}: {exc}.") from exc
    return _read_feedback(state_dir)


def write_vote(state_dir: Path, url: str, vote: int, via: str, now: datetime | None = None) -> None:
    """Set (1 or -1) or clear (0) the vote on the item behind `url`. With no radar.db this writes
    feedback.json as before; otherwise it opens the store for writing, upgrading a schema 1 file.
    It never creates the store, and a URL with no item is dropped."""
    path = state_dir / DB_NAME
    at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()

    def to_file() -> None:
        votes = _read_feedback(state_dir)
        votes.pop(url, None)
        if vote:
            votes[url] = vote
        _write_feedback(state_dir, votes)

    if not path.exists():
        to_file()
        return
    try:
        conn = _connect(str(path))
        try:
            version = _version(conn)
            if version > SCHEMA_VERSION:
                raise StoreError(
                    f"{path} was made by a newer version of Research Radar (schema {version}). Run `radar update`."
                )
            if version == 0:  # an empty file; the next scan builds the store
                to_file()
                return
            if version == 1:
                _upgrade_1_to_2(conn, state_dir)
            key = _key_for_link(conn, url)
            if key is None:
                return
            if vote:
                conn.execute("INSERT OR REPLACE INTO vote (key, value, at, via) VALUES (?, ?, ?, ?)", (key, vote, at, via))
            else:
                conn.execute("DELETE FROM vote WHERE key = ?", (key,))
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise StoreError(f"Could not write the vote to {path}: {exc}.") from exc


# migration from seen.json and vectors.json ------------------------------------------------------


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} does not hold a mapping")
    return data


def _topics_by_url(state_dir: Path) -> dict[str, str]:
    """The topic each URL was last reported under, from the day files, rising reports left out.
    The files are read directly, so building the store never changes them."""
    topics: dict[str, str] = {}
    for path in sorted((state_dir / "reports").glob("*.json")):
        try:
            day = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for report in day.get("reports", []) if isinstance(day, dict) else []:
            if isinstance(report, dict) and report.get("url") and report.get("topic") and not report.get("rising"):
                topics[report["url"]] = report["topic"]
    return topics


def _kind(key: str, source: str | None) -> str:
    return _KIND_BY_SOURCE.get(source or "") or _KIND_BY_PREFIX.get(key.split(":", 1)[0], "repo")


def _build(conn: sqlite3.Connection, state_dir: Path, embed_model: str, now: datetime) -> None:
    """Create the schema and fill it from seen.json and vectors.json, inside the caller's transaction."""
    for statement in SCHEMA.split(";"):  # executescript would commit first, so one statement at a time
        if statement.strip():
            conn.execute(statement)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    seen = _read_json(state_dir / "seen.json")
    vectors = _read_json(state_dir / "vectors.json")
    topics = _topics_by_url(state_dir)
    today = now.astimezone().date().isoformat()
    at = now.astimezone(timezone.utc).isoformat()

    keys: dict[str, str] = {}  # url: key
    # The earliest entry of a key gives its URL, so the link shown stays the first one.
    for url, entry in sorted(seen.items(), key=lambda pair: (str(pair[1].get("date", "")), pair[0])):
        if not isinstance(entry, dict):
            raise ValueError(f"seen.json entry for {url} is not a mapping")
        source = entry.get("source")
        key = key_from_url(url, source)
        keys[url] = key
        link = (key, source or "", url)
        if conn.execute("SELECT 1 FROM item WHERE key = ?", (key,)).fetchone():
            conn.execute("INSERT OR IGNORE INTO link (key, source, url) VALUES (?, ?, ?)", link)
            continue
        shown_on = str(entry.get("date") or today)
        held_until = None
        if entry.get("gone"):
            state, state_on = "gone", str(entry["gone"])
        elif entry.get("dropped"):
            state, state_on = "dropped", str(entry["dropped"])
        elif entry.get("held"):
            state, state_on = "held", str(entry["held"])
            held_until = _plus_days(state_on, HOLD_DAYS, today)
        else:
            state, state_on = "shown", shown_on
        meta = {k: entry[k] for k in ("full_name", "stars") if entry.get(k) is not None}
        try:
            first_seen = datetime.fromisoformat(shown_on).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            first_seen = at
        conn.execute(
            "INSERT INTO item (key, kind, url, title, text, meta, first_seen, last_seen, state, state_on, topic,"
            " shown_stars, held_until) VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                _kind(key, source),
                url,
                str(entry.get("title") or url),
                json.dumps(meta, ensure_ascii=False),
                first_seen,
                first_seen,
                state,
                state_on,
                topics.get(url),
                _number(entry.get("stars")),
                held_until,
            ),
        )
        conn.execute("INSERT OR IGNORE INTO link (key, source, url) VALUES (?, ?, ?)", link)

    for url, vector in vectors.items():
        key = keys.get(url) or key_from_url(url)
        if not conn.execute("SELECT 1 FROM item WHERE key = ?", (key,)).fetchone():
            # A vector without an entry in seen.json; a vote may still need it.
            conn.execute(
                "INSERT INTO item (key, kind, url, title, text, first_seen, last_seen, state, state_on)"
                " VALUES (?, ?, ?, ?, '', ?, ?, 'seen', ?)",
                (key, _kind(key, None), url, url, at, at, today),
            )
            conn.execute("INSERT OR IGNORE INTO link (key, source, url) VALUES (?, '', ?)", (key, url))
        conn.execute(
            "INSERT OR REPLACE INTO vector (key, model, data) VALUES (?, ?, ?)", (key, embed_model, _pack(vector))
        )

    _import_votes(conn, state_dir)  # _persist retires feedback.json together with the other two files
