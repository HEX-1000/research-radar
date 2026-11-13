"""Step E: store schema 2, the upgrade from schema 1, and votes read and written through the store.

`data/schema1.radar.db` was written by the scan of the code before this step (SCHEMA_VERSION = 1)
against the mocked sources of conftest.py, so the upgrade is tested on a file the old code made."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_radar import candidates
from research_radar.candidates import CandidateStore, StoreError
from research_radar.pipeline import scan
from research_radar.store import Store

from conftest import FakeOllama, db_rows, write_state

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "data" / "schema1.radar.db"
ARXIV = "https://arxiv.org/abs/2610.12345"
HF_COPY = "https://huggingface.co/papers/2610.12345"  # a second link of the same paper
KIT = "https://github.com/alice/local-llm-kit"
BOB = "https://github.com/bob/cat-photo-app"
TABLES = ("item", "link", "scan", "score", "decision", "observation", "vector")


def run(cfg, http, now=NOW, **kwargs):
    return scan(cfg, http=http, ollama=FakeOllama(), now=now, **kwargs)


def dump(path: Path, tables=TABLES) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(path)
    try:
        return {t: sorted(conn.execute(f"SELECT * FROM {t}").fetchall()) for t in tables}
    finally:
        conn.close()


def user_version(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def schema1_state(cfg, votes=None) -> Path:
    """A state folder holding the schema 1 file the old code made, and its feedback.json."""
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, cfg.state_dir / "radar.db")
    conn = sqlite3.connect(cfg.state_dir / "radar.db")
    conn.execute("INSERT INTO link (key, source, url) VALUES ('arxiv:2610.12345', 'huggingface-papers', ?)", (HF_COPY,))
    conn.commit()
    conn.close()
    if votes is not None:
        write_state(cfg, "feedback.json", votes)
    return cfg.state_dir / "radar.db"


def open_store(cfg, write=True):
    return CandidateStore.open(cfg.state_dir, embed_model=cfg.settings["embed_model"], now=NOW, write=write)


def snapshot(folder: Path) -> dict[str, str]:
    return {
        str(p.relative_to(folder)): hashlib.sha1(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


# the fixture itself ---------------------------------------------------------------------------


def test_the_fixture_is_a_schema_1_file():
    assert user_version(FIXTURE) == 1
    conn = sqlite3.connect(FIXTURE)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    conn.close()
    assert names == set(TABLES)


# upgrade from schema 1 ---------------------------------------------------------------------------


def test_a_schema_1_file_made_by_the_old_code_is_upgraded_in_place(make_cfg):
    cfg = make_cfg()
    db = schema1_state(cfg, {ARXIV: 1, BOB: -1, "https://github.com/unknown/repo": 1})
    feedback = cfg.state_dir / "feedback.json"
    mtime = datetime.fromtimestamp(feedback.stat().st_mtime, timezone.utc)
    before = dump(db)
    assert before["item"] and before["vector"]

    with open_store(cfg) as store:
        assert (store.schema, store.create) == (2, False)
    assert user_version(db) == 2
    assert dump(db) == before  # every row stays
    votes = db_rows(cfg, "SELECT * FROM vote ORDER BY key")
    assert [(v["key"], v["value"], v["via"]) for v in votes] == [
        ("arxiv:2610.12345", 1, "tui"),
        ("github:bob/cat-photo-app", -1, "tui"),
    ]  # the vote for an unknown URL has no item and is dropped
    assert datetime.fromisoformat(votes[0]["at"]) == mtime
    assert db_rows(cfg, "SELECT * FROM label") == []
    assert not feedback.exists() and (cfg.state_dir / "feedback.json.migrated").exists()

    again = dump(db)
    with open_store(cfg):
        pass
    assert dump(db) == again  # a second open does nothing


def test_the_upgrade_without_feedback_json_adds_empty_tables(make_cfg):
    cfg = make_cfg()
    db = schema1_state(cfg)
    before = dump(db)
    with open_store(cfg):
        pass
    assert user_version(db) == 2 and dump(db) == before
    assert db_rows(cfg, "SELECT * FROM vote") == []


def test_a_schema_1_file_is_never_rebuilt_from_old_json_files(make_cfg):
    cfg = make_cfg()
    db = schema1_state(cfg)
    write_state(cfg, "seen.json", {"https://github.com/stale/entry": {"date": "2026-10-01", "title": "x", "source": "github"}})
    before = dump(db)
    with open_store(cfg) as store:
        assert store.create is False
    assert dump(db) == before
    assert (cfg.state_dir / "seen.json").exists()


def test_a_schema_0_file_is_rebuilt_from_the_json_files(make_cfg):
    cfg = make_cfg()
    cfg.state_dir.mkdir(parents=True)
    sqlite3.connect(cfg.state_dir / "radar.db").close()  # an empty file, user_version 0
    write_state(cfg, "seen.json", {KIT: {"date": "2026-10-20", "title": "alice/local-llm-kit", "source": "github"}})
    write_state(cfg, "feedback.json", {KIT: 1})
    with open_store(cfg) as store:
        assert (store.create, store.schema) == (True, 2)
        assert store.conn.execute("SELECT key, value, via FROM vote").fetchall() == [("github:alice/local-llm-kit", 1, "tui")]


def test_a_failed_upgrade_changes_nothing(make_cfg, monkeypatch):
    cfg = make_cfg()
    db = schema1_state(cfg, {ARXIV: 1})
    before, files = dump(db), snapshot(cfg.state_dir)

    def broken(conn, state_dir):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(candidates, "_import_votes", broken)
    with pytest.raises(StoreError, match="disk full"):
        open_store(cfg)
    assert user_version(db) == 1 and dump(db) == before
    assert snapshot(cfg.state_dir) == files  # feedback.json stays too


def test_a_preview_reads_a_schema_1_store_and_its_votes(make_cfg):
    cfg = make_cfg()
    schema1_state(cfg, {BOB: -1, ARXIV: 1})
    files = snapshot(cfg.state_dir)
    model = cfg.settings["embed_model"]
    with open_store(cfg, write=False) as store:
        assert store.schema == 1
        likes, dislikes = store.vote_vectors(model)
    assert len(likes) == 1 and len(dislikes) == 1
    assert candidates.read_votes(cfg.state_dir) == {BOB: -1, ARXIV: 1}
    assert snapshot(cfg.state_dir) == files  # nothing written, nothing renamed


# built from the JSON files ----------------------------------------------------------------------


def test_a_store_built_from_json_is_schema_2_with_the_votes(make_cfg, http_factory):
    cfg = make_cfg()
    write_state(cfg, "seen.json", {KIT: {"date": "2026-10-20", "title": "alice/local-llm-kit", "source": "github"}})
    write_state(cfg, "vectors.json", {})
    write_state(cfg, "feedback.json", {KIT: -1, "https://github.com/gone/missing": 1})
    run(cfg, http_factory())
    db = cfg.state_dir / "radar.db"
    assert user_version(db) == 2
    votes = db_rows(cfg, "SELECT key, value, via FROM vote")
    assert votes == [{"key": "github:alice/local-llm-kit", "value": -1, "via": "tui"}]
    assert not (cfg.state_dir / "feedback.json").exists()
    assert (cfg.state_dir / "feedback.json.migrated").exists()
    assert Store(cfg.state_dir).votes() == {KIT: -1}


def test_a_scan_upgrades_a_schema_1_store_and_counts_its_votes(make_cfg, http_factory):
    cfg = make_cfg()
    db = schema1_state(cfg, {BOB: -1})
    run(cfg, http_factory(), now=NOW + timedelta(days=1))
    assert user_version(db) == 2
    assert db_rows(cfg, "SELECT key, value FROM vote") == [{"key": "github:bob/cat-photo-app", "value": -1}]
    assert not (cfg.state_dir / "feedback.json").exists()


# votes from the interface --------------------------------------------------------------------------


def test_a_vote_without_a_store_goes_to_feedback_json_and_creates_no_store(make_cfg):
    cfg = make_cfg()
    cfg.state_dir.mkdir(parents=True)
    store = Store(cfg.state_dir)
    assert store.set_vote("2026-10-27", KIT, 1) == 1
    assert json.loads((cfg.state_dir / "feedback.json").read_text(encoding="utf-8")) == {KIT: 1}
    assert not (cfg.state_dir / "radar.db").exists()
    assert store.set_vote("2026-10-27", KIT, 1) == 0  # the same button clears it
    assert store.votes() == {}
    assert store.set_vote("2026-10-27", KIT, -1) == -1
    assert store.votes() == {KIT: -1}


def test_a_vote_with_a_store_goes_to_the_vote_table_and_toggles(make_cfg, http_factory):
    cfg = make_cfg()
    result = run(cfg, http_factory())
    store = Store(cfg.state_dir)
    url = result.new_reports[0].url
    key = db_rows(cfg, "SELECT key FROM link WHERE url = ?", (url,))[0]["key"]

    assert store.set_vote(result.date, url, 1) == 1
    row = db_rows(cfg, "SELECT * FROM vote")[0]
    assert (row["key"], row["value"], row["via"]) == (key, 1, "tui")
    assert datetime.fromisoformat(row["at"]).tzinfo is not None
    assert store.set_vote(result.date, url, 1) == 0
    assert db_rows(cfg, "SELECT * FROM vote") == []  # a cleared vote is deleted
    assert store.set_vote(result.date, url, -1) == -1
    assert store.set_vote(result.date, url, 1) == 1  # the other button replaces it
    assert [r["value"] for r in db_rows(cfg, "SELECT value FROM vote")] == [1]
    assert store.votes()[url] == 1
    assert store.load_day(result.date)["reports"][0].vote == 1
    assert not (cfg.state_dir / "feedback.json").exists()


def test_a_vote_for_a_url_with_no_item_is_dropped(make_cfg, http_factory):
    cfg = make_cfg()
    result = run(cfg, http_factory())
    store = Store(cfg.state_dir)
    store.set_vote(result.date, "https://example.org/never-seen", 1)
    assert db_rows(cfg, "SELECT * FROM vote") == []
    assert "https://example.org/never-seen" not in store.votes()


def test_a_vote_on_a_schema_1_store_upgrades_it_first(make_cfg):
    cfg = make_cfg()
    db = schema1_state(cfg, {BOB: -1})
    Store(cfg.state_dir).set_vote("2026-10-27", KIT, 1)
    assert user_version(db) == 2
    assert {r["key"]: r["value"] for r in db_rows(cfg, "SELECT key, value FROM vote")} == {
        "github:bob/cat-photo-app": -1,
        "github:alice/local-llm-kit": 1,
    }
    assert not (cfg.state_dir / "feedback.json").exists()


def test_votes_are_read_for_every_url_of_the_voted_item(make_cfg):
    cfg = make_cfg()
    schema1_state(cfg)
    Store(cfg.state_dir).set_vote("2026-10-27", HF_COPY, -1)  # voted through the second link
    assert Store(cfg.state_dir).votes() == {ARXIV: -1, HF_COPY: -1}


def test_votes_feed_the_vectors_and_a_telegram_vote_has_its_via(make_cfg, http_factory):
    cfg = make_cfg()
    result = run(cfg, http_factory())
    url = result.new_reports[0].url
    candidates.write_vote(cfg.state_dir, url, -1, "telegram", now=NOW)
    assert db_rows(cfg, "SELECT via, at FROM vote") == [{"via": "telegram", "at": NOW.isoformat()}]
    model = cfg.settings["embed_model"]
    with open_store(cfg, write=False) as store:
        likes, dislikes = store.vote_vectors(model)
    assert likes == [] and len(dislikes) == 1


def test_every_connection_waits_for_a_busy_store(make_cfg):
    cfg = make_cfg()
    db = schema1_state(cfg)
    blocker = sqlite3.connect(db, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        conn = candidates._connect(str(db))
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == candidates.BUSY_TIMEOUT * 1000
        conn.close()
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


# retention -----------------------------------------------------------------------------------------


def test_retention_keeps_the_vector_of_a_labelled_item(make_cfg, http_factory):
    cfg = make_cfg(max_items=2)
    first = run(cfg, http_factory(), now=NOW - timedelta(days=40))
    keys = {r["key"] for r in db_rows(cfg, "SELECT key FROM vector")}
    shown = {r["key"] for r in db_rows(cfg, "SELECT key FROM item WHERE state = 'shown'")}
    others = sorted(keys - shown)
    assert len(others) >= 2 and first.date
    labelled, voted, plain = others[0], others[1], others[2:]
    conn = sqlite3.connect(cfg.state_dir / "radar.db")
    conn.execute("INSERT INTO label (key, topic, value, at) VALUES (?, '*', 'lure', ?)", (labelled, NOW.isoformat()))
    conn.execute("INSERT INTO vote (key, value, at, via) VALUES (?, 1, ?, 'tui')", (voted, NOW.isoformat()))
    conn.commit()
    conn.close()

    run(cfg, http_factory())
    after = {r["key"] for r in db_rows(cfg, "SELECT key FROM vector")}
    assert {labelled, voted} <= after
    assert not (set(plain) & after)
    assert shown & keys <= after
