"""Step C: the candidate store, its migration from seen.json and vectors.json, and the competing rule."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from research_radar import cli, pipeline
from research_radar.candidates import CandidateStore, StoreError
from research_radar.http import Http
from research_radar.models import Item, Report, key_from_url
from research_radar.pipeline import scan, topic_fingerprint
from research_radar.store import Store

from conftest import ARXIV_ATOM, FakeOllama, db_rows, make_handler, stored_item, write_state

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
OUTCOMES = {"shown", "below_bar", "topic_full", "day_full", "held", "excluded", "lure"}


def run(cfg, http, ollama=None, now=NOW, **kwargs):
    return scan(cfg, http=http, ollama=ollama or FakeOllama(), now=now, **kwargs)


def local_day(moment: datetime) -> str:
    return moment.astimezone().date().isoformat()


def snapshot(folder: Path) -> dict[str, str]:
    return {
        str(p.relative_to(folder)): hashlib.sha1(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


# keys ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, source, key",
    [
        ("https://github.com/Alice/Local-LLM-Kit", "github", "github:alice/local-llm-kit"),
        ("https://arxiv.org/abs/2610.12345v2", "arxiv", "arxiv:2610.12345"),
        ("https://arxiv.org/abs/hep-th/9901001", "arxiv", "arxiv:hep-th/9901001"),
        ("https://huggingface.co/papers/2610.12345", "huggingface-papers", "arxiv:2610.12345"),
        ("https://huggingface.co/Org/Terrain-Segmenter", "huggingface-models", "hf:org/terrain-segmenter"),
        ("https://codeberg.org/Carol/STAC-tools", "forgejo", "forgejo:codeberg.org/carol/stac-tools"),
        ("https://codeberg.org/carol/stac-tools", None, "url:https://codeberg.org/carol/stac-tools"),
        ("https://example.org/Post", None, "url:https://example.org/Post"),
    ],
)
def test_one_key_per_item_across_sources(url, source, key):
    assert key_from_url(url, source) == key


def test_a_paper_key_comes_from_its_arxiv_id():
    paper = Item("huggingface-papers", "paper", "T", "https://huggingface.co/papers/x", meta={"arxiv_id": "2610.1v3"})
    assert paper.key == "arxiv:2610.1"


# migration -----------------------------------------------------------------------------------


def _vector(position: int) -> list[float]:
    vector = [0.0] * 256
    vector[position] = 1.0
    return vector


def _old_state(cfg) -> dict:
    """seen.json, vectors.json, a vote and a day report, as version 0.3.1 left them."""
    earlier = local_day(NOW - timedelta(days=3))
    held_on = local_day(NOW - timedelta(days=2))

    def entry(name, day=earlier, source="github", **extra):
        return {"date": day, "title": name, "source": source, "full_name": name if source == "github" else None,
                "stars": 40 if source == "github" else None, **extra}

    seen = {
        "https://github.com/Old/Shown": entry("Old/Shown"),
        "https://github.com/alice/local-llm-kit": entry("alice/local-llm-kit"),  # the search returns it again
        "https://arxiv.org/abs/2609.00001": entry("A paper", source="arxiv"),
        "https://huggingface.co/papers/2609.00001": entry("A paper", day=local_day(NOW - timedelta(days=1)), source="huggingface-papers"),
        "https://github.com/held/repo": entry("held/repo", day=held_on, held=held_on, held_reason="4 KB, 50 stars, owner with 0 followers"),
        "https://github.com/gone/repo": entry("gone/repo", gone=held_on),
        "https://github.com/dropped/repo": entry("dropped/repo", dropped=held_on),
        "https://codeberg.org/carol/old-tools": entry("codeberg.org/carol/old-tools", source="forgejo"),
    }
    write_state(cfg, "seen.json", seen)
    write_state(cfg, "vectors.json", {"https://github.com/Old/Shown": _vector(1), "https://github.com/voted/elsewhere": _vector(2)})
    write_state(cfg, "feedback.json", {"https://github.com/voted/elsewhere": -1})
    Store(cfg.state_dir).save_day(
        earlier,
        [Report(date=earlier, topic="Local-first AI tools", score=0.5, source="github", kind="repo", title="Old/Shown", url="https://github.com/Old/Shown")],
        [],
    )
    return {"earlier": earlier, "held_on": held_on}


def test_a_preview_reads_the_old_files_and_writes_nothing(make_cfg, http_factory):
    cfg = make_cfg()
    _old_state(cfg)
    before = snapshot(cfg.home)
    result = run(cfg, http_factory(), use_llm=False)
    assert result.preview and result.new_reports
    assert snapshot(cfg.home) == before  # no radar.db, and seen.json and vectors.json stay


def test_the_first_scan_moves_seen_and_vectors_into_the_store(make_cfg, http_factory):
    cfg = make_cfg()
    days = _old_state(cfg)
    model = cfg.settings["embed_model"]
    result = run(cfg, http_factory())
    # Shown by the older version and without scores in the store: its state alone keeps it out.
    assert not [r for r in result.new_reports if r.title == "alice/local-llm-kit" and not r.rising]
    assert not db_rows(cfg, "SELECT key FROM score WHERE key = 'github:alice/local-llm-kit'")

    state = cfg.state_dir
    for name in ("seen.json", "vectors.json"):
        assert not (state / name).exists() and (state / f"{name}.migrated").exists()

    shown = stored_item(cfg, "github:old/shown")
    assert (shown["state"], shown["state_on"], shown["topic"]) == ("shown", days["earlier"], "Local-first AI tools")
    assert shown["url"] == "https://github.com/Old/Shown"
    paper = stored_item(cfg, "arxiv:2609.00001")
    assert paper["url"] == "https://arxiv.org/abs/2609.00001"  # the earlier entry gives the link shown
    links = db_rows(cfg, "SELECT source, url FROM link WHERE key = 'arxiv:2609.00001' ORDER BY source")
    assert [r["source"] for r in links] == ["arxiv", "huggingface-papers"]
    held = stored_item(cfg, "github:held/repo")
    week_later = (date.fromisoformat(days["held_on"]) + timedelta(days=7)).isoformat()
    assert (held["state"], held["state_on"], held["held_until"]) == ("held", days["held_on"], week_later)
    assert stored_item(cfg, "github:gone/repo")["state"] == "gone"
    assert stored_item(cfg, "github:dropped/repo")["state"] == "dropped"
    assert stored_item(cfg, "forgejo:codeberg.org/carol/old-tools")["kind"] == "repo"

    vectors = db_rows(cfg, "SELECT key, model FROM vector WHERE key IN ('github:old/shown', 'github:voted/elsewhere')")
    assert sorted((r["key"], r["model"]) for r in vectors) == [("github:old/shown", model), ("github:voted/elsewhere", model)]
    assert stored_item(cfg, "github:voted/elsewhere")["state"] == "seen"  # a vector without an entry is kept for its vote
    with CandidateStore.open(state, embed_model=model, now=NOW, write=False) as candidates:
        likes, dislikes = candidates.vote_vectors(Store(state).votes(), model)
    assert likes == [] and dislikes == [_vector(2)]


def test_a_failed_migration_leaves_the_json_files(make_cfg, http_factory, monkeypatch, capsys):
    cfg = make_cfg()
    write_state(cfg, "seen.json", {"https://github.com/a/b": {"date": "2026-10-20", "title": "a/b", "source": "github"}})
    write_state(cfg, "vectors.json", {"https://github.com/a/b": ["not a number"]})
    before = snapshot(cfg.state_dir)
    with pytest.raises(StoreError, match="Could not create the store"):
        run(cfg, http_factory())
    assert snapshot(cfg.state_dir) == before  # no radar.db, no temporary file, no lock

    monkeypatch.setattr(pipeline, "Http", lambda **k: http_factory())
    monkeypatch.setattr(pipeline, "Ollama", lambda *a, **k: FakeOllama())
    assert cli.cmd_scan(cfg, None, False, False) == 1
    assert "error: Could not create the store" in capsys.readouterr().out


def test_a_scan_that_crashes_leaves_the_store_as_it_was(make_cfg, http_factory, monkeypatch):
    cfg = make_cfg()

    def boom(*a, **k):
        raise RuntimeError("boom")

    write_state(cfg, "seen.json", {})
    monkeypatch.setattr(pipeline, "summarize", boom)
    with pytest.raises(RuntimeError):
        run(cfg, http_factory())
    assert not (cfg.state_dir / "radar.db").exists() and (cfg.state_dir / "seen.json").exists()

    monkeypatch.undo()
    run(cfg, http_factory(fail={"export.arxiv.org"}))  # the paper is left for the next scan
    stored = (cfg.state_dir / "radar.db").read_bytes()
    monkeypatch.setattr(pipeline, "summarize", boom)
    with pytest.raises(RuntimeError):
        run(cfg, http_factory(), now=NOW + timedelta(hours=1))
    assert (cfg.state_dir / "radar.db").read_bytes() == stored


# the competing rule --------------------------------------------------------------------------


def _scores(cfg) -> list[dict]:
    return db_rows(cfg, "SELECT key, topic, topic_mark, scan FROM score ORDER BY key, topic")


def test_the_same_item_in_two_scans_is_scored_once(make_cfg, http_factory):
    cfg = make_cfg()
    first_ollama = FakeOllama()
    first = run(cfg, http_factory(), ollama=first_ollama)
    scores = _scores(cfg)
    keys = {r["key"] for r in scores}
    assert len(scores) == len(keys) * len(cfg.topics)  # every candidate against every topic
    assert len(first_ollama.embedded) == len(keys)  # and embedded once
    decisions = {r["key"]: r["outcome"] for r in db_rows(cfg, "SELECT key, outcome FROM decision")}
    assert set(decisions.values()) <= OUTCOMES
    assert {k for k, o in decisions.items() if o == "shown"} == {key_from_url(r.url, r.source) for r in first.new_reports}

    second_ollama = FakeOllama()
    second = run(cfg, http_factory(), ollama=second_ollama, now=NOW + timedelta(hours=1))
    assert second.new_reports == [] and second_ollama.embedded == []
    assert _scores(cfg) == scores
    assert db_rows(cfg, "SELECT DISTINCT scan FROM decision") == [{"scan": 1}]
    assert len(db_rows(cfg, "SELECT id FROM scan")) == 2
    stars = db_rows(cfg, "SELECT stars FROM observation WHERE key = 'github:alice/local-llm-kit' ORDER BY at")
    assert stars == [{"stars": 120}, {"stars": 120}]  # one row per scan that saw it


def test_an_edited_topic_is_scored_once_with_the_stored_vectors(make_cfg, http_factory):
    cfg = make_cfg()
    run(cfg, http_factory())
    before = _scores(cfg)
    shown_before = {r["key"] for r in db_rows(cfg, "SELECT key FROM item WHERE state = 'shown'")}
    topic = cfg.topics[0]
    topic.description += " Also vector databases."

    ollama = FakeOllama()
    run(cfg, http_factory(), ollama=ollama, now=NOW + timedelta(hours=1))
    assert ollama.embedded == []  # the vectors of the first scan are used again
    added = [r for r in _scores(cfg) if r not in before]
    assert added and {(r["topic"], r["topic_mark"]) for r in added} == {(topic.name, topic_fingerprint(topic))}
    assert not {r["key"] for r in added} & shown_before  # a shown item never competes again

    after = _scores(cfg)
    run(cfg, http_factory(), now=NOW + timedelta(hours=2))
    assert _scores(cfg) == after


def test_the_same_paper_from_arxiv_and_hugging_face_is_one_item(make_cfg):
    base = make_handler()
    twin = [{"paper": {"id": "2610.12345", "title": "Retrieval with small local embedding models",
                       "summary": "We study retrieval and context compression using embeddings that run locally.",
                       "publishedAt": "2026-10-26T17:00:00.000Z"}}]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "huggingface.co" and request.url.path == "/api/daily_papers":
            return httpx.Response(200, json=twin)
        return base(request)

    cfg = make_cfg()
    http = Http(transport=httpx.MockTransport(handler), sleep=lambda s: None)
    try:
        result = run(cfg, http)
        again = run(cfg, http, now=NOW + timedelta(hours=1))
    finally:
        http.close()
    key = "arxiv:2610.12345"
    assert "2610.12345" in ARXIV_ATOM
    assert [e.item.key for e in result.explain].count(key) == 1
    assert len([r for r in result.new_reports if key_from_url(r.url, r.source) == key]) <= 1
    assert stored_item(cfg, key)["url"] == "https://arxiv.org/abs/2610.12345"  # arXiv is read first
    links = db_rows(cfg, "SELECT source FROM link WHERE key = ? ORDER BY source", (key,))
    assert [r["source"] for r in links] == ["arxiv", "huggingface-papers"]
    assert not [r for r in again.new_reports if key_from_url(r.url, r.source) == key]


# retention -----------------------------------------------------------------------------------


def test_old_vectors_stay_only_for_shown_and_voted_items(make_cfg, http_factory):
    cfg = make_cfg(max_items=2)
    first = run(cfg, http_factory(), now=NOW - timedelta(days=40))
    bob = "https://github.com/bob/cat-photo-app"
    assert bob not in {r.url for r in first.new_reports}
    Store(cfg.state_dir).set_vote(first.date, bob, -1)
    before = {r["key"] for r in db_rows(cfg, "SELECT key FROM vector")}
    assert "github:bob/cat-photo-app" in before

    run(cfg, http_factory())
    after = {r["key"] for r in db_rows(cfg, "SELECT key FROM vector")}
    shown = {r["key"] for r in db_rows(cfg, "SELECT key FROM item WHERE state = 'shown'")}
    assert after == (before & shown) | {"github:bob/cat-photo-app"}
    assert after < before
    assert len(db_rows(cfg, "SELECT key FROM item")) > len(after)  # other rows stay
