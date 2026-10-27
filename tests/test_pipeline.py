from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from research_radar import ranker
from research_radar.config import Topic
from research_radar.models import Item, Report
from research_radar.notes import render_note
from research_radar.pipeline import scan, window_start
from research_radar.store import Store

from conftest import FakeOllama

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def run(cfg, http, ollama=None, now=NOW, **kwargs):
    return scan(cfg, http=http, ollama=ollama or FakeOllama(), now=now, **kwargs)


def test_scan_picks_relevant_items_and_writes_a_note(make_cfg, http_factory):
    cfg = make_cfg()
    result = run(cfg, http_factory())
    titles = {r.title for r in result.new_reports}
    assert "alice/local-llm-kit" in titles
    assert "bob/cat-photo-app" not in titles
    assert result.skipped == []
    note = result.note_path.read_text(encoding="utf-8")
    assert note.startswith(f"# Radar {result.date}")
    assert "- **alice/local-llm-kit** (https://github.com/alice/local-llm-kit):" in note
    assert "120 stars, license MIT, Python" in note
    assert chr(0x2014) not in note


def test_a_second_scan_reports_nothing_twice(make_cfg, http_factory):
    cfg = make_cfg()
    first = run(cfg, http_factory())
    second = run(cfg, http_factory(), now=NOW + timedelta(hours=1))
    assert first.new_reports
    assert second.new_reports == []
    assert len(second.all_reports) == len(first.new_reports)  # the day's note keeps the earlier items


def test_caps_per_topic_and_total(make_cfg, http_factory):
    cfg = make_cfg(max_items=2, max_per_topic=1)
    result = run(cfg, http_factory())
    assert len(result.new_reports) <= 2
    topics = [r.topic for r in result.new_reports]
    assert len(topics) == len(set(topics))


def test_nothing_is_padded_when_the_bar_is_high(make_cfg, http_factory):
    result = run(make_cfg(min_score=0.99), http_factory())
    assert result.new_reports == []
    assert "Nothing new today." in result.note_path.read_text(encoding="utf-8")


def test_failed_source_is_listed_and_keeps_its_window(make_cfg, http_factory):
    cfg = make_cfg()
    result = run(cfg, http_factory(fail={"export.arxiv.org"}))
    assert [s.source for s in result.skipped] == ["arxiv"]
    assert "arxiv" not in Store(cfg.state_dir).last_runs()
    assert "github" in Store(cfg.state_dir).last_runs()
    assert "- arxiv:" in result.note_path.read_text(encoding="utf-8")


def test_without_ollama_it_falls_back_to_keywords(make_cfg, http_factory):
    result = run(make_cfg(), http_factory(), ollama=FakeOllama(up=False))
    assert result.ranking == "keywords"
    assert any("Ollama is not reachable" in w for w in result.warnings)
    assert result.new_reports  # keyword hits still produce notes
    assert all(r.verdict in ("watch", "read") for r in result.new_reports)
    assert "keyword matching" in result.note_path.read_text(encoding="utf-8")


def test_model_answers_are_used_but_verdict_is_checked(make_cfg, http_factory):
    odd = FakeOllama(chat_answer={"what": "A paper about retrieval.", "why": "", "verdict": "install"})
    result = run(make_cfg(), http_factory(), ollama=odd)
    papers = [r for r in result.new_reports if r.kind == "paper"]
    assert papers and all(r.verdict == "read" for r in papers)  # a paper cannot be installed
    assert all(r.why for r in result.new_reports)  # an empty "why" gets a plain default


def test_links_and_numbers_come_from_the_source_not_the_model(make_cfg, http_factory):
    liar = FakeOllama(chat_answer={"what": "See https://evil.example with 9999 stars.", "why": "x", "verdict": "watch"})
    result = run(make_cfg(), http_factory(), ollama=liar)
    repo = next(r for r in result.new_reports if r.title == "alice/local-llm-kit")
    assert repo.url == "https://github.com/alice/local-llm-kit"
    assert repo.meta["stars"] == 120


def test_window_uses_last_run_and_is_capped(make_cfg):
    cfg = make_cfg(first_run_days=3, max_catchup_days=14)
    assert window_start(cfg, None, NOW, None) == NOW - timedelta(days=3)
    assert window_start(cfg, (NOW - timedelta(days=2)).isoformat(), NOW, None) == NOW - timedelta(days=2)
    assert window_start(cfg, (NOW - timedelta(days=90)).isoformat(), NOW, None) == NOW - timedelta(days=14)
    assert window_start(cfg, None, NOW, 7) == NOW - timedelta(days=7)


def test_rising_repository_is_reported_once(make_cfg, http_factory):
    cfg = make_cfg()
    store = Store(cfg.state_dir)
    url = "https://github.com/old/project"
    earlier = (NOW - timedelta(days=3)).astimezone().date().isoformat()
    store.save_seen({url: {"date": earlier, "title": "old/project", "source": "github", "full_name": "old/project", "stars": 40}})
    store.save_day(
        earlier,
        [Report(date=earlier, topic="Local-first AI tools", score=0.5, source="github", kind="repo", title="old/project", url=url)],
        [],
        [],
        "embeddings",
    )
    first = run(cfg, http_factory(repo_stars=300))
    rising = [r for r in first.new_reports if r.rising]
    assert len(rising) == 1 and rising[0].topic == "Local-first AI tools"
    assert "rising, from 40 to 300 stars" in first.note_path.read_text(encoding="utf-8")
    second = run(cfg, http_factory(repo_stars=300), now=NOW + timedelta(hours=1))
    assert not [r for r in second.new_reports if r.rising]


def test_votes_flip_and_feed_back_into_ranking(make_cfg, http_factory):
    cfg = make_cfg()
    result = run(cfg, http_factory())
    store = Store(cfg.state_dir)
    url = result.new_reports[0].url
    assert store.set_vote(result.date, url, 1) == 1
    assert store.set_vote(result.date, url, 1) == 0  # the same key twice clears it
    assert store.set_vote(result.date, url, -1) == -1
    likes, dislikes = store.vote_vectors()
    assert likes == [] and len(dislikes) == 1
    assert store.load_day(result.date)["reports"][0].vote in (-1, 0)


def test_ranker_feedback_pushes_similar_items_down():
    vec = [1.0, 0.0, 0.0]
    assert ranker.feedback_adjust(vec, [], [[1.0, 0.0, 0.0]], 0.2) == pytest.approx(-0.1)
    assert ranker.feedback_adjust(vec, [[1.0, 0.0, 0.0]], [], 0.2) == pytest.approx(0.1)
    assert ranker.feedback_adjust(vec, [], [], 0.2) == 0.0


def test_global_and_topic_excludes():
    topic = Topic(name="T", description="local llm", keywords=["llm"], exclude=["casino"])
    items = [
        Item("github", "repo", "a/llm-casino", "https://x/1", "local llm casino"),
        Item("github", "repo", "a/llm-tool", "https://x/2", "local llm helper"),
        Item("github", "repo", "a/llm-cracked", "https://x/3", "cracked local llm"),
    ]
    scored, mode = ranker.rank(items, [topic], embed=None, global_exclude=["cracked"])
    assert mode == "keywords"
    assert [s.item.title for s in scored] == ["a/llm-tool"]


def test_note_for_an_empty_topic_and_skipped_source():
    text = render_note("2026-10-27", [], ["Topic A"], [], [], "embeddings")
    assert "## Topic A\nNothing new today." in text
    assert "## Skipped sources\n- none" in text
