"""Step F: per-topic min_score and max_per_topic, choice by margin, the bar in --explain."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import yaml

from conftest import FakeOllama, fixture_text
from research_radar import cli
from research_radar import pipeline
from research_radar.config import ConfigError, Topic, parse_config
from research_radar.models import Item
from research_radar.pipeline import scan, topic_fingerprint
from research_radar.ranker import Scored, rank, select

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def entry(name, score, bar, key=None):
    key = key or f"{name}-{score}"
    return Scored(Item("github", "repo", key, f"https://github.com/x/{key}"), Topic(name, "d"), score, bar)


def config(tmp_path, topic_patches=None, **settings):
    data = yaml.safe_load(fixture_text())
    data["settings"].update(settings)
    for patch, topic in zip(topic_patches or [], data["topics"]):
        topic.update(patch)
    return parse_config(data, tmp_path / "home")


# config ---------------------------------------------------------------------------------------


def test_topic_keys_are_read_and_default_to_none(tmp_path):
    cfg = config(tmp_path, [{"min_score": 0.4, "max_per_topic": 0}])
    assert (cfg.topics[0].min_score, cfg.topics[0].max_per_topic) == (0.4, 0)
    assert (cfg.topics[1].min_score, cfg.topics[1].max_per_topic) == (None, None)
    assert not [n for n in cfg.notices if "min_score" in n or "max_per_topic" in n]  # known keys


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"min_score": 1.5}, "topic 'Local-first AI tools': min_score must be a number from 0 to 1"),
        ({"min_score": -0.1}, "min_score must be a number from 0 to 1"),
        ({"min_score": "0.4"}, "min_score must be a number from 0 to 1"),
        ({"min_score": True}, "min_score must be a number from 0 to 1"),
        ({"max_per_topic": -1}, "topic 'Local-first AI tools': max_per_topic must be a whole number"),
        ({"max_per_topic": 1.5}, "max_per_topic must be a whole number"),
        ({"max_per_topic": "2"}, "max_per_topic must be a whole number"),
        ({"max_per_topic": False}, "max_per_topic must be a whole number"),
    ],
)
def test_a_bad_topic_value_raises_with_the_topic_name(tmp_path, patch, message):
    with pytest.raises(ConfigError, match=message):
        config(tmp_path, [patch])


def test_adding_either_key_does_not_change_the_fingerprint():
    plain = Topic("t", "a description", keywords=["k"])
    with_keys = Topic("t", "a description", keywords=["k"], min_score=0.3, max_per_topic=1)
    assert topic_fingerprint(plain) == topic_fingerprint(with_keys)


# rank and select -----------------------------------------------------------------------------

VECTORS = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}


def embedder(item_vectors):
    """Topic texts get a fixed axis (alpha, beta); an item text starting with a key gets its vector."""

    def embed(texts, kind="document"):
        out = []
        for text in texts:
            axis = next((v for k, v in VECTORS.items() if k in text.lower() and kind == "query"), None)
            out.append(axis or next(v for k, v in item_vectors.items() if text.startswith(k)))
        return out

    return embed


def test_the_margin_picks_the_topic():
    a, b = Topic("A", "alpha"), Topic("B", "beta")
    item = Item("github", "repo", "kit", "https://github.com/x/kit")
    embed = embedder({"kit": [0.8, 0.6]})  # cosine 0.8 with A, 0.6 with B

    def pick(bars):
        return rank([item], [a, b], embed=embed, bars=bars).scored[0]

    assert pick({"A": 0.0, "B": 0.0}).topic.name == "A"  # equal bars: the best score, as before
    assert pick({"A": 0.5, "B": 0.5}).topic.name == "A"
    moved = pick({"A": 0.7, "B": 0.3})  # margins 0.1 and 0.3
    assert moved.topic.name == "B"
    assert moved.score == pytest.approx(0.6)
    assert moved.bar == 0.3
    assert moved.margin == pytest.approx(0.3)


def test_the_ranking_is_sorted_by_margin_not_score():
    a, b = Topic("A", "alpha"), Topic("B", "beta")
    hi = Item("github", "repo", "hi", "https://github.com/x/hi")
    lo = Item("github", "repo", "lo", "https://github.com/x/lo")
    embed = embedder({"hi": [1.0, 0.0], "lo": [0.0, 1.0]})  # hi scores 1.0 in A, lo scores 1.0 in B
    ranking = rank([hi, lo], [a, b], embed=embed, bars={"A": 0.9, "B": 0.2})
    assert [s.item.title for s in ranking.scored] == [lo.title, hi.title]  # margins 0.8 and 0.1


def test_a_topic_bar_lets_an_item_through_that_the_global_bar_stops():
    entries = [entry("A", 0.45, 0.40), entry("B", 0.45, 0.55)]
    chosen = select(entries, bars={"A": 0.40, "B": 0.55}, max_items=8, caps={"A": 3, "B": 3})
    assert [e.topic.name for e in chosen] == ["A"]
    assert entries[1].outcome == "below_bar"


def test_max_items_keeps_the_items_that_clear_their_bars_by_the_most():
    entries = [entry("A", 0.60, 0.50, "a"), entry("B", 0.50, 0.30, "b")]
    entries.sort(key=lambda e: e.margin, reverse=True)  # rank does this
    chosen = select(entries, bars={"A": 0.5, "B": 0.3}, max_items=1, caps={"A": 3, "B": 3})
    assert [e.item.title for e in chosen] == ["b"]
    assert entries[1].outcome == "day_full"


def test_per_topic_caps_apply_by_topic():
    entries = [entry("A", 0.9, 0.5, "a1"), entry("A", 0.8, 0.5, "a2"), entry("B", 0.7, 0.5, "b1"), entry("B", 0.6, 0.5, "b2")]
    chosen = select(entries, bars={"A": 0.5, "B": 0.5}, max_items=8, caps={"A": 1, "B": 2})
    assert [e.item.title for e in chosen] == ["a1", "b1", "b2"]
    assert entries[1].outcome == "topic_full"


def test_a_cap_of_zero_mutes_a_topic():
    entries = [entry("A", 0.9, 0.5, "a"), entry("B", 0.7, 0.5, "b")]
    chosen = select(entries, bars={"A": 0.5, "B": 0.5}, max_items=8, caps={"A": 0, "B": 3})
    assert [e.item.title for e in chosen] == ["b"]
    assert entries[0].outcome == "topic_full"  # scored and stored, not shown


# the scan ------------------------------------------------------------------------------------


def run(cfg, http, ollama=None):
    return scan(cfg, http=http, ollama=ollama or FakeOllama(), now=NOW)


def test_a_topic_bar_and_a_zero_cap_change_what_a_scan_shows(tmp_path, http_factory):
    base = run(config(tmp_path / "a", min_score=0.2), http_factory())
    assert base.new_reports
    topic = base.new_reports[0].topic
    index = [t["name"] for t in yaml.safe_load(fixture_text())["topics"]].index(topic)
    padding = [{} for _ in range(index)]

    raised = run(config(tmp_path / "b", padding + [{"min_score": 0.99}], min_score=0.2), http_factory())
    assert topic not in {r.topic for r in raised.new_reports}
    assert not any(e.topic.name == topic and e.selected for e in raised.explain)

    muted = run(config(tmp_path / "c", padding + [{"max_per_topic": 0}], min_score=0.2), http_factory())
    assert topic not in {r.topic for r in muted.new_reports}
    assert any(e.topic.name == topic and e.outcome == "topic_full" for e in muted.explain)


def test_a_keyword_preview_uses_the_lower_of_the_bar_and_half(tmp_path, http_factory):
    high = config(tmp_path / "a", min_score=0.9)
    low = config(tmp_path / "b", [{"min_score": 0.3}, {"min_score": 0.3}], min_score=0.9)
    r1 = run(high, http_factory(), FakeOllama(up=False))
    r2 = run(low, http_factory(), FakeOllama(up=False))
    assert r1.preview and r2.preview
    assert {e.bar for e in r1.explain} == {0.5}
    assert {e.bar for e in r2.explain} == {0.3}
    assert r1.new_reports  # one keyword hit scores 0.65, above 0.5 although the setting is 0.9


def test_explain_prints_the_bar_of_each_topic(tmp_path, http_factory, capsys, monkeypatch):
    cfg = config(tmp_path, [{"min_score": 0.31}, {"min_score": 0.43}], min_score=0.2)
    real = pipeline.scan
    monkeypatch.setattr(pipeline, "scan", lambda cfg, **k: real(cfg, http=http_factory(), ollama=FakeOllama(), now=NOW, **k))
    assert cli.cmd_scan(cfg, None, False, True) == 0
    out = capsys.readouterr().out
    assert "score   bar  hits  chosen  topic / item" in out
    bars = {"Local-first AI tools": "0.31", "Geospatial machine learning": "0.43"}
    rows = [line for line in out.splitlines() if any(f" {n} / " in line for n in bars)]
    assert rows
    for line in rows:
        name = next(n for n in bars if f" {n} / " in line)
        assert line.split()[1] == bars[name]
