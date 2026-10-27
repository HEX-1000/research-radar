"""A topic that is new or was edited gets a one time look back."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from research_radar.config import Topic
from research_radar.http import Http
from research_radar.pipeline import scan, topic_fingerprint
from research_radar.store import Store

from conftest import FakeOllama, GITHUB_SEARCH, make_handler

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

COOKING = {
    "full_name": "eve/cooking-app",
    "html_url": "https://github.com/eve/cooking-app",
    "description": "Recipes and cooking planner for the kitchen",
    "stargazers_count": 60,
    "language": "Python",
    "created_at": "2026-10-25T08:00:00Z",
    "topics": [],
    "license": {"spdx_id": "MIT"},
}
OLD_LLM = {
    "full_name": "dan/old-llm-thing",
    "html_url": "https://github.com/dan/old-llm-thing",
    "description": "Run a local llm with embeddings and retrieval, no cloud needed",
    "stargazers_count": 90,
    "language": "Python",
    "created_at": "2026-10-25T09:00:00Z",
    "topics": [],
    "license": {"spdx_id": "MIT"},
}


def cooking_topic() -> Topic:
    return Topic(
        name="Cooking",
        description="Recipes, cooking and kitchen planning tools.",
        keywords=["cooking", "recipes"],
        options={"github": {"queries": ["cooking"]}},
    )


def lookback_http(calls: list[str], fail: set[str] | None = None) -> Http:
    """GitHub only returns the two extra repositories when the query reaches back past the 26th."""
    base = make_handler(fail=fail)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.url.host}{request.url.path}")
        is_search = request.url.host == "api.github.com" and request.url.path == "/search/repositories"
        query = request.url.params.get("q", "")
        if is_search and query.startswith("cooking") and request.url.host not in (fail or set()):
            reaches_back = "created:>=2026-10-24" in query
            return httpx.Response(200, json={"items": [COOKING, OLD_LLM] if reaches_back else []})
        return base(request)

    return Http(transport=httpx.MockTransport(handler), sleep=lambda seconds: None)


def run(cfg, http, now=NOW, **kwargs):
    return scan(cfg, http=http, ollama=FakeOllama(), now=now, **kwargs)


def test_the_first_scan_only_records_the_topics(make_cfg):
    cfg = make_cfg()
    calls: list[str] = []
    result = run(cfg, lookback_http(calls))
    assert result.backfilled == []
    marks = Store(cfg.state_dir).topic_marks()
    assert set(marks) == {t.name for t in cfg.topics}


def test_a_scan_before_this_feature_does_not_look_back_for_every_topic(make_cfg):
    cfg = make_cfg()
    run(cfg, lookback_http([]))
    (cfg.state_dir / "topics.json").unlink()  # state written by version 0.2.0
    result = run(cfg, lookback_http([]), now=NOW + timedelta(hours=1))
    assert result.backfilled == []
    assert Store(cfg.state_dir).topic_marks() is not None


def test_a_new_topic_gets_a_look_back_once(make_cfg):
    cfg = make_cfg()
    run(cfg, lookback_http([]))
    cfg.topics.append(cooking_topic())

    calls: list[str] = []
    result = run(cfg, lookback_http(calls), now=NOW + timedelta(hours=1))
    titles = {r.title: r.topic for r in result.new_reports}
    assert result.backfilled == ["Cooking"]
    assert titles.get("eve/cooking-app") == "Cooking"
    assert "dan/old-llm-thing" not in titles  # the look back only answers the new topic

    again: list[str] = []
    third = run(cfg, lookback_http(again), now=NOW + timedelta(hours=2))
    assert third.backfilled == []
    assert len(again) < len(calls)


def test_an_edited_topic_gets_a_look_back(make_cfg):
    cfg = make_cfg()
    run(cfg, lookback_http([]))
    cfg.topics[0].description += " Also vector databases."
    result = run(cfg, lookback_http([]), now=NOW + timedelta(hours=1))
    assert result.backfilled == [cfg.topics[0].name]


def test_a_failed_look_back_is_tried_again_at_the_next_scan(make_cfg):
    cfg = make_cfg()
    run(cfg, lookback_http([]))
    cfg.topics.append(cooking_topic())
    broken = run(cfg, lookback_http([], fail={"api.github.com"}), now=NOW + timedelta(hours=1))
    assert broken.backfilled == []
    assert any("look back" in w for w in broken.warnings)
    assert "Cooking" not in Store(cfg.state_dir).topic_marks()

    fixed = run(cfg, lookback_http([]), now=NOW + timedelta(hours=2))
    assert fixed.backfilled == ["Cooking"]
    assert "Cooking" in Store(cfg.state_dir).topic_marks()


def test_a_short_explicit_window_is_respected_and_the_topic_stays_pending(make_cfg):
    cfg = make_cfg()
    run(cfg, lookback_http([]))
    cfg.topics.append(cooking_topic())
    short = run(cfg, lookback_http([]), now=NOW + timedelta(hours=1), days=1)
    assert short.backfilled == []
    assert "Cooking" not in Store(cfg.state_dir).topic_marks()

    wide = run(cfg, lookback_http([]), now=NOW + timedelta(hours=2), days=3)
    assert wide.backfilled == []  # three days already cover the look back
    assert "Cooking" in Store(cfg.state_dir).topic_marks()


def test_the_fingerprint_follows_the_text_of_the_topic():
    topic = cooking_topic()
    same = cooking_topic()
    assert topic_fingerprint(topic) == topic_fingerprint(same)
    same.options["github"]["queries"].append("recipes")
    assert topic_fingerprint(topic) != topic_fingerprint(same)
