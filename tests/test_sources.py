from __future__ import annotations

from datetime import datetime, timezone

import pytest

from research_radar.http import SourceError
from research_radar.sources import arxiv, forgejo, github, huggingface
from research_radar.sources.base import FetchContext, settle

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
SINCE = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def ctx_for(cfg, http):
    return FetchContext(config=cfg, http=http, since=SINCE, until=NOW)


def test_github_builds_query_with_window_and_quotes():
    q = github.build_query("remote sensing", is_topic=False, since_date="2026-10-24", min_stars=5)
    assert q.startswith('"remote sensing" created:>=2026-10-24 stars:>=5')
    assert github.build_query("stac", is_topic=True, since_date="2026-10-24", min_stars=5).startswith("topic:stac ")


def test_github_fetch_parses_and_dedupes(make_cfg, http_factory):
    items = github.fetch(ctx_for(make_cfg(), http_factory()))
    assert [i.title for i in items] == ["alice/local-llm-kit", "bob/cat-photo-app"]
    first = items[0]
    assert first.meta["stars"] == 120 and first.meta["license"] == "MIT"
    assert "Topics: ollama, retrieval" in first.text
    assert items[1].meta["license"] is None


def test_github_readme_is_stripped(http_factory):
    text = github.fetch_readme(http_factory(), "alice/local-llm-kit")
    assert text == "Tool A tool to run a local llm."


def test_github_star_counts(http_factory):
    assert github.star_counts(http_factory(repo_stars=321), ["a/b", "c/d"]) == {"a/b": 321, "c/d": 321}


def test_arxiv_query_uses_categories_terms_and_dates():
    q = arxiv.build_query(["cs.IR", "cs.CL"], ["retrieval"], SINCE, NOW)
    assert q == '(cat:cs.IR OR cat:cs.CL) AND (all:"retrieval") AND submittedDate:[202610241200 TO 202610271200]'


def test_arxiv_parse_cleans_title_and_id(make_cfg, http_factory):
    items = arxiv.fetch(ctx_for(make_cfg(), http_factory()))
    assert len(items) == 1
    item = items[0]
    assert item.title == "Retrieval with small local embedding models"
    assert item.url == "https://arxiv.org/abs/2610.12345"
    assert item.kind == "paper" and item.meta["categories"] == ["cs.IR"]


def test_arxiv_bad_xml_is_a_source_error():
    with pytest.raises(SourceError, match="Atom"):
        arxiv.parse_feed("<html>not atom")


def test_huggingface_papers_and_models(make_cfg, http_factory):
    cfg = make_cfg()
    papers = huggingface.fetch_papers(ctx_for(cfg, http_factory()))
    assert papers[0].url == "https://huggingface.co/papers/2610.55555"
    assert papers[0].meta["upvotes"] == 12
    models = huggingface.fetch_models(ctx_for(cfg, http_factory()))
    assert models[0].title == "org/terrain-segmenter"
    assert "remote-sensing" in models[0].text and "region:us" not in models[0].text


def test_huggingface_shape_change_is_reported():
    with pytest.raises(SourceError):
        huggingface.parse_papers({"error": "moved"})


def test_forgejo_keeps_only_recent_non_forks(make_cfg, http_factory):
    items = forgejo.fetch(ctx_for(make_cfg(), http_factory()))
    assert [i.title for i in items] == ["codeberg.org/carol/stac-tools"]


def test_settle_fails_only_when_nothing_worked(make_cfg, http_factory):
    ctx = ctx_for(make_cfg(), http_factory())
    settle(ctx, "x", 1, ["one bad request"])
    assert ctx.warnings == ["x: one bad request"]
    with pytest.raises(SourceError):
        settle(ctx, "x", 0, ["all failed"])


def test_failed_source_raises(make_cfg, http_factory):
    with pytest.raises(SourceError):
        arxiv.fetch(ctx_for(make_cfg(), http_factory(fail={"export.arxiv.org"})))
