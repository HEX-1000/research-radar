from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from research_radar import pipeline, ranker
from research_radar.candidates import CandidateStore
from research_radar.config import Topic
from research_radar.digest import MONTHS, Digest, render_text
from research_radar.http import Http
from research_radar.models import Item, Report
from research_radar.notes import ranking_note, render_note
from research_radar.pipeline import OVERLAP_DAYS, scan, window_start
from research_radar.sources.github import RepoCheck
from research_radar.store import Store

from conftest import FakeOllama, stored_item, write_state

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


def test_without_ollama_it_is_a_preview_that_shows_items(make_cfg, http_factory):
    result = run(make_cfg(), http_factory(), ollama=FakeOllama(up=False))
    assert result.ranking == "keywords" and result.preview
    assert any("Ollama is not reachable" in w for w in result.warnings)
    assert result.new_reports  # keyword hits still produce plain notes
    assert all(r.verdict in ("watch", "read") for r in result.new_reports)
    assert result.note_path is None


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
    # Reported by an older version: the first scan moves it into the store, with its topic.
    write_state(cfg, "seen.json", {url: {"date": earlier, "title": "old/project", "source": "github", "full_name": "old/project", "stars": 40}})
    store.save_day(
        earlier,
        [Report(date=earlier, topic="Local-first AI tools", score=0.5, source="github", kind="repo", title="old/project", url=url)],
        [],
    )
    first = run(cfg, http_factory(repo_stars=300))
    rising = [r for r in first.new_reports if r.rising]
    assert len(rising) == 1 and rising[0].topic == "Local-first AI tools"
    assert "rising, from 40 to 300 stars" in first.note_path.read_text(encoding="utf-8")
    digest = render_text(Digest(date=first.date, reports=rising))
    assert "Rising: from 40 to 300 stars since it was first reported.\nlicense MIT, Python\n" in digest
    second = run(cfg, http_factory(repo_stars=300), now=NOW + timedelta(hours=1))
    assert not [r for r in second.new_reports if r.rising]


def _seen_repo(name: str, date: str) -> dict:
    return {"date": date, "title": name, "source": "github", "full_name": name, "stars": 40}


def test_rising_check_goes_on_past_a_deleted_repository(make_cfg, http_factory, monkeypatch):
    cfg = make_cfg()
    earlier = (NOW - timedelta(days=3)).astimezone().date().isoformat()
    names = ["a/first", "gone/lure", "b/second"]
    write_state(cfg, "seen.json", {f"https://github.com/{n}": _seen_repo(n, earlier) for n in names})
    asked: list[str] = []

    def check(http, full_names):
        asked.extend(full_names)
        return RepoCheck(
            current={n: {"stars": 300, "license": None, "language": None} for n in full_names if n != "gone/lure"},
            gone=[n for n in full_names if n == "gone/lure"],
        )

    monkeypatch.setattr(pipeline, "check_repos", check)
    first = run(cfg, http_factory())
    assert sorted(r.title for r in first.new_reports if r.rising) == ["a/first", "b/second"]
    removed = [w for w in first.warnings if "removed from GitHub" in w]
    day = date.fromisoformat(earlier)
    assert removed == [f"gone/lure, shown on {day.day} {MONTHS[day.month - 1]}, has been removed from GitHub since."]
    assert removed[0] in first.note_path.read_text(encoding="utf-8")
    gone = stored_item(cfg, "github:gone/lure")
    assert gone["state"] == "gone" and gone["state_on"] == first.date

    asked.clear()
    second = run(cfg, http_factory(), now=NOW + timedelta(hours=1))
    assert "gone/lure" not in asked  # a deleted repository is not asked about again
    assert not [w for w in second.warnings if "removed from GitHub" in w]


def test_two_scans_on_one_day_keep_both_sets_of_notes(make_cfg, http_factory):
    cfg = make_cfg()
    first = run(cfg, http_factory(fail={"export.arxiv.org"}))
    second = run(cfg, http_factory(), now=NOW + timedelta(hours=1))
    assert first.new_reports and first.ranking == "embeddings" and second.ranking == "embeddings"
    day = Store(cfg.state_dir).load_day(first.date)
    assert [s["ranking"] for s in day["scans"]] == ["embeddings", "embeddings"]
    assert [s.source for s in day["skipped"]] == ["arxiv"]
    assert {r.url for r in first.new_reports} <= {r.url for r in day["reports"]}
    note = second.note_path.read_text(encoding="utf-8")
    assert "- arxiv:" in note


def test_a_day_file_from_before_per_scan_notes_still_reads(make_cfg):
    cfg = make_cfg()
    store = Store(cfg.state_dir)
    old = {
        "reports": [Report(date="2026-10-30", topic="T", score=0.8, source="arxiv", kind="paper", title="p", url="u").to_dict()],
        "skipped": [{"source": "forgejo", "reason": "HTTP 429"}],
        "warnings": ["w"],
        "ranking": "keywords",
    }
    store._write(store._day_path("2026-10-30"), old)
    day = store.load_day("2026-10-30")
    assert day["ranking"] == "keywords" and day["warnings"] == ["w"] and day["skipped"][0].source == "forgejo"
    assert "Ranked by keyword matching" in render_note("2026-10-30", day["reports"], ["T"], [], [], day["ranking"])
    merged = store.merge_day("2026-10-30", [], [], ["x"], "embeddings", at="later")
    assert merged["warnings"] == ["w", "x"] and len(merged["scans"]) == 2


def test_the_keyword_line_counts_only_keyword_ranked_items():
    def item(url: str, ranking: str) -> Report:
        return Report(date="d", topic="T", score=0.5, source="arxiv", kind="paper", title=url, url=url, ranking=ranking)

    mixed = [item("a", "keywords"), item("b", "embeddings"), item("c", "embeddings")]
    assert ranking_note(mixed, "keywords") == (
        "1 of 3 items were ranked by keyword matching because the embedding model was not available."
    )
    assert ranking_note([item("b", "embeddings")], "keywords") == ""
    assert ranking_note([item("a", "keywords")], "embeddings").startswith("Ranked by keyword matching")


def test_votes_flip_and_feed_back_into_ranking(make_cfg, http_factory):
    cfg = make_cfg()
    result = run(cfg, http_factory())
    store = Store(cfg.state_dir)
    url = result.new_reports[0].url
    assert store.set_vote(result.date, url, 1) == 1
    assert store.set_vote(result.date, url, 1) == 0  # the same key twice clears it
    assert store.set_vote(result.date, url, -1) == -1
    model = cfg.settings["embed_model"]
    with CandidateStore.open(cfg.state_dir, embed_model=model, now=NOW, write=False) as candidates:
        likes, dislikes = candidates.vote_vectors(model)
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
    ranking = ranker.rank(items, [topic], embed=None, global_exclude=["cracked"])
    assert ranking.mode == "keywords"
    assert [s.item.title for s in ranking.scored] == ["a/llm-tool"]
    assert ranking.left_out == {"url:https://x/1": "excluded", "url:https://x/3": "lure"}  # "cracked" is lure wording


@pytest.mark.parametrize(
    "keyword, text, hit",
    [
        ("stac", "MiniMax-AI/mcode-stack. A coding harness", False),  # reported under Remote sensing on 30 October
        ("stac", "carol/stac-tools. Small tools for STAC catalogs", True),
        ("dem", "We demonstrate a method", False),
        ("dem", "A DEM of the valley, DEMs from lidar", True),
        ("llm", "Run local LLMs", True),
        ("mcp", "x/thing. Topics: mcp_server", True),
        ("backtest", "Backtesting engine for ETFs", True),
        ("backtest", "x/prebacktester", False),
        ("sentinel-2", "Sentinel-2A imagery", True),
    ],
)
def test_keywords_match_whole_words_or_word_starts(keyword, text, hit):
    topic = Topic(name="T", description="d", keywords=[keyword])
    item = Item("github", "repo", text.split(".")[0], "https://x/1", text)
    assert ranker.keyword_hits(item, topic) == int(hit)


def test_repositories_that_offer_cracked_software_are_dropped():
    topic = Topic(name="T", description="video editing", keywords=["davinci resolve"])
    lure = Item(
        "github", "repo", "NavyPigeonDepression/Davinci-Resolve-Studio-Full", "https://x/1",
        "Topics: davinci-resolve-crack-2026, davinci-resolve-studio-crack-windows",
    )
    keygen = Item("github", "repo", "x/Resolve-Keygen", "https://x/2", "DaVinci Resolve license key generator")
    honest = [
        Item("github", "repo", "a/davinci-resolve-scripts", "https://x/3", "Scripts for DaVinci Resolve"),
        Item("github", "repo", "b/pavement-crack-detection", "https://x/4", "Crack detection and crack patching on roads"),
        Item("github", "repo", "c/crack-free-electrodes", "https://x/6", "Crack-free battery electrodes. Topics: serial-keyboard"),
        Item("arxiv", "paper", "Crack segmentation", "https://x/5", "Free download of the full version of our DaVinci dataset"),
    ]
    assert ranker.is_lure(lure) and ranker.is_lure(keygen)
    assert not any(ranker.is_lure(i) for i in honest)
    ranking = ranker.rank([lure, keygen, *honest], [topic], embed=None)
    assert {s.item.url for s in ranking.scored} == {"https://x/3", "https://x/4", "https://x/5", "https://x/6"}
    assert set(ranking.left_out.values()) == {"lure"}


def test_note_for_an_empty_topic_and_skipped_source():
    text = render_note("2026-10-27", [], ["Topic A"], [], [], "embeddings")
    assert "## Topic A\nNothing new today." in text
    assert "## Skipped sources\n- none" in text


def test_arxiv_window_starts_five_days_before_the_last_run(make_cfg):
    cfg = make_cfg(first_run_days=3, max_catchup_days=14)
    last = (NOW - timedelta(days=2)).isoformat()
    assert window_start(cfg, last, NOW, None, OVERLAP_DAYS["arxiv"]) == NOW - timedelta(days=7)
    assert window_start(cfg, last, NOW, None, OVERLAP_DAYS.get("github", 0)) == NOW - timedelta(days=2)
    old = (NOW - timedelta(days=90)).isoformat()
    assert window_start(cfg, old, NOW, None, OVERLAP_DAYS["arxiv"]) == NOW - timedelta(days=14)  # still capped
    assert window_start(cfg, None, NOW, None, OVERLAP_DAYS["arxiv"]) == NOW - timedelta(days=3)  # no overlap on a first run


def test_a_paper_submitted_before_the_last_run_and_announced_after_it_is_found(make_cfg):
    from conftest import make_handler

    cfg = make_cfg()
    Store(cfg.state_dir).save_last_runs({"arxiv": (NOW - timedelta(days=1)).isoformat()})
    queries: list[str] = []
    inner = make_handler()

    def handler(request):
        if request.url.host == "export.arxiv.org":
            queries.append(request.url.params["search_query"])
        return inner(request)

    http = Http(transport=httpx.MockTransport(handler), sleep=lambda seconds: None)
    run(cfg, http)
    http.close()
    since = (NOW - timedelta(days=6)).strftime("%Y%m%d%H%M")  # last run, minus 5 days
    assert queries and all(f"submittedDate:[{since} TO " in q for q in queries)
