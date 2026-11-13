from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from conftest import FORGEJO

from research_radar.http import Http, SourceError
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


def test_github_repo_check_reads_stars_and_license(http_factory):
    check = github.check_repos(http_factory(repo_stars=321), ["a/b", "c/d"])
    assert check.current == {n: {"stars": 321, "license": "MIT", "language": "Python"} for n in ("a/b", "c/d")}
    assert check.gone == []


def _repo_http(answers: dict[str, httpx.Response], asked: list[str]) -> Http:
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.removeprefix("/repos/")
        asked.append(name)
        return answers.get(name) or httpx.Response(200, json={"stargazers_count": 7, "license": None})

    return Http(transport=httpx.MockTransport(handler), sleep=lambda seconds: None)


def test_github_repo_check_goes_on_past_deleted_and_blocked_repositories():
    asked: list[str] = []
    answers = {
        "gone/one": httpx.Response(404, json={"message": "Not Found"}),
        "blocked/two": httpx.Response(403, json={"message": "Repository access blocked"}),
        "dmca/three": httpx.Response(451, json={"message": "Repository access blocked"}),
    }
    http = _repo_http(answers, asked)
    check = github.check_repos(http, ["a/b", "gone/one", "blocked/two", "dmca/three", "c/d"])
    http.close()
    assert check.gone == ["gone/one", "dmca/three"]  # a blocked repository is asked again next time
    assert sorted(check.current) == ["a/b", "c/d"]
    assert check.current["a/b"] == {"stars": 7, "license": None, "language": None}


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(503, text="down"),
        httpx.Response(403, json={"message": "API rate limit exceeded"}, headers={"x-ratelimit-remaining": "0"}),
    ],
)
def test_github_repo_check_stops_when_github_cannot_answer(answer):
    asked: list[str] = []
    http = _repo_http({"x/fails": answer}, asked)
    check = github.check_repos(http, ["a/b", "x/fails", "c/d"])
    http.close()
    assert "c/d" not in asked  # the rest wait for the next scan
    assert list(check.current) == ["a/b"] and check.gone == []


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


def _forgejo_run(cfg, status: int, text: str = "") -> tuple[list[httpx.Request], Exception | None, list]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, text=text) if status != 200 else httpx.Response(200, json=FORGEJO)

    http = Http(transport=httpx.MockTransport(handler), sleep=lambda seconds: None)
    try:
        return requests, None, forgejo.fetch(ctx_for(cfg, http))
    except SourceError as exc:
        return requests, exc, []
    finally:
        http.close()


def test_forgejo_stops_asking_a_server_that_refuses_searches(make_cfg, monkeypatch):
    monkeypatch.delenv("RADAR_CODEBERG_TOKEN", raising=False)
    refusal = "Search and filtering is seeing a high influx of requests, your search request will not be processed."
    cfg = make_cfg()
    terms = [t for topic in cfg.topics for t in (topic.source_option("forgejo", "queries") or topic.keywords[:2])]
    assert len(terms) > 1
    requests, error, _ = _forgejo_run(cfg, 429, refusal)
    assert {r.url.params["q"] for r in requests} == {terms[0]}  # the first query and its retries, nothing after it
    assert "refuses searches" in str(error) and "high influx of requests" in str(error)
    assert "put one in the environment variable RADAR_CODEBERG_TOKEN" in str(error)
    assert all("authorization" not in r.headers for r in requests)


def test_forgejo_sends_the_token_named_for_the_server(make_cfg, monkeypatch):
    monkeypatch.setenv("RADAR_CODEBERG_TOKEN", "tok-123")
    requests, error, items = _forgejo_run(make_cfg(), 200)
    assert error is None and [i.title for i in items] == ["codeberg.org/carol/stac-tools"]
    assert requests and all(r.headers["authorization"] == "token tok-123" for r in requests)
    assert {r.url.host for r in requests} == {"codeberg.org"}


def test_forgejo_names_a_refused_token_and_stops(make_cfg, monkeypatch):
    monkeypatch.setenv("RADAR_CODEBERG_TOKEN", "expired")
    requests, error, _ = _forgejo_run(make_cfg(), 401, "token is invalid")
    assert len(requests) == 1
    assert "did not accept the access token in RADAR_CODEBERG_TOKEN" in str(error) and "expired" not in str(error)


def test_forgejo_token_settings(make_cfg, monkeypatch):
    monkeypatch.setenv("MY_GITEA", "g")
    cfg = make_cfg()
    assert cfg.forgejo_token("git.example.org") == ("", None)  # no variable named for that server
    cfg.sources["forgejo"]["token_env"] = {"git.example.org": "MY_GITEA"}
    assert cfg.forgejo_token("git.example.org") == ("MY_GITEA", "g")
    cfg.sources["forgejo"]["token_env"] = "MY_GITEA"  # one variable for every server
    assert cfg.forgejo_token("codeberg.org") == ("MY_GITEA", "g")


def test_settle_fails_only_when_nothing_worked(make_cfg, http_factory):
    ctx = ctx_for(make_cfg(), http_factory())
    settle(ctx, "x", 1, ["one bad request"])
    assert ctx.warnings == ["x: one bad request"]
    with pytest.raises(SourceError):
        settle(ctx, "x", 0, ["all failed"])


def test_failed_source_raises(make_cfg, http_factory):
    with pytest.raises(SourceError):
        arxiv.fetch(ctx_for(make_cfg(), http_factory(fail={"export.arxiv.org"})))
