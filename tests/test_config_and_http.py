from __future__ import annotations

import httpx
import pytest
import yaml

from research_radar.config import ConfigError, example_text, init_home, load_config, parse_config, resolve_home
from research_radar.http import Http, SourceError


def test_example_file_parses_and_has_two_topics(tmp_path):
    cfg = parse_config(yaml.safe_load(example_text()), tmp_path)
    assert [t.name for t in cfg.topics] == ["Local-first AI tools", "Geospatial machine learning"]
    assert cfg.source_enabled("github") and cfg.source_enabled("forgejo")
    assert cfg.topics[0].source_option("arxiv", "categories") == ["cs.IR", "cs.CL"]


def test_init_never_overwrites(tmp_path):
    target = init_home(tmp_path / "h")
    target.write_text("topics: []\n", encoding="utf-8")
    init_home(tmp_path / "h")
    assert target.read_text(encoding="utf-8") == "topics: []\n"


def test_load_config_needs_init(tmp_path):
    with pytest.raises(ConfigError, match="radar init"):
        load_config(tmp_path / "nothing")


@pytest.mark.parametrize(
    "data, message",
    [
        ({"topics": []}, "non-empty"),
        ({"topics": [{"name": "A"}]}, "no description"),
        ({"topics": [{"description": "x"}]}, "no name"),
        ({"topics": [{"name": "A", "description": "x"}, {"name": "A", "description": "y"}]}, "twice"),
        ({"sources": {"nope": {}}, "topics": [{"name": "A", "description": "x"}]}, "unknown source"),
    ],
)
def test_bad_config_gives_clear_errors(tmp_path, data, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(data, tmp_path)


def test_home_resolution_order(monkeypatch, tmp_path):
    monkeypatch.setenv("RADAR_HOME", str(tmp_path / "env"))
    assert resolve_home(str(tmp_path / "arg")) == tmp_path / "arg"
    assert resolve_home() == tmp_path / "env"
    monkeypatch.delenv("RADAR_HOME")
    assert resolve_home().name == ".research-radar"


def test_http_retries_429_then_succeeds():
    attempts = {"n": 0}
    sleeps: list[float] = []

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(200, json={"ok": True})

    http = Http(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    assert http.get_json("https://example.org/x") == {"ok": True}
    assert sleeps == [2.0]


def test_http_gives_up_with_source_error():
    http = Http(transport=httpx.MockTransport(lambda r: httpx.Response(503)), sleep=lambda s: None)
    with pytest.raises(SourceError, match="503"):
        http.get("https://example.org/x")


def test_http_spaces_requests_per_bucket():
    clock = {"t": 0.0}
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds

    http = Http(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        sleep=sleep,
        clock=lambda: clock["t"],
    )
    http.get("https://example.org/a", bucket="b", min_interval=3.0)
    http.get("https://example.org/b", bucket="b", min_interval=3.0)
    assert sleeps == [3.0]


def test_github_token_is_sent_only_to_github():
    seen: list[str | None] = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={})

    http = Http(github_token="secret", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    http.get("https://api.github.com/rate_limit")
    http.get("https://codeberg.org/api/v1/x")
    assert seen == ["Bearer secret", None]


def test_rate_limit_message_is_clear():
    def handler(request):
        return httpx.Response(403, headers={"x-ratelimit-remaining": "0"})

    http = Http(transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(SourceError, match="rate limit"):
        http.get("https://api.github.com/search/repositories")


def github_http(responses, sleeps):
    """An Http client whose GitHub answers come from a list, one per request."""
    queue = list(responses)

    def handler(request):
        return queue.pop(0)

    return Http(transport=httpx.MockTransport(handler), sleep=sleeps.append)


def test_secondary_rate_limit_waits_for_retry_after_and_then_succeeds():
    sleeps: list[float] = []
    http = github_http(
        [
            httpx.Response(403, headers={"retry-after": "7"}, json={"message": "You have exceeded a secondary rate limit."}),
            httpx.Response(200, json={"ok": True}),
        ],
        sleeps,
    )
    assert http.get_json("https://api.github.com/search/repositories") == {"ok": True}
    assert sleeps == [8.0]


def test_secondary_rate_limit_without_a_header_waits_a_minute():
    sleeps: list[float] = []
    http = github_http(
        [httpx.Response(403, json={"message": "You have exceeded a secondary rate limit."}), httpx.Response(200, json={})],
        sleeps,
    )
    http.get("https://api.github.com/search/repositories")
    assert sleeps == [60.0]


def test_primary_limit_waits_for_the_reset_when_it_is_close():
    import time as clock_module

    sleeps: list[float] = []
    soon = str(int(clock_module.time()) + 10)
    http = github_http(
        [httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": soon}), httpx.Response(200, json={})],
        sleeps,
    )
    http.get("https://api.github.com/search/repositories")
    assert len(sleeps) == 1 and 5 <= sleeps[0] <= 12


def test_a_far_reset_is_not_waited_for_and_the_message_names_the_limit():
    import time as clock_module

    sleeps: list[float] = []
    later = str(int(clock_module.time()) + 3000)
    http = github_http(
        [httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": later}, json={"message": "API rate limit exceeded"})],
        sleeps,
    )
    with pytest.raises(SourceError, match=r"rate limit.*GitHub says: API rate limit exceeded"):
        http.get("https://api.github.com/search/repositories")
    assert sleeps == []


def test_a_limit_that_never_clears_gives_up_after_the_retries():
    sleeps: list[float] = []
    busy = httpx.Response(403, headers={"retry-after": "1"}, json={"message": "secondary rate limit"})
    http = github_http([busy, busy, busy], sleeps)
    with pytest.raises(SourceError, match="rate limit"):
        http.get("https://api.github.com/search/repositories")
    assert len(sleeps) == 2


def test_other_403_errors_show_what_github_said_and_are_not_retried():
    sleeps: list[float] = []
    http = github_http([httpx.Response(403, json={"message": "Resource not accessible by personal access token"})], sleeps)
    with pytest.raises(SourceError, match=r"HTTP 403 \(Resource not accessible by personal access token\)"):
        http.get("https://api.github.com/search/repositories")
    assert sleeps == []
