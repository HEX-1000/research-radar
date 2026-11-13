"""The trust check: thin GitHub repositories are held back a week and then asked about again."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from research_radar.http import Http
from research_radar.pipeline import scan
from research_radar.store import Store

from conftest import GITHUB_SEARCH, FakeOllama, make_handler

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
LURE_URL = "https://github.com/lure/local-llm-kit"
LURE_TEXT = GITHUB_SEARCH["items"][0]["description"]


def _raw(name: str, size: int, stars: int, owner: str, text: str) -> dict:
    return {
        "full_name": name,
        "html_url": f"https://github.com/{name}",
        "description": text,
        "stargazers_count": stars,
        "size": size,
        "forks_count": 0,
        "owner": {"login": owner},
        "language": "Python",
        "created_at": "2026-10-26T08:00:00Z",
        "topics": ["ollama", "retrieval"],
        "license": None,
    }


GOOD = {**GITHUB_SEARCH["items"][0], "size": 5000, "owner": {"login": "alice"}}


class World:
    """GitHub as the tests need it: a search result, owners with follower counts, a repository lookup."""

    def __init__(self, search_items, followers=None, repo=None, user_fails=False):
        self.search_items = search_items
        self.followers = followers or {}
        self.repo = repo  # None: 404, an int: that status, a dict: that repository
        self.user_fails = user_fails
        self.calls: list[str] = []

    def http(self) -> Http:
        base = make_handler()

        def handler(request: httpx.Request) -> httpx.Response:
            host, path = request.url.host, request.url.path
            if host == "api.github.com":
                self.calls.append(path)
                if path == "/search/repositories":
                    return httpx.Response(200, json={"items": self.search_items})
                if path.startswith("/users/"):
                    if self.user_fails:
                        return httpx.Response(503, text="down")
                    login = path.rsplit("/", 1)[1]
                    return httpx.Response(200, json={"followers": self.followers.get(login, 0)})
                if path.endswith("/readme"):
                    return httpx.Response(200, text="# Tool\nA tool to run a local llm.")
                if path == "/repos/lure/local-llm-kit":
                    if isinstance(self.repo, dict):
                        return httpx.Response(200, json=self.repo)
                    return httpx.Response(self.repo or 404, text="gone")
            return base(request)

        return Http(transport=httpx.MockTransport(handler), sleep=lambda s: None)


def run(cfg, world, now=NOW, **kwargs):
    http = world.http()
    try:
        return scan(cfg, http=http, ollama=FakeOllama(), now=now, **kwargs)
    finally:
        http.close()


def lure_world(**kwargs):
    return World([_raw("lure/local-llm-kit", 50, 50, "lure", LURE_TEXT), GOOD], **kwargs)


def local_day(moment: datetime) -> str:
    return moment.astimezone().date().isoformat()


def test_a_thin_repository_is_held_and_the_next_candidate_takes_its_place(make_cfg):
    cfg = make_cfg(max_items=1, max_per_topic=1)
    result = run(cfg, lure_world())
    titles = [e.item.title for e in result.explain]
    assert titles.index("lure/local-llm-kit") < titles.index("alice/local-llm-kit")  # it ranked first
    assert [r.title for r in result.new_reports] == ["alice/local-llm-kit"]
    held = next(e for e in result.explain if e.item.title == "lure/local-llm-kit")
    assert held.held == "50 KB, 50 stars, owner with 0 followers" and not held.selected
    assert "Held back for a week: lure/local-llm-kit (50 KB, 50 stars, owner with 0 followers)." in result.warnings
    note = result.note_path.read_text(encoding="utf-8")
    assert "lure/local-llm-kit" in note and LURE_URL not in note  # named, never linked
    entry = Store(cfg.state_dir).seen()[LURE_URL]
    assert entry["held"] == local_day(NOW) and "owner with 0 followers" in entry["held_reason"]


def test_a_held_repository_is_not_a_candidate_again_before_the_week_is_over(make_cfg):
    cfg = make_cfg()
    world = lure_world()
    run(cfg, world)
    world.calls.clear()
    second = run(cfg, world, now=NOW + timedelta(days=3))
    assert "lure/local-llm-kit" not in {r.title for r in second.new_reports}
    assert not [w for w in second.warnings if "lure/local-llm-kit" in w]
    assert "/repos/lure/local-llm-kit" not in world.calls  # not asked about yet


def test_a_popular_owner_keeps_a_small_repository(make_cfg):
    cfg = make_cfg()
    result = run(cfg, lure_world(followers={"lure": 500}))
    assert "lure/local-llm-kit" in {r.title for r in result.new_reports}
    assert not [w for w in result.warnings if w.startswith("Held back")]


def test_a_large_or_unstarred_repository_costs_no_owner_request(make_cfg):
    cfg = make_cfg()
    world = World([_raw("big/kit", 5000, 80, "big", LURE_TEXT), _raw("new/kit", 20, 3, "new", LURE_TEXT)])
    result = run(cfg, world)
    assert {"big/kit", "new/kit"} <= {e.item.title for e in result.explain}
    assert not [c for c in world.calls if c.startswith("/users/")]
    assert not [w for w in result.warnings if w.startswith("Held back")]


def test_a_failed_owner_request_counts_as_thin(make_cfg):
    cfg = make_cfg()
    result = run(cfg, lure_world(user_fails=True))
    assert "lure/local-llm-kit" not in {r.title for r in result.new_reports}
    assert "Held back for a week: lure/local-llm-kit (50 KB, 50 stars, owner with unknown followers)." in result.warnings


def test_the_trust_check_can_be_turned_off(make_cfg):
    cfg = make_cfg(trust_check=False)
    world = lure_world()
    result = run(cfg, world)
    assert "lure/local-llm-kit" in {r.title for r in result.new_reports}
    assert not [c for c in world.calls if c.startswith("/users/")]


def _held(cfg, held_on: datetime):
    Store(cfg.state_dir).save_seen(
        {
            LURE_URL: {
                "date": local_day(held_on),
                "title": "lure/local-llm-kit",
                "source": "github",
                "full_name": "lure/local-llm-kit",
                "stars": 50,
                "held": local_day(held_on),
                "held_reason": "50 KB, 50 stars, owner with 0 followers",
            }
        }
    )


def test_recheck_after_a_week_when_the_repository_is_gone(make_cfg):
    cfg = make_cfg()
    held_on = NOW - timedelta(days=7)
    _held(cfg, held_on)
    day = held_on.astimezone().date()
    result = run(cfg, World([GOOD], repo=None))
    line = f"lure/local-llm-kit, held back on {day.day} {day.strftime('%B')}, has been removed from GitHub since."
    assert line in result.warnings
    entry = Store(cfg.state_dir).seen()[LURE_URL]
    assert entry["gone"] == local_day(NOW) and "held" not in entry


def test_recheck_when_it_is_still_thin_drops_it_without_a_line(make_cfg):
    cfg = make_cfg()
    _held(cfg, NOW - timedelta(days=7))
    still = _raw("lure/local-llm-kit", 60, 80, "lure", LURE_TEXT)
    result = run(cfg, World([GOOD], repo=still))
    assert "lure/local-llm-kit" not in {r.title for r in result.new_reports}
    assert not [w for w in result.warnings if "lure/local-llm-kit" in w]
    entry = Store(cfg.state_dir).seen()[LURE_URL]
    assert entry["dropped"] == local_day(NOW) and "held" not in entry
    again = run(cfg, World([GOOD], repo=still), now=NOW + timedelta(days=1))
    assert "lure/local-llm-kit" not in {r.title for r in again.new_reports}


def test_recheck_when_it_has_grown_makes_it_a_candidate(make_cfg):
    cfg = make_cfg()
    _held(cfg, NOW - timedelta(days=7))
    grown = _raw("lure/local-llm-kit", 4000, 300, "lure", LURE_TEXT)
    result = run(cfg, World([GOOD], repo=grown))
    assert "lure/local-llm-kit" in {r.title for r in result.new_reports}
    entry = Store(cfg.state_dir).seen()[LURE_URL]
    assert "held" not in entry and entry["stars"] == 300


def test_recheck_when_the_request_fails_asks_again_at_the_next_scan(make_cfg):
    cfg = make_cfg()
    _held(cfg, NOW - timedelta(days=7))
    run(cfg, World([GOOD], repo=503))
    assert Store(cfg.state_dir).seen()[LURE_URL]["held"] == local_day(NOW - timedelta(days=7))


def test_a_held_repository_is_not_reported_as_rising(make_cfg):
    cfg = make_cfg()
    _held(cfg, NOW - timedelta(days=2))
    world = World([GOOD])
    run(cfg, world)
    assert "/repos/lure/local-llm-kit" not in world.calls
