"""Step K: follow-ups to step I (label, kind and fixed note by feed, entry ids), follower counts
for repositories that selection did not ask about, and `huggingface-models` off by default."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
import yaml

from conftest import GITHUB_SEARCH, FakeOllama, db_rows, fixture_text, make_handler
from research_radar import labelling, ranker
from research_radar.candidates import ScanRecord
from research_radar.config import DEFAULT_SOURCES, example_text, parse_config
from research_radar.digest import KIND_LABEL
from research_radar.http import Http
from research_radar.models import Item, Report
from research_radar.notes import meta_line
from research_radar.pipeline import MAX_UNCHOSEN_OWNERS, _ask_unchosen_owners, scan
from research_radar.ranker import Scored
from research_radar.sources import feeds
from research_radar.summarize import watch_note
from test_step_i import (  # noqa: F401  (web is a fixture)
    BLOG,
    DTK,
    LONG_TEXT,
    M2C,
    config,
    context,
    atom,
    release,
    rss,
    run,
    web,
    when,
)
from test_trust_check import NOW as TRUST_NOW
from test_trust_check import World, _raw

COMMITS = "https://github.com/matt-kempster/m2c/commits.atom"
BRANCH_COMMITS = "https://github.com/matt-kempster/m2c/commits/main.atom"
TAGS = "https://github.com/ethteck/splat/tags.atom"
M2C_TITLE = "Don't crash decompiling functions declared as data"


def commit(sha: str, days: float, title: str = M2C_TITLE, body: str = "") -> tuple[str, str, str, str]:
    return (
        f"tag:github.com,2008:Grit::Commit/{sha}|{body}",
        when(days),
        f"https://github.com/matt-kempster/m2c/commit/{sha}",
        title,
    )


# label and kind by feed (decision 25) --------------------------------------------------------------


def test_a_github_commits_url_is_labelled_owner_name_with_kind_commit(tmp_path, web):
    site = web({COMMITS: atom("Recent Commits to m2c:master", [commit("abc", 1)])})
    cfg = config(tmp_path, {"watch": [COMMITS]})
    [item] = feeds.fetch(context(cfg, site))
    assert item.kind == "commit"
    assert item.title == f"matt-kempster/m2c: {M2C_TITLE}"  # not the feed's own title
    assert item.meta["label"] == "matt-kempster/m2c"


def test_the_commits_feed_of_a_branch_is_a_commits_feed(tmp_path, web):
    site = web({BRANCH_COMMITS: atom("Recent Commits to m2c:main", [commit("abc", 1)])})
    cfg = config(tmp_path, {"watch": [BRANCH_COMMITS]})
    [item] = feeds.fetch(context(cfg, site))
    assert (item.kind, item.meta["label"]) == ("commit", "matt-kempster/m2c")


def test_a_releases_feed_is_a_release_in_both_spellings(tmp_path, web):
    site = web({DTK: atom("Release notes", [release("encounter/decomp-toolkit", "v1", 1)])})
    by_name = config(tmp_path, {"watch": ["encounter/decomp-toolkit"]})
    by_url = config(tmp_path, {"watch": [DTK]})
    [one] = feeds.fetch(context(by_name, site))
    [two] = feeds.fetch(context(by_url, site))
    assert (one.kind, one.title) == (two.kind, two.title) == ("release", "encounter/decomp-toolkit: v1")


def test_another_github_feed_is_named_owner_name_and_is_an_article(tmp_path, web):
    site = web({TAGS: atom("Tags from splat", [("tag:github.com,2008:Repository/1/v1", when(1), "https://github.com/ethteck/splat/releases/tag/v1", "v1")])})
    cfg = config(tmp_path, {"watch": [TAGS]})
    [item] = feeds.fetch(context(cfg, site))
    assert (item.kind, item.title) == ("article", "ethteck/splat: v1")


def test_another_watched_feed_keeps_its_own_title_and_is_an_article(tmp_path, web):
    site = web({BLOG: rss(days=1)})
    cfg = config(tmp_path, {"watch": [BLOG]})
    items = feeds.fetch(context(cfg, site))
    assert items and {i.kind for i in items} == {"article"}
    assert all(i.title.startswith("Example blog: ") and i.meta["label"] == "Example blog" for i in items)
    assert all(i.meta["watch_topic"] for i in items)  # still a watch item: the pipeline goes by watch_topic


def test_a_feed_url_entry_stays_an_article_and_keeps_its_title_even_on_github(tmp_path, web):
    site = web({COMMITS: atom("Recent Commits to m2c:master", [commit("abc", 1)])})
    cfg = config(tmp_path, {"feeds": {"urls": [COMMITS]}})
    [item] = feeds.fetch(context(cfg, site))
    assert item.kind == "article" and item.title.startswith("Recent Commits to m2c:master: ")
    assert "watch_topic" not in item.meta


@pytest.mark.parametrize(
    "url, label",
    [
        ("https://github.com/o/r/commits.atom", "o/r"),
        ("https://github.com/o/r/releases.atom", "o/r"),
        ("https://github.com/o/r/commits/feature/x.atom", "o/r"),
        ("https://github.com/o/r.git/tags.atom", "o/r.git"),
        ("https://github.com/o/r", None),  # no path after owner/name
        ("https://github.com/o/", None),
        ("https://github.com/o/na me/commits.atom", None),  # not a GITHUB_NAME
        ("https://gitlab.com/o/r/commits.atom", None),
        ("https://example.org/o/r/feed.xml", None),
    ],
)
def test_github_label(url, label):
    assert feeds.github_label(url) == label


@pytest.mark.parametrize(
    "url, kind",
    [
        ("https://github.com/o/r/releases.atom", "release"),
        ("https://github.com/o/r/commits.atom", "commit"),
        ("https://github.com/o/r/commits/main.atom", "commit"),
        ("https://github.com/o/r/commits/feature/x.atom", "commit"),
        ("https://github.com/o/r/tags.atom", "article"),
        ("https://github.com/o/r/commits", "article"),
        ("https://gitlab.com/o/r/commits.atom", "article"),
        ("https://example.org/blog/feed.xml", "article"),
    ],
)
def test_watch_kind(url, kind):
    assert feeds.watch_kind(url) == kind


# the fixed note (decision 25) -----------------------------------------------------------------------


def _watched(kind: str, label: str, entry: str) -> Item:
    return Item("feeds", kind, f"{label}: {entry}", "https://x.org/e", meta={"label": label})


def test_the_three_fixed_notes():
    assert watch_note(_watched("release", "o/r", "v1.2")) == (
        "New release v1.2 of o/r.", "You watch o/r under this topic.", "watch",
    )
    assert watch_note(_watched("commit", "o/r", "Fix the linker")) == (
        "New commit to o/r: Fix the linker.", "You watch o/r under this topic.", "watch",
    )
    assert watch_note(_watched("article", "Example blog", "A new post")) == (
        "New entry in Example blog: A new post.", "You watch Example blog under this topic.", "watch",
    )


@pytest.mark.parametrize("ending", [".", "!", "?"])
def test_a_title_that_ends_with_a_stop_gets_no_second_one(ending):
    assert watch_note(_watched("commit", "o/r", f"Fix it{ending}"))[0] == f"New commit to o/r: Fix it{ending}"
    assert watch_note(_watched("article", "Blog", f"Why{ending}"))[0] == f"New entry in Blog: Why{ending}"


def test_a_scan_gives_each_kind_its_own_note_and_label(tmp_path, web):
    site = web(
        {
            COMMITS: atom("Recent Commits to m2c:master", [commit("abc", 1, body="short")]),
            DTK: atom("Releases", [release("encounter/decomp-toolkit", "v1", 1, body="short")]),
            BLOG: rss(days=1),
        }
    )
    cfg = config(tmp_path, {"watch": [COMMITS, "encounter/decomp-toolkit", BLOG]})
    reports = {r.kind: r for r in run(cfg, site).new_reports if r.source == "feeds" and r.verdict == "watch"}
    assert reports["commit"].title == f"matt-kempster/m2c: {M2C_TITLE}"
    assert reports["commit"].what == f"New commit to matt-kempster/m2c: {M2C_TITLE}."
    assert reports["commit"].why == "You watch matt-kempster/m2c under this topic."
    assert reports["release"].what == "New release v1 of encounter/decomp-toolkit."
    assert reports["article"].what.startswith("New entry in Example blog: ")
    assert reports["article"].what.endswith(".") and not reports["article"].what.endswith("..")


def test_a_commit_with_a_long_message_goes_to_the_chat_model(tmp_path, web):
    site = web({COMMITS: atom("Recent Commits to m2c:master", [commit("abc", 1, body=LONG_TEXT)])})
    cfg = config(tmp_path, {"watch": [COMMITS]})
    ollama = FakeOllama()
    result = run(cfg, site, ollama)
    [report] = [r for r in result.new_reports if r.kind == "commit"]
    assert report.what == "A toolkit for running models locally." and report.verdict == "install"  # every verdict is allowed


# the earlier count by kind (decision 25) -----------------------------------------------------------


def report_of(kind: str, **meta) -> Report:
    return Report(date="2026-10-27", topic="t", score=0.0, source="feeds", kind=kind, title="o/r: x", url="https://x.org/x", meta=meta)


@pytest.mark.parametrize(
    "kind, one, many",
    [
        ("release", "release", "releases"),
        ("commit", "commit", "commits"),
        ("article", "entry", "entries"),
    ],
)
def test_the_meta_line_uses_the_noun_of_the_kind(kind, one, many):
    assert meta_line(report_of(kind, earlier_entries=1, earlier_since="2026-09-13")) == f"and 1 earlier {one} since 13 September"
    assert meta_line(report_of(kind, earlier_entries=3, earlier_since="2026-09-13")) == f"and 3 earlier {many} since 13 September"
    assert meta_line(report_of(kind, earlier_entries=2)) == f"and 2 earlier {many}"
    assert meta_line(report_of(kind)) == ""
    assert meta_line(report_of(kind, earlier_entries=0, earlier_since="2026-09-13")) == ""


def test_commit_has_a_label_and_a_scan_shows_the_count_in_the_note(tmp_path, web):
    assert KIND_LABEL["commit"] == "commit"
    site = web({COMMITS: atom("Recent Commits to m2c:master", [commit("c3", 1), commit("c2", 2), commit("c1", 3)])})
    cfg = config(tmp_path, {"watch": [COMMITS]})
    note = run(cfg, site).note_path.read_text(encoding="utf-8")
    assert "and 2 earlier commits since" in note


# entry ids (decision 26) ---------------------------------------------------------------------------


def _feed_with_guid(guid: str, title: str) -> str:
    moment = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)  # a day before the window's end
    return (
        '<rss version="2.0"><channel><title>Blog</title>'
        f"<item><title>{title}</title><link>https://example.org/{title.lower().replace(' ', '-')}</link>"
        f"<guid>{guid}</guid><pubDate>{format_datetime(moment)}</pubDate><description>x</description></item>"
        "</channel></rss>"
    )


def test_an_id_without_a_scheme_is_prefixed_with_its_feed_and_one_with_a_scheme_is_not():
    assert feeds.entry_key("https://a.org/feed", "1234") == "https://a.org/feed#1234"
    assert feeds.entry_key("https://a.org/feed", "tag:github.com,2008:Repository/1/v1") == "tag:github.com,2008:Repository/1/v1"
    assert feeds.entry_key("https://a.org/feed", "https://a.org/post/1") == "https://a.org/post/1"
    assert feeds.entry_key("https://a.org/feed", "urn:uuid:1225c695") == "urn:uuid:1225c695"
    assert feeds.entry_key("https://a.org/feed", "post:7") == "post:7"  # looks like a scheme, so it is kept


def test_the_same_short_guid_from_two_feeds_gives_two_keys(tmp_path, web):
    one, two = "https://example.org/a.xml", "https://example.org/b.xml"
    site = web({one: _feed_with_guid("1234", "First post"), two: _feed_with_guid("1234", "Second post")})
    cfg = config(tmp_path, {"watch": [one, two]})
    items = feeds.fetch(context(cfg, site))
    assert len(items) == 2
    assert items[0].meta["entry_id"] == f"{one}#1234" and items[1].meta["entry_id"] == f"{two}#1234"
    assert items[0].key != items[1].key
    assert items[0].key == f"feed:{one}#1234"


def test_both_entries_are_shown_when_two_feeds_share_a_guid(tmp_path, web):
    one, two = "https://example.org/a.xml", "https://example.org/b.xml"
    site = web({one: _feed_with_guid("1234", "First post"), two: _feed_with_guid("1234", "Second post")})
    cfg = config(tmp_path, {"watch": [one, two]})
    titles = sorted(r.title for r in run(cfg, site).new_reports if r.source == "feeds")
    assert titles == ["Blog: First post", "Blog: Second post"]


def test_github_and_url_ids_do_not_change(tmp_path, web):
    site = web({DTK: atom("Releases", [release("encounter/decomp-toolkit", "v1", 1)]), BLOG: rss(days=1)})
    cfg = config(tmp_path, {"watch": ["encounter/decomp-toolkit"], "feeds": {"urls": [BLOG]}})  # a watched feed gives one entry; urls gives both
    ids = {i.meta["entry_id"] for i in feeds.fetch(context(cfg, site))}
    assert "tag:github.com,2008:Repository/1/v1" in ids
    assert "https://example.org/blog/a" in ids  # an RSS guid that is a URL
    assert "https://example.org/blog/b" in ids  # no guid: the link is the id
    assert not [i for i in ids if "#" in i]


# follower counts outside selection (decision 29) -----------------------------------------------------


def test_label_band_lives_in_ranker_and_labelling_uses_it():
    assert ranker.LABEL_BAND == 0.08 and labelling.LABEL_BAND is ranker.LABEL_BAND


def _topic():
    data = yaml.safe_load(example_text())
    return parse_config(data, Path("unused")).topics[0]


def _entry(topic, name: str, outcome: str, *, score=0.5, bar=0.5, size=50, stars=50, source="github", owner=None) -> Scored:
    owner = name.split("/")[0] if owner is None else owner
    item = Item(source, "repo", name, f"https://github.com/{name}", meta={"owner": owner, "size_kb": size, "stars": stars})
    return Scored(item, topic, score, bar, outcome=outcome)


class _Users:
    """api.github.com/users/<login> with a count per login, and a log of the requests."""

    def __init__(self, counts: dict[str, int] | None = None, status: int = 200) -> None:
        self.counts = counts or {}
        self.status = status
        self.asked: list[str] = []

    def http(self) -> Http:
        def handler(request: httpx.Request) -> httpx.Response:
            login = request.url.path.rsplit("/", 1)[1]
            self.asked.append(login)
            if self.status != 200:
                return httpx.Response(self.status, text="down")
            return httpx.Response(200, json={"followers": self.counts.get(login, 0)})

        return Http(transport=httpx.MockTransport(handler), sleep=lambda seconds: None)


def _record() -> ScanRecord:
    return ScanRecord(now=TRUST_NOW, day="2026-10-27", embed_model="m")


def _ask(entries, users: _Users, record: ScanRecord | None = None) -> ScanRecord:
    record = record or _record()
    http = users.http()
    try:
        _ask_unchosen_owners(http, entries, record)
    finally:
        http.close()
    return record


@pytest.mark.parametrize("outcome", ["day_full", "topic_full", "below_bar"])
def test_a_thin_repository_that_was_not_chosen_gets_a_follower_count(outcome):
    topic = _topic()
    entries = [_entry(topic, "thin/repo", outcome, score=0.45, bar=0.5)]  # 0.05 below the bar: in the band
    users = _Users({"thin": 7})
    record = _ask(entries, users)
    assert users.asked == ["thin"] and record.followers == {"thin": 7}
    assert entries[0].outcome == outcome and not entries[0].held and not entries[0].selected  # nothing else changes


def test_nothing_is_asked_below_the_band_or_for_a_repository_that_is_not_thin():
    topic = _topic()
    entries = [
        _entry(topic, "far/repo", "below_bar", score=0.5 - ranker.LABEL_BAND - 0.001, bar=0.5),  # just under the band
        _entry(topic, "big/repo", "day_full", size=5000),  # not small
        _entry(topic, "nostars/repo", "day_full", stars=None),  # unknown stars: not thin
        _entry(topic, "nosize/repo", "day_full", size=None),  # unknown size is not thin
        _entry(topic, "shown/repo", "shown"),  # chosen: select already asked
        _entry(topic, "held/repo", "held"),  # held: select already asked
        _entry(topic, "lang/repo", "day_full", source="forgejo"),  # not GitHub
        _entry(topic, "anon/repo", "day_full", owner=""),  # no owner to ask about
    ]
    users = _Users()
    record = _ask(entries, users)
    assert users.asked == [] and record.followers == {}


def test_a_small_repository_with_few_stars_is_asked_too():
    topic = _topic()
    users = _Users()
    _ask([_entry(topic, "few/repo", "day_full", stars=3)], users)
    assert users.asked == ["few"]


def test_a_repository_exactly_at_the_band_edge_is_asked():
    topic = _topic()
    users = _Users()
    _ask([_entry(topic, "edge/repo", "below_bar", score=0.5 - ranker.LABEL_BAND, bar=0.5)], users)
    assert users.asked == ["edge"]


def test_the_band_follows_the_bar_of_the_items_own_topic():
    topic = _topic()
    low, high = _entry(topic, "low/repo", "below_bar", score=0.45, bar=0.5), _entry(topic, "high/repo", "below_bar", score=0.45, bar=0.6)
    users = _Users()
    _ask([low, high], users)
    assert users.asked == ["low"]  # 0.45 is 0.15 under a bar of 0.6


def test_at_most_ten_owners_are_asked_in_the_order_of_scored():
    topic = _topic()
    entries = [_entry(topic, f"o{n:02d}/repo", "day_full") for n in range(MAX_UNCHOSEN_OWNERS + 5)]
    users = _Users()
    record = _ask(entries, users)
    assert MAX_UNCHOSEN_OWNERS == 10
    assert users.asked == [f"o{n:02d}" for n in range(10)]
    assert len(record.followers) == 10


def test_two_repositories_of_one_owner_cost_one_request_and_one_place_of_the_ten():
    topic = _topic()
    entries = [_entry(topic, "same/one", "day_full"), _entry(topic, "same/two", "topic_full")]
    entries += [_entry(topic, f"o{n:02d}/repo", "day_full") for n in range(9)]
    users = _Users()
    record = _ask(entries, users)
    assert users.asked.count("same") == 1 and len(users.asked) == 10  # same + the first nine others
    assert len(record.followers) == 10


def test_an_owner_that_select_already_asked_about_is_not_asked_again():
    topic = _topic()
    record = _record()
    record.followers["thin"] = 1  # what `hold` leaves behind
    users = _Users({"thin": 9})
    _ask([_entry(topic, "Thin/repo", "day_full")], users, record)  # the owner's case does not matter
    assert users.asked == [] and record.followers == {"thin": 1}


def test_an_owner_that_select_asked_about_does_not_use_up_a_place_of_the_ten():
    topic = _topic()
    record = _record()
    record.followers.update({"early": 0})
    entries = [_entry(topic, "early/repo", "day_full")] + [_entry(topic, f"o{n:02d}/repo", "day_full") for n in range(10)]
    users = _Users()
    _ask(entries, users, record)
    assert users.asked == [f"o{n:02d}" for n in range(10)]


def test_a_failed_request_leaves_the_count_empty_and_is_not_repeated():
    topic = _topic()
    entries = [_entry(topic, "x/one", "day_full"), _entry(topic, "x/two", "day_full")]
    users = _Users(status=503)
    record = _ask(entries, users)
    assert set(users.asked) == {"x"} and record.followers == {"x": None}  # the second repository adds no request (Http itself retries a 503)


# ... in a whole scan

GOOD = {**GITHUB_SEARCH["items"][0], "size": 5000, "owner": {"login": "alice"}}
THIN_TEXTS = {
    "owna": "Run a local llm with embeddings and retrieval, no cloud needed",
    "ownd": "embeddings for cats",
}


def thin_world(**kwargs) -> World:
    items = [GOOD] + [_raw(f"{owner}/repo", 50, 50, owner, text) for owner, text in THIN_TEXTS.items()]
    return World(items, followers={"owna": 7, "ownd": 1}, **kwargs)


def scan_with(cfg, world, now=TRUST_NOW, **kwargs):
    http = world.http()
    try:
        return scan(cfg, http=http, ollama=FakeOllama(), now=now, **kwargs)
    finally:
        http.close()


def outcomes(result) -> dict[str, str]:
    return {e.item.title: e.outcome for e in result.explain}


def followers_of(cfg, key: str):
    return db_rows(cfg, "SELECT followers FROM observation WHERE key = ?", (key,))


def asked_users(world: World) -> list[str]:
    return [c for c in world.calls if c.startswith("/users/")]


def test_a_scan_asks_about_thin_repositories_the_day_cap_stopped(tmp_path, make_cfg):
    cfg = make_cfg(max_items=1, max_per_topic=1)
    world = thin_world()
    result = scan_with(cfg, world)
    got = outcomes(result)
    assert got["alice/local-llm-kit"] == "shown" and got["owna/repo"] == "day_full" and got["ownd/repo"] == "day_full"
    assert sorted(asked_users(world)) == ["/users/owna", "/users/ownd"]
    assert followers_of(cfg, "github:owna/repo") == [{"followers": 7}]
    assert followers_of(cfg, "github:ownd/repo") == [{"followers": 1}]
    assert db_rows(cfg, "SELECT outcome FROM decision d JOIN item i ON i.key = d.key WHERE i.title = 'owna/repo'") == [{"outcome": "day_full"}]
    assert not [w for w in result.warnings if "Held back" in w]  # not held: nothing new to read at the recheck
    assert db_rows(cfg, "SELECT state FROM item WHERE key = 'github:owna/repo'") == [{"state": "seen"}]


def test_a_scan_asks_about_thin_repositories_the_topic_cap_stopped(tmp_path, make_cfg):
    cfg = make_cfg(max_items=5, max_per_topic=1)
    world = thin_world()
    result = scan_with(cfg, world)
    assert outcomes(result)["owna/repo"] == "topic_full"
    assert followers_of(cfg, "github:owna/repo") == [{"followers": 7}]


def test_a_scan_asks_about_thin_repositories_just_under_the_bar(tmp_path):
    data = yaml.safe_load(fixture_text())
    data["settings"].update({"min_score": 0.2, "feedback_weight": 0.2})
    data["topics"][0]["min_score"] = 0.8  # alice clears it; owna (0.726) and ownd (0.761) are below it, inside the band
    cfg = parse_config(data, tmp_path / "home")
    world = thin_world()
    got = outcomes(scan_with(cfg, world))
    assert got["owna/repo"] == "below_bar" and got["ownd/repo"] == "below_bar"
    assert sorted(asked_users(world)) == ["/users/owna", "/users/ownd"]
    assert followers_of(cfg, "github:owna/repo") == [{"followers": 7}]
    assert followers_of(cfg, "github:ownd/repo") == [{"followers": 1}]


def test_a_scan_does_not_ask_about_thin_repositories_far_under_every_bar(make_cfg):
    cfg = make_cfg(min_score=0.95)  # every score is more than 0.08 under it
    world = thin_world()
    got = outcomes(scan_with(cfg, world))
    assert got["owna/repo"] == "below_bar" and got["ownd/repo"] == "below_bar"
    assert asked_users(world) == []
    assert followers_of(cfg, "github:owna/repo") == [{"followers": None}]


def test_trust_check_off_asks_nothing(tmp_path, make_cfg):
    cfg = make_cfg(max_items=1, max_per_topic=1, trust_check=False)
    world = thin_world()
    result = scan_with(cfg, world)
    assert outcomes(result)["owna/repo"] == "day_full"
    assert asked_users(world) == []
    assert followers_of(cfg, "github:owna/repo") == [{"followers": None}]


def test_a_preview_asks_nothing(tmp_path, make_cfg):
    cfg = make_cfg(max_items=1, max_per_topic=1)
    world = thin_world()
    http = world.http()
    try:
        result = scan(cfg, http=http, ollama=FakeOllama(up=False), now=TRUST_NOW)
    finally:
        http.close()
    assert result.preview
    assert asked_users(world) == []


def test_an_owner_that_hold_asked_about_is_asked_once_in_a_scan(make_cfg):
    cfg = make_cfg(max_items=1, max_per_topic=1)
    first = _raw("owna/first", 50, 50, "owna", "A local llm kit with embeddings")  # ranks first, so selection holds it
    second = _raw("owna/second", 50, 50, "owna", THIN_TEXTS["ownd"])  # stopped by the day cap
    world = World([GOOD, first, second], followers={"owna": 1})  # 2 followers or fewer: the check holds it
    got = outcomes(scan_with(cfg, world))
    assert got["owna/first"] == "held" and got["owna/second"] == "day_full"
    assert asked_users(world) == ["/users/owna"]  # by `hold`; the new step finds the count and does not ask again
    assert followers_of(cfg, "github:owna/first") == [{"followers": 1}]
    assert followers_of(cfg, "github:owna/second") == [{"followers": 1}]


# huggingface-models is off by default (decision 30) -----------------------------------------------------


def _parsed(tmp_path, sources=None, drop=False):
    data = yaml.safe_load(example_text())
    if drop:
        data["sources"].pop("huggingface-models")
    if sources is not None:
        data["sources"] = {**data["sources"], **sources}
    return parse_config(data, tmp_path / "home")


def test_the_default_for_huggingface_models_is_off():
    assert DEFAULT_SOURCES["huggingface-models"]["enabled"] is False
    assert DEFAULT_SOURCES["huggingface-models"]["limit"] == 25
    for name in ("github", "arxiv", "huggingface-papers", "feeds", "forgejo"):
        assert DEFAULT_SOURCES[name]["enabled"] is True


def test_a_file_without_an_entry_for_it_leaves_it_off(tmp_path):
    cfg = _parsed(tmp_path, drop=True)
    assert not cfg.source_enabled("huggingface-models") and cfg.sources["huggingface-models"]["limit"] == 25
    assert cfg.source_enabled("huggingface-papers")


def test_the_example_file_has_it_off(tmp_path):
    cfg = _parsed(tmp_path)
    assert not cfg.source_enabled("huggingface-models")
    assert "trending models, matched on tags only; turn on if your topics are about models" in example_text()


def test_a_file_that_turns_it_on_keeps_it_on(tmp_path):
    assert _parsed(tmp_path, {"huggingface-models": {"enabled": True, "limit": 10}}).source_enabled("huggingface-models")
    assert _parsed(tmp_path, {"huggingface-models": True}, drop=True).source_enabled("huggingface-models")
    cfg = _parsed(tmp_path, {"huggingface-models": {"enabled": True}}, drop=True)
    assert cfg.source_enabled("huggingface-models") and cfg.sources["huggingface-models"]["limit"] == 25


def test_a_scan_with_the_default_does_not_read_the_models_list(tmp_path):
    calls: list[str] = []
    data = yaml.safe_load(fixture_text())
    data["sources"].pop("huggingface-models")
    cfg = parse_config(data, tmp_path / "home")
    http = Http(transport=httpx.MockTransport(make_handler(calls)), sleep=lambda seconds: None)
    try:
        scan(cfg, http=http, ollama=FakeOllama(), now=TRUST_NOW)
    finally:
        http.close()
    assert "huggingface.co/api/models" not in calls and "huggingface.co/api/daily_papers" in calls
