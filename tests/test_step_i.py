"""Step I: the feeds source, the watch list, its selection rules and the doctor line."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import yaml

from conftest import FakeOllama, db_rows, make_handler, stored_item
from research_radar import cli
from research_radar.config import ConfigError, example_text, parse_config
from research_radar.digest import KIND_LABEL
from research_radar.http import Http, SourceError
from research_radar.llm import OllamaError
from research_radar.models import Item, Report
from research_radar.notes import meta_line
from research_radar.pipeline import scan, topic_fingerprint
from research_radar.sources import feeds
from research_radar.sources.base import FetchContext
from research_radar.store import Store

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
GHIDRA = "https://github.com/NationalSecurityAgency/ghidra/releases.atom"
SPLAT = "https://github.com/ethteck/splat/releases.atom"
DTK = "https://github.com/encounter/decomp-toolkit/releases.atom"
XBE = "https://github.com/XboxDev/ghidra-xbe/releases.atom"
M2C = "https://github.com/matt-kempster/m2c/commits.atom"
BLOG = "https://example.org/blog/feed.xml"

LONG_TEXT = "Adds a new decompiler pass for overlays and fixes the linker script generator. " * 4  # over 200 characters


def when(days: float = 0, hours: float = 0) -> str:
    return (NOW - timedelta(days=days, hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def atom(title: str, entries: list[tuple[str, str, str, str]]) -> str:
    """A GitHub style Atom feed. Entry: (id, updated, link, title); the text goes in the title's
    sibling `content` when the id carries it after a '|'."""
    rows = ""
    for entry_id, updated, link, name in entries:
        entry_id, _, body = entry_id.partition("|")
        content = f'<content type="html">{body}</content>' if body else ""
        date = f"<updated>{updated}</updated>" if updated else ""
        rows += (
            f"<entry><id>{entry_id}</id>{date}"
            f'<link rel="alternate" type="text/html" href="{link}"/><title>{name}</title>{content}</entry>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<feed xmlns="http://www.w3.org/2005/Atom"><id>tag:github.com,2008:x</id><title>{title}</title>{rows}</feed>'
    )


def release(repo: str, tag: str, days: float = 0, body: str = "", hours: float = 0) -> tuple[str, str, str, str]:
    return (
        f"tag:github.com,2008:Repository/1/{tag}|{body}",
        when(days, hours),
        f"https://github.com/{repo}/releases/tag/{tag}",
        tag,
    )


RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Example blog</title>
<item><title>Local &amp; fast retrieval</title><link>https://example.org/blog/a</link>
  <guid>https://example.org/blog/a</guid><pubDate>{recent}</pubDate>
  <description>&lt;p&gt;Run a &lt;b&gt;local llm&lt;/b&gt; with embeddings and retrieval, no cloud needed &amp;amp; fast&lt;/p&gt;</description></item>
<item><title>Without a guid</title><link>/blog/b</link><pubDate>{recent}</pubDate><description>x</description></item>
<item><title>No date at all</title><link>https://example.org/blog/c</link><guid>c</guid></item>
<item><title>Too old</title><link>https://example.org/blog/d</link><guid>d</guid><pubDate>Mon, 01 Jan 2024 10:00:00 +0000</pubDate></item>
</channel></rss>
"""


def rss(days: float = 1) -> str:
    stamp = (NOW - timedelta(days=days)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    return RSS.replace("{recent}", stamp)


class Web:
    """Feeds behind a mock transport; every other host answers as in conftest."""

    def __init__(self, pages: dict[str, str | int] | None = None) -> None:
        self.pages = pages or {}
        self.calls: list[str] = []
        self.rest = make_handler()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in self.pages or url.split("?")[0] in self.pages:
            self.calls.append(url)
            page = self.pages.get(url, self.pages.get(url.split("?")[0]))
            if isinstance(page, int):
                return httpx.Response(page, text="nope")
            return httpx.Response(200, content=page.encode("utf-8"), headers={"content-type": "application/xml"})
        return self.rest(request)

    def http(self) -> Http:
        return Http(transport=httpx.MockTransport(self), sleep=lambda seconds: None)


@pytest.fixture
def web():
    made: list[Http] = []

    def build(pages=None) -> Web:
        site = Web(pages)
        site.client = site.http()
        made.append(site.client)
        return site

    yield build
    for client in made:
        client.close()


def config(tmp_path, *patches, **settings):
    data = yaml.safe_load(example_text())
    data["settings"].update({"min_score": 0.2, "feedback_weight": 0.2})
    data["settings"].update(settings)
    for patch, topic in zip(patches, data["topics"]):
        topic.update(patch)
    return parse_config(data, tmp_path / "home")


def run(cfg, site, ollama=None, now=NOW, **kwargs):
    return scan(cfg, http=site.client, ollama=ollama or FakeOllama(), now=now, **kwargs)


def context(cfg, site, since=NOW - timedelta(days=3), until=NOW) -> FetchContext:
    return FetchContext(config=cfg, http=site.client, since=since, until=until)


# configuration ----------------------------------------------------------------------------------


def test_feeds_is_a_source_and_watch_is_a_topic_key(tmp_path):
    cfg = config(tmp_path, {"watch": ["ethteck/splat", " https://example.org/f.xml "], "feeds": {"urls": [BLOG]}})
    assert cfg.sources["feeds"] == {"enabled": True}
    assert cfg.topics[0].watch == ["ethteck/splat", "https://example.org/f.xml"]
    assert cfg.topics[0].source_option("feeds", "urls") == [BLOG]
    assert cfg.topics[1].watch == []
    assert not [n for n in cfg.notices if "watch" in n or "feeds" in n]  # both are known keys


@pytest.mark.parametrize("entry", ["not a repository", "owner", "a/b/c", "ftp://example.org/feed", "https://", "owner/na me"])
def test_a_bad_watch_entry_raises_and_names_it(tmp_path, entry):
    with pytest.raises(ConfigError, match="topic 'Local-first AI tools': watch entry .* is neither"):
        config(tmp_path, {"watch": [entry]})


def test_watch_must_be_a_list_and_feed_urls_must_be_urls(tmp_path):
    with pytest.raises(ConfigError, match="watch must be a list of strings"):
        config(tmp_path, {"watch": {"a": "b/c"}})
    with pytest.raises(ConfigError, match="feeds.urls entry 'example.org/feed' is not an http"):
        config(tmp_path, {"feeds": {"urls": ["example.org/feed"]}})


def test_owner_name_means_the_releases_feed():
    assert feeds.feed_url("NationalSecurityAgency/ghidra") == GHIDRA
    assert feeds.feed_url("https://github.com/N64Recomp/N64Recomp/commits.atom") == (
        "https://github.com/N64Recomp/N64Recomp/commits.atom"
    )


def test_the_fingerprint_changes_only_for_a_topic_with_a_watch_list(tmp_path):
    plain = config(tmp_path).topics[0]
    old = [plain.description.strip(), plain.keywords, plain.exclude, plain.options]  # the payload before step I
    expected = hashlib.sha1(json.dumps(old, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
    assert topic_fingerprint(plain) == expected
    watching = config(tmp_path, {"watch": ["ethteck/splat"]}).topics[0]
    assert topic_fingerprint(watching) != expected
    assert topic_fingerprint(config(tmp_path, {"watch": []}).topics[0]) == expected


# reading ----------------------------------------------------------------------------------------


def test_an_atom_releases_feed_becomes_one_release_item(tmp_path, web):
    body = "&lt;p&gt;Adds &lt;b&gt;ELF&lt;/b&gt; output &amp;amp; fixes.&lt;/p&gt;" + "x" * 3000
    site = web({DTK: atom("Release notes from decomp-toolkit", [release("encounter/decomp-toolkit", "v1.2.0", 1, body)])})
    cfg = config(tmp_path, {"watch": ["encounter/decomp-toolkit"]})
    [item] = feeds.fetch(context(cfg, site))
    assert (item.source, item.kind) == ("feeds", "release")
    assert item.title == "encounter/decomp-toolkit: v1.2.0"  # owner/name is the label, not the feed's title
    assert item.url == "https://github.com/encounter/decomp-toolkit/releases/tag/v1.2.0"
    assert item.text.startswith("Adds ELF output & fixes. xxx") and len(item.text) == 2000
    assert item.published == "2026-10-26T12:00:00+00:00"
    assert item.meta["feed"] == DTK and item.meta["watch_topic"] == "Local-first AI tools"
    assert item.meta["entry_id"] == "tag:github.com,2008:Repository/1/v1.2.0"
    assert "earlier_entries" not in item.meta


def test_an_rss_feed_is_read_and_a_feed_url_is_labelled_with_its_title(tmp_path, web):
    site = web({BLOG: rss(days=1)})
    cfg = config(tmp_path, {"feeds": {"urls": [BLOG]}})
    items = feeds.fetch(context(cfg, site))
    by_title = {i.title: i for i in items}
    assert sorted(by_title) == ["Example blog: Local & fast retrieval", "Example blog: Without a guid"]  # no date, too old: skipped
    first = by_title["Example blog: Local & fast retrieval"]
    assert first.kind == "article" and "watch_topic" not in first.meta
    assert first.text == "Run a local llm with embeddings and retrieval, no cloud needed & fast"
    assert first.meta["entry_id"] == "https://example.org/blog/a"
    second = by_title["Example blog: Without a guid"]
    assert second.url == "https://example.org/blog/b" and second.meta["entry_id"] == "https://example.org/blog/b"  # guid else link


def test_the_window_decides_and_an_entry_without_a_date_is_skipped(tmp_path, web):
    entries = [
        release("o/r", "inside", days=1),
        release("o/r", "before", days=5),
        release("o/r", "after", days=-1),
        ("tag:undated", "", "https://github.com/o/r/releases/tag/undated", "undated"),
    ]
    site = web({"https://github.com/o/r/releases.atom": atom("Releases", entries)})
    cfg = config(tmp_path, {"watch": ["o/r"]})
    [item] = feeds.fetch(context(cfg, site))
    assert item.title == "o/r: inside" and "earlier_entries" not in item.meta
    assert feeds.fetch(context(cfg, site, since=NOW + timedelta(days=2), until=NOW + timedelta(days=3))) == []


def test_a_watched_feed_gives_its_newest_entry_with_the_earlier_count(tmp_path, web):
    entries = [release("encounter/decomp-toolkit", f"v1.{n}", hours=n) for n in (0, 3, 7)] + [
        release("encounter/decomp-toolkit", "v0.9", days=9)  # outside the window
    ]
    site = web({DTK: atom("Releases", entries)})
    cfg = config(tmp_path, {"watch": ["encounter/decomp-toolkit"]})
    [item] = feeds.fetch(context(cfg, site))
    assert item.title.endswith(": v1.0")
    assert item.meta["earlier_entries"] == 2
    assert item.meta["earlier_since"] == (NOW - timedelta(hours=7)).date().isoformat()


def test_a_feed_url_gives_at_most_thirty_entries(tmp_path, web):
    rows = "".join(
        f"<item><title>t{n}</title><link>https://example.org/{n}</link><guid>g{n}</guid>"
        f"<pubDate>{(NOW - timedelta(hours=n + 1)).strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate></item>"
        for n in range(45)
    )
    site = web({BLOG: f'<rss version="2.0"><channel><title>B</title>{rows}</channel></rss>'})
    items = feeds.fetch(context(config(tmp_path, {"feeds": {"urls": [BLOG]}}), site))
    assert len(items) == 30 and items[0].title == "B: t0" and items[-1].title == "B: t29"


def test_watch_entries_are_read_before_feed_urls_and_a_feed_is_read_once(tmp_path, web):
    page = atom("Releases", [release("o/r", "v1", days=1)])
    url = "https://github.com/o/r/releases.atom"
    site = web({url: page, BLOG: rss()})
    cfg = config(tmp_path, {"feeds": {"urls": [BLOG, url]}}, {"watch": ["o/r"]})
    items = feeds.fetch(context(cfg, site))
    assert site.calls.count(url) == 1
    assert [i.kind for i in items][0] == "release"  # in both lists: a watch item, read first
    assert items[0].meta["watch_topic"] == "Geospatial machine learning"
    assert [i.kind for i in items].count("release") == 1


def test_two_topics_watching_one_feed_give_it_to_the_first(tmp_path, web):
    site = web({SPLAT: atom("Releases", [release("ethteck/splat", "v1", days=1)])})
    cfg = config(tmp_path, {"watch": ["ethteck/splat"]}, {"watch": ["ethteck/splat"]})
    [item] = feeds.fetch(context(cfg, site))
    assert item.meta["watch_topic"] == "Local-first AI tools"


def test_requests_are_spaced_per_host(tmp_path, web):
    spaced: list[tuple[str, float]] = []
    site = web({SPLAT: atom("R", []), DTK: atom("R", []), BLOG: rss()})
    original = site.client.get

    def spy(url, **kwargs):
        spaced.append((kwargs.get("bucket"), kwargs.get("min_interval")))
        return original(url, **kwargs)

    site.client.get = spy
    feeds.fetch(context(config(tmp_path, {"watch": ["ethteck/splat", "encounter/decomp-toolkit"], "feeds": {"urls": [BLOG]}}), site))
    assert spaced == [("feeds:github.com", 1.0), ("feeds:github.com", 1.0), ("feeds:example.org", 1.0)]


def test_a_failed_feed_is_a_warning_and_the_others_are_read(tmp_path, web):
    site = web({SPLAT: 503, DTK: atom("R", [release("encounter/decomp-toolkit", "v1", days=1)]), XBE: "<html>not a feed</html>"})  # well formed XML, but not a feed
    cfg = config(tmp_path, {"watch": ["ethteck/splat", "encounter/decomp-toolkit", "XboxDev/ghidra-xbe"]})
    ctx = context(cfg, site)
    [item] = feeds.fetch(ctx)
    assert item.title == "encounter/decomp-toolkit: v1"
    assert len(ctx.warnings) == 2
    assert ctx.warnings[0].startswith("feeds: ethteck/splat: github.com: HTTP 503")
    assert ctx.warnings[1].startswith("feeds: XboxDev/ghidra-xbe: not an Atom or RSS feed")


def test_a_body_over_two_megabytes_or_with_entities_is_skipped_with_a_warning(tmp_path, web):
    big = atom("Releases", [release("o/r", "v1", days=1)]).replace("</feed>", "<!-- " + "x" * 2_100_000 + " --></feed>")
    bomb = '<?xml version="1.0"?><!DOCTYPE f [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]><feed xmlns="http://www.w3.org/2005/Atom">&b;</feed>'
    site = web({"https://github.com/o/r/releases.atom": big, SPLAT: bomb})
    cfg = config(tmp_path, {"watch": ["o/r", "ethteck/splat"]})
    ctx = context(cfg, site)
    assert feeds.fetch(ctx) == []
    assert "larger than 2 MB" in ctx.warnings[0] and "declares XML entities" in ctx.warnings[1]


def test_a_feed_with_nothing_configured_reads_nothing(tmp_path, web):
    site = web()
    assert feeds.fetch(context(config(tmp_path), site)) == []
    assert site.calls == []


# keys -------------------------------------------------------------------------------------------


def test_a_release_key_differs_from_its_repository_and_from_other_releases():
    def item(tag):
        return Item(
            "feeds", "release", f"o/r: {tag}", f"https://github.com/o/r/releases/tag/{tag}",
            meta={"entry_id": f"tag:github.com,2008:Repository/1/{tag}"},
        )

    one, two = item("v1"), item("v2")
    assert one.key == "feed:tag:github.com,2008:Repository/1/v1"
    assert len({one.key, two.key, "github:o/r"}) == 3
    # without an id of its own an item falls back to its URL, as any other source does
    assert Item("feeds", "article", "t", "https://example.org/a").key == "url:https://example.org/a"


# selection --------------------------------------------------------------------------------------


def test_watch_items_are_shown_under_their_topic_outside_both_caps(tmp_path, web):
    pages = {
        GHIDRA: atom("Releases", [release("NationalSecurityAgency/ghidra", "v12", days=1, body=LONG_TEXT)]),
        SPLAT: atom("Releases", [release("ethteck/splat", "v0.9", days=1)]),
        DTK: atom("Releases", [release("encounter/decomp-toolkit", "v1.2", days=1)]),
        BLOG: rss(),
    }
    site = web(pages)
    cfg = config(
        tmp_path,
        {"watch": ["NationalSecurityAgency/ghidra", "ethteck/splat", "encounter/decomp-toolkit"], "feeds": {"urls": [BLOG]}},
        max_items=1,
        max_per_topic=1,
    )
    ollama = FakeOllama()
    result = run(cfg, site, ollama)
    reports = result.new_reports
    watch = [r for r in reports if r.kind == "release"]
    ranked = [r for r in reports if r.kind != "release"]
    assert [r.title for r in reports[:3]] == [r.title for r in watch]  # before the other items
    assert sorted(r.title for r in watch) == [
        "NationalSecurityAgency/ghidra: v12",
        "encounter/decomp-toolkit: v1.2",
        "ethteck/splat: v0.9",
    ]
    assert {r.topic for r in watch} == {"Local-first AI tools"}
    assert all(r.score == 0.0 and r.source == "feeds" for r in watch)
    assert len(ranked) == 1  # max_items 1 holds for the ranked items alone
    rows = db_rows(cfg, "SELECT d.outcome, d.score, d.topic FROM decision d JOIN item i ON i.key = d.key WHERE i.kind = 'release'")
    assert rows == [{"outcome": "watch", "score": 0.0, "topic": "Local-first AI tools"}] * 3
    assert {stored_item(cfg, f"feed:tag:github.com,2008:Repository/1/{t}")["state"] for t in ("v12", "v0.9", "v1.2")} == {"shown"}
    assert db_rows(cfg, "SELECT 1 FROM score s JOIN item i ON i.key = s.key WHERE i.kind = 'release'") == []  # never scored


def test_a_muted_topic_still_shows_its_watch_items(tmp_path, web):
    site = web({SPLAT: atom("Releases", [release("ethteck/splat", "v1", days=1, body=LONG_TEXT)])})
    cfg = config(tmp_path, {"watch": ["ethteck/splat"], "max_per_topic": 0, "min_score": 0.99})
    assert [r.title for r in run(cfg, site).new_reports if r.kind == "release"] == ["ethteck/splat: v1"]


def test_article_items_are_ranked_and_the_unrelated_one_stays_out(tmp_path, web):
    rows = (
        "<item><title>Local llm kit</title><link>https://example.org/a</link><guid>a</guid><pubDate>{d}</pubDate>"
        "<description>Run a local llm with embeddings and retrieval, no cloud needed</description></item>"
        "<item><title>Cats</title><link>https://example.org/b</link><guid>b</guid><pubDate>{d}</pubDate>"
        "<description>Pictures of cats on a sofa</description></item>"
    ).replace("{d}", (NOW - timedelta(days=1)).strftime("%a, %d %b %Y %H:%M:%S +0000"))
    site = web({BLOG: f'<rss version="2.0"><channel><title>B</title>{rows}</channel></rss>'})
    cfg = config(tmp_path, {"feeds": {"urls": [BLOG]}}, min_score=0.3)
    result = run(cfg, site)
    articles = [r for r in result.new_reports if r.kind == "article"]
    assert [r.title for r in articles] == ["B: Local llm kit"]
    assert articles[0].score > 0 and articles[0].topic == "Local-first AI tools"
    outcomes = {r["title"]: r["outcome"] for r in db_rows(cfg, "SELECT i.title, d.outcome FROM decision d JOIN item i ON i.key = d.key WHERE i.kind = 'article'")}
    assert outcomes == {"B: Local llm kit": "shown", "B: Cats": "below_bar"}


def test_short_release_text_gets_the_fixed_note_and_longer_text_a_model(tmp_path, web):
    site = web(
        {
            XBE: atom("Releases", [release("XboxDev/ghidra-xbe", "build-77", days=1, body="build 77")]),  # 8 characters
            GHIDRA: atom("Releases", [release("NationalSecurityAgency/ghidra", "v12", days=1, body=LONG_TEXT)]),
        }
    )
    cfg = config(tmp_path, {"watch": ["XboxDev/ghidra-xbe", "NationalSecurityAgency/ghidra"]})
    ollama = FakeOllama()
    result = run(cfg, site, ollama)
    notes = {r.title: r for r in result.new_reports if r.kind == "release"}
    short = notes["XboxDev/ghidra-xbe: build-77"]
    assert short.what == "New release build-77 of XboxDev/ghidra-xbe."
    assert short.why == "You watch XboxDev/ghidra-xbe under this topic."
    assert short.verdict == "watch"
    long = notes["NationalSecurityAgency/ghidra: v12"]
    assert long.what == "A toolkit for running models locally." and long.verdict == "install"  # a release allows every verdict
    assert ollama.chat_calls == 1 + len([r for r in result.new_reports if r.kind != "release"])


def test_a_later_release_is_shown_and_an_earlier_one_is_not_shown_again(tmp_path, web):
    page = atom("Releases", [release("ethteck/splat", "v1", days=1)])
    site = web({SPLAT: page})
    cfg = config(tmp_path, {"watch": ["ethteck/splat"]})
    first = run(cfg, site)
    assert [r.title for r in first.new_reports if r.kind == "release"] == ["ethteck/splat: v1"]
    later = NOW + timedelta(days=1)
    site.pages[SPLAT] = atom(
        "Releases",
        [release("ethteck/splat", "v2", days=-1, hours=2), release("ethteck/splat", "v1", days=1)],  # v1 is old news
    )
    second = run(cfg, site, now=later)
    assert [r.title for r in second.new_reports if r.kind == "release"] == ["ethteck/splat: v2"]
    third = run(cfg, site, now=later + timedelta(hours=1))
    assert [r for r in third.new_reports if r.kind == "release"] == []
    assert stored_item(cfg, "github:ethteck/splat") is None  # the repository's own key was never used


def test_a_failed_feed_holds_the_window_of_the_source(tmp_path, web):
    site = web({SPLAT: 503, DTK: atom("R", [release("encounter/decomp-toolkit", "v1", days=1, body=LONG_TEXT)])})
    cfg = config(tmp_path, {"watch": ["ethteck/splat", "encounter/decomp-toolkit"]})
    result = run(cfg, site)
    assert any(w.startswith("feeds: ethteck/splat") for w in result.warnings)
    assert [r.title for r in result.new_reports if r.kind == "release"] == ["encounter/decomp-toolkit: v1"]
    last_runs = Store(cfg.state_dir).last_runs()
    assert "feeds" not in last_runs and "github" in last_runs
    # the next scan reads the same window again and the feed that worked is not shown twice
    site.pages[SPLAT] = atom("R", [release("ethteck/splat", "v3", days=0, hours=5)])
    again = run(cfg, site, now=NOW + timedelta(hours=1))
    assert [r.title for r in again.new_reports if r.kind == "release"] == ["ethteck/splat: v3"]
    assert "feeds" in Store(cfg.state_dir).last_runs()


def test_a_topic_that_gains_a_watch_entry_looks_back_three_days_for_it(tmp_path, web):
    site = web()
    cfg = config(tmp_path)
    run(cfg, site)  # the topics are recorded without a watch list
    watching = config(tmp_path, {"watch": ["ethteck/splat"]})
    site.pages[SPLAT] = atom(
        "Releases", [release("ethteck/splat", "v2", days=2), release("ethteck/splat", "v1", days=5)]
    )
    result = run(watching, site, now=NOW + timedelta(hours=1))
    [report] = [r for r in result.new_reports if r.kind == "release"]
    assert report.title == "ethteck/splat: v2"  # the release of 5 days ago is older than first_run_days
    assert "earlier_entries" not in report.meta
    assert "Topic" not in result.warnings  # no failure
    assert result.backfilled == ["Local-first AI tools"]


def test_the_look_back_and_the_window_of_one_feed_make_one_item_with_a_count(tmp_path, web):
    site = web()
    run(config(tmp_path), site)
    watching = config(tmp_path, {"watch": ["ethteck/splat"]})
    site.pages[SPLAT] = atom(
        "Releases",
        [
            release("ethteck/splat", "v3", days=0, hours=-0.5),  # after the last scan: in the window
            release("ethteck/splat", "v2", days=2),  # in the look back
            release("ethteck/splat", "v1", days=2, hours=3),  # in the look back
        ],
    )
    result = run(watching, site, now=NOW + timedelta(hours=1))
    [report] = [r for r in result.new_reports if r.kind == "release"]
    assert report.title == "ethteck/splat: v3"
    assert report.meta["earlier_entries"] == 2
    assert report.meta["earlier_since"] == (NOW - timedelta(days=2, hours=3)).date().isoformat()


# what the reader sees ---------------------------------------------------------------------------


def report(**meta) -> Report:
    return Report(date="2026-10-27", topic="t", score=0.0, source="feeds", kind="release", title="o/r: v1", url="https://x.org/v1", meta=meta)


def test_the_meta_line_of_a_release_counts_the_earlier_entries():
    assert meta_line(report()) == ""
    assert meta_line(report(earlier_entries=0, earlier_since="2026-10-02")) == ""
    assert meta_line(report(earlier_entries=3, earlier_since="2026-10-02")) == "and 3 earlier entries since 2 October"
    assert meta_line(report(earlier_entries=1, earlier_since="2026-10-02")) == "and 1 earlier entry since 2 October"
    assert meta_line(report(earlier_entries=2)) == "and 2 earlier entries"


def test_release_and_article_have_a_label():
    assert KIND_LABEL["release"] == "release" and KIND_LABEL["article"] == "article"


def test_a_scan_with_a_watch_item_renders_in_the_note_and_the_digest(tmp_path, web):
    site = web({DTK: atom("R", [release("encounter/decomp-toolkit", "v1", days=1, body="short"), release("encounter/decomp-toolkit", "v0", days=2)])})
    cfg = config(tmp_path, {"watch": ["encounter/decomp-toolkit"]})
    result = run(cfg, site)
    note = result.note_path.read_text(encoding="utf-8")
    assert "**encounter/decomp-toolkit: v1**" in note and "(and 1 earlier entry since 2026-10-25" not in note
    assert "and 1 earlier entry since 25 October" in note
    assert "release · feeds" in result.digest_path.read_text(encoding="utf-8")


# radar doctor -----------------------------------------------------------------------------------


class _NoOllama:
    def __init__(self, *args) -> None: ...

    def models(self):
        raise OllamaError("down")

    def close(self) -> None: ...


def test_doctor_reads_each_feed_once_and_prints_count_and_newest_date(tmp_path, web, monkeypatch, capsys):
    site = web(
        {
            DTK: atom("R", [release("encounter/decomp-toolkit", "v1", days=1), release("encounter/decomp-toolkit", "v0", days=30)]),
            SPLAT: atom("R", []),
            M2C: atom("Recent Commits", []),
            XBE: 503,
            BLOG: rss(),
        }
    )
    cfg = config(tmp_path, {"watch": ["encounter/decomp-toolkit", "ethteck/splat", M2C, "XboxDev/ghidra-xbe"], "feeds": {"urls": [BLOG, DTK]}})
    for name in ("github", "arxiv", "huggingface-papers", "huggingface-models", "forgejo"):
        cfg.sources[name]["enabled"] = False
    monkeypatch.setattr(cli, "Ollama", _NoOllama)
    monkeypatch.setattr(cli, "Http", lambda **kw: site.client)
    monkeypatch.setattr(site.client, "close", lambda: None)
    assert cli.cmd_doctor(cfg) == 1  # one feed answered 503
    out = capsys.readouterr().out
    assert "ok    feeds (encounter/decomp-toolkit): 2 entries, newest 2026-10-26" in out
    assert "ok    feeds (ethteck/splat): no entries. A repository without releases" in out
    assert "https://github.com/ethteck/splat/commits.atom instead" in out
    assert f"ok    feeds ({M2C}): no entries\n" in out  # a commits feed: no hint
    assert "FAIL  feeds (XboxDev/ghidra-xbe): github.com: HTTP 503" in out
    assert f"ok    feeds ({BLOG}): 4 entries, newest" in out  # the doctor counts every entry, in the window or not
    assert site.calls.count(DTK) == 1  # listed twice, read once
    assert any("Error" in o or "FAIL  ollama" in o for o in out.splitlines())


def test_doctor_prints_nothing_about_feeds_when_the_source_is_off(tmp_path, web, monkeypatch, capsys):
    site = web({DTK: atom("R", [])})
    cfg = config(tmp_path, {"watch": ["encounter/decomp-toolkit"]})
    for name in ("github", "arxiv", "huggingface-papers", "huggingface-models", "forgejo", "feeds"):
        cfg.sources[name]["enabled"] = False
    monkeypatch.setattr(cli, "Ollama", _NoOllama)
    monkeypatch.setattr(cli, "Http", lambda **kw: site.client)
    monkeypatch.setattr(site.client, "close", lambda: None)
    cli.cmd_doctor(cfg)
    assert "feeds" not in capsys.readouterr().out and site.calls == []


def test_parse_feed_rejects_other_documents():
    with pytest.raises(SourceError, match="not an Atom or RSS feed"):
        feeds.parse_feed("<html><body/></html>")
    with pytest.raises(SourceError, match="not a readable feed"):
        feeds.parse_feed("not xml at all")
