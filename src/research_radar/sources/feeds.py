"""Atom and RSS feeds: the owner's watch list and articles from blogs.

A topic's `watch` entries are shown when something new appears, without a bar. Its `feeds.urls`
entries give articles that are scored like any other item. `owner/name` in a watch list means the
releases feed of that GitHub repository.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit

from ..config import GITHUB_NAME, Config
from ..http import Http, SourceError
from ..models import Item
from .base import FetchContext

NAME = "feeds"
MAX_BYTES = 2_000_000  # a larger body is skipped
MAX_TEXT = 2000
MAX_ARTICLES = 30  # entries per `feeds.urls` feed and scan

_ATOM = "{http://www.w3.org/2005/Atom}"
_SCRIPTS = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]*>")
_GITHUB_FEED_PATH = re.compile(r"^/(?P<owner>[^/]+)/(?P<name>[^/]+)/.+$")
_RELEASES_PATH = re.compile(r"^/[^/]+/[^/]+/releases\.atom$")
_COMMITS_PATH = re.compile(r"^/[^/]+/[^/]+/commits(\.atom|/.+\.atom)$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def is_feed_url(value: str) -> bool:
    parts = urlsplit(value)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def is_github_name(value: str) -> bool:
    return bool(GITHUB_NAME.match(value))


def feed_url(entry: str) -> str:
    """The URL a `watch` entry stands for: owner/name is the releases feed of that repository."""
    entry = entry.strip()
    return f"https://github.com/{entry}/releases.atom" if is_github_name(entry) else entry


def is_github_releases(url: str) -> bool:
    parts = urlsplit(url)
    return parts.netloc.lower() == "github.com" and parts.path.endswith("/releases.atom")


def github_label(url: str) -> str | None:
    """`owner/name` for a feed on github.com whose path starts with `/owner/name/`, else None."""
    parts = urlsplit(url)
    if parts.netloc.lower() != "github.com":
        return None
    found = _GITHUB_FEED_PATH.match(parts.path)
    label = f"{found['owner']}/{found['name']}" if found else ""
    return label if is_github_name(label) else None


def watch_kind(url: str) -> str:
    """The kind of an item from a watched feed: `release` and `commit` for the two GitHub feeds,
    `article` for any other."""
    parts = urlsplit(url)
    if parts.netloc.lower() == "github.com":
        if _RELEASES_PATH.match(parts.path):
            return "release"
        if _COMMITS_PATH.match(parts.path):
            return "commit"
    return "article"


def entry_key(feed: str, entry_id: str) -> str:
    """The id an entry is stored under. An id that starts with a URI scheme (a GitHub `tag:` id, an
    http(s) guid) is unique everywhere and is used as it is; any other id, such as a guid `1234`, is
    only unique inside its feed and is prefixed with the feed's URL."""
    return entry_id if _URI_SCHEME.match(entry_id) else f"{feed}#{entry_id}"


@dataclass
class FeedRef:
    """One feed to read: where, what to call it, and whether it is a watch entry (and for which topic)."""

    url: str
    label: str | None = None  # owner/name for a watched github.com feed; else the feed's own title is used
    watch_topic: str | None = None
    entry: str | None = None  # a watch entry as written (owner/name or a URL); what `radar doctor` calls the feed


def configured(config: Config) -> list[FeedRef]:
    """Every feed the topics list: watch entries first, in file order, then `feeds.urls`.
    A feed listed twice is read once, under its first mention."""
    refs: dict[str, FeedRef] = {}
    for topic in config.topics:
        for entry in topic.watch:
            url = feed_url(entry)
            refs.setdefault(url, FeedRef(url, github_label(url), topic.name, entry.strip()))
    for topic in config.topics:
        for url in topic.source_option(NAME, "urls") or []:
            refs.setdefault(str(url).strip(), FeedRef(str(url).strip()))
    return list(refs.values())


@dataclass
class Entry:
    id: str
    title: str
    link: str
    published: datetime | None
    text: str


@dataclass
class Feed:
    title: str
    entries: list[Entry] = field(default_factory=list)


def plain_text(value: str) -> str:
    """HTML to plain text: scripts and styles dropped, tags stripped, entities unescaped, cut to MAX_TEXT."""
    text = _TAGS.sub(" ", _SCRIPTS.sub(" ", value or ""))
    return " ".join(html.unescape(text).split())[:MAX_TEXT]


def _text(element: ET.Element | None) -> str:
    """The text of an element, including the children of an XHTML content element."""
    if element is None:
        return ""
    if len(element):
        return " ".join(element.itertext())
    return element.text or ""


def _moment(value: str | None, rss: bool) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        moment = parsedate_to_datetime(value) if rss else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, IndexError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _atom_link(entry: ET.Element, base: str) -> str:
    links = entry.findall(f"{_ATOM}link")
    chosen = next((link for link in links if link.get("rel", "alternate") == "alternate" and link.get("href")), None)
    chosen = chosen if chosen is not None else next((link for link in links if link.get("href")), None)
    return urljoin(base, chosen.get("href", "").strip()) if chosen is not None else ""


def parse_feed(data: bytes | str, url: str = "") -> Feed:
    """Read an Atom or RSS 2.0 document. An entry without an id or an http(s) link is left out; the
    caller filters by date. Raises SourceError for a document that is too large, declares entities,
    is not XML or is neither Atom nor RSS."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    if len(data) > MAX_BYTES:
        raise SourceError(f"the feed is larger than {MAX_BYTES // 1_000_000} MB, so it was skipped")
    if b"<!ENTITY" in data:  # entity tricks can blow a small file up in memory
        raise SourceError("the feed declares XML entities, so it was skipped")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise SourceError(f"not a readable feed ({exc})") from exc

    feed = Feed(title=" ".join(_text(root.find(f"{_ATOM}title")).split()))
    if root.tag == f"{_ATOM}feed":
        for raw in root.findall(f"{_ATOM}entry"):
            link = _atom_link(raw, url)
            entry_id = _text(raw.find(f"{_ATOM}id")).strip() or link
            content = raw.find(f"{_ATOM}content")
            body = _text(content if content is not None and _text(content).strip() else raw.find(f"{_ATOM}summary"))
            when = _moment(_text(raw.find(f"{_ATOM}published")) or _text(raw.find(f"{_ATOM}updated")), rss=False)
            _keep(feed, entry_id, _text(raw.find(f"{_ATOM}title")), link, when, body)
    elif root.tag == "rss":
        channel = root.find("channel")
        if channel is None:
            raise SourceError("the RSS feed has no channel")
        feed.title = " ".join(_text(channel.find("title")).split())
        for raw in channel.findall("item"):
            href = _text(raw.find("link")).strip()
            link = urljoin(url, href) if href else ""
            entry_id = _text(raw.find("guid")).strip() or link
            _keep(feed, entry_id, _text(raw.find("title")), link, _moment(_text(raw.find("pubDate")), rss=True), _text(raw.find("description")))
    else:
        raise SourceError("not an Atom or RSS feed")
    return feed


def _keep(feed: Feed, entry_id: str, title: str, link: str, when: datetime | None, body: str) -> None:
    if not entry_id or not is_feed_url(link):
        return
    feed.entries.append(Entry(entry_id, " ".join(html.unescape(title).split()), link, when, plain_text(body)))


def read_feed(http: Http, url: str) -> Feed:
    """Fetch and parse one feed. Raises SourceError."""
    response = http.get(url, bucket="feeds:" + urlsplit(url).netloc, min_interval=1.0)
    return parse_feed(response.content, url)


def describe_feed(feed: Feed, url: str) -> str:
    """What `radar doctor` prints for a feed it read: the entry count and the newest date."""
    dates = [e.published for e in feed.entries if e.published]
    if not feed.entries:
        hint = (
            ". A repository without releases has an empty releases feed: watch its commits feed "
            + url.removesuffix("releases.atom")
            + "commits.atom instead"
            if is_github_releases(url)
            else ""
        )
        return "no entries" + hint
    newest = f", newest {_date(max(dates))}" if dates else ", no dates"
    return f"{len(feed.entries)} {'entry' if len(feed.entries) == 1 else 'entries'}{newest}"


def _date(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).date().isoformat()


def fold_earlier(newer: Item, older: Item) -> None:
    """Count `older`, an item of the same watched feed from an earlier window, and what it counted
    itself, as earlier entries of `newer`."""
    count = 1 + int(older.meta.get("earlier_entries") or 0)
    newer.meta["earlier_entries"] = int(newer.meta.get("earlier_entries") or 0) + count
    since = older.meta.get("earlier_since") or (older.published or "")[:10]
    known = newer.meta.get("earlier_since")
    if since and (not known or since < known):
        newer.meta["earlier_since"] = since


def _item(ref: FeedRef, feed: Feed, entry: Entry) -> Item:
    label = ref.label or feed.title or urlsplit(ref.url).netloc
    meta: dict = {"feed": ref.url, "entry_id": entry_key(ref.url, entry.id), "label": label}
    if ref.watch_topic:
        meta["watch_topic"] = ref.watch_topic
    return Item(
        source=NAME,
        kind=watch_kind(ref.url) if ref.watch_topic else "article",
        title=f"{label}: {entry.title or entry.link}",
        url=entry.link,
        text=entry.text,
        published=entry.published.isoformat() if entry.published else None,
        meta=meta,
    )


def fetch(ctx: FetchContext) -> list[Item]:
    items: list[Item] = []
    for ref in configured(ctx.config):
        try:
            feed = read_feed(ctx.http, ref.url)
        except SourceError as exc:
            ctx.warn(f"{NAME}: {ref.entry or ref.url}: {exc}")
            continue
        dated = [(e.published, e) for e in feed.entries if e.published and ctx.since <= e.published <= ctx.until]
        dated.sort(key=lambda pair: pair[0], reverse=True)  # stable: the entry listed first wins a tie
        if not dated:
            continue
        if ref.watch_topic:
            # One item per watched feed and scan: a burst of releases is one item and a count.
            item = _item(ref, feed, dated[0][1])
            if len(dated) > 1:
                item.meta["earlier_entries"] = len(dated) - 1
                item.meta["earlier_since"] = _date(dated[-1][0])
            items.append(item)
        else:
            items.extend(_item(ref, feed, e) for _, e in dated[:MAX_ARTICLES])
    return items
