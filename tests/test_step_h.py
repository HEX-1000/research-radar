"""Step H: Telegram messages with vote buttons, delivery marks per item, vote collection."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from research_radar import cli, deliver
from research_radar.candidates import CandidateStore
from research_radar.config import init_home
from research_radar.digest import Digest, render_telegram, vote_hash
from research_radar.models import Report
from research_radar.store import Store
from test_step_g import MODEL, Db, make_cfg

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
TODAY = "2026-11-14"
TOKEN = "123456:SECRET-TOKEN"
CHAT = "42"


def report(name, verdict="read", topic="Alpha", **extra) -> Report:
    return Report(
        date=TODAY,
        topic=topic,
        score=0.6,
        source="github",
        kind="repo",
        title=f"alice/{name}",
        url=f"https://github.com/alice/{name}",
        what="Does a thing.",
        why="Fits.",
        verdict=verdict,
        meta={"stars": 10},
        **extra,
    )


class FakeTelegram:
    """The Bot API behind a mock transport; records every call as (method, payload)."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, dict]] = []
        self.updates: list[dict] = []
        self.fail: dict[str, int | bool] = {}  # method -> nth call (1 based) that is refused; True is every call
        self.counts: dict[str, int] = {}
        monkeypatch.setenv("RADAR_TELEGRAM_TOKEN", TOKEN)
        monkeypatch.setattr(deliver, "_telegram_client", lambda: httpx.Client(transport=httpx.MockTransport(self.handle)))

    def handle(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[1]
        payload = json.loads(request.content)
        self.calls.append((method, payload))
        self.counts[method] = self.counts.get(method, 0) + 1
        rule = self.fail.get(method)
        if rule is True or rule == self.counts[method]:
            return httpx.Response(400, json={"ok": False, "description": "refused"})
        if method == "getUpdates":
            offset = payload.get("offset")
            if offset is not None:
                self.updates = [u for u in self.updates if u["update_id"] >= offset]
            return httpx.Response(200, json={"ok": True, "result": list(self.updates)[: payload.get("limit", 100)]})
        return httpx.Response(200, json={"ok": True, "result": True})

    def of(self, method):
        return [p for m, p in self.calls if m == method]

    def press(self, update_id, url, sign, message_id=7, chat=CHAT, data=None):
        self.updates.append(
            {
                "update_id": update_id,
                "callback_query": {
                    "data": data if data is not None else f"v1{sign}{vote_hash(url)}",
                    "message": {"message_id": message_id, "chat": {"id": int(chat)}},
                },
            }
        )


@pytest.fixture
def tg(monkeypatch):
    return FakeTelegram(monkeypatch)


def telegram_cfg(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.delivery["telegram"].update({"enabled": True, "chat_id": CHAT})
    return cfg


# sending ------------------------------------------------------------------------------------


def test_messages_buttons_and_notification_flags(tmp_path, tg):
    cfg = telegram_cfg(tmp_path)
    store = Store(cfg.state_dir)
    items = [report("one", "install"), report("two", "read", topic="Beta"), report("three", "skip")]
    store.merge_day(TODAY, items, [], ["a note"], "embeddings", at=NOW.isoformat())
    result = deliver.deliver_pending(cfg, store, TODAY)[0]
    assert result.ok and result.sent == 3
    sent = tg.of("sendMessage")
    assert len(sent) == 4  # header, two items with buttons, one "also matched" with the notes
    assert sent[0]["text"].startswith("<b>Research Radar</b>") and "disable_notification" not in sent[0]
    assert all(m["disable_notification"] is True for m in sent[1:])
    assert all(m["parse_mode"] == "HTML" for m in sent)
    one = sent[1]
    assert one["text"].startswith("<i>Alpha</i>\n") and "alice/one" in one["text"]
    h = vote_hash("https://github.com/alice/one")
    assert len(h) == 16
    assert one["reply_markup"] == {
        "inline_keyboard": [[{"text": "+", "callback_data": "v1+" + h}, {"text": "-", "callback_data": "v1-" + h}]]
    }
    assert sent[2]["text"].startswith("<i>Beta</i>\n")
    assert "reply_markup" not in sent[3] and "Also matched" in sent[3]["text"] and "a note" in sent[3]["text"]
    assert len(one["reply_markup"]["inline_keyboard"][0][0]["callback_data"]) <= 64


def test_a_failed_message_leaves_the_earlier_items_delivered(tmp_path, tg):
    cfg = telegram_cfg(tmp_path)
    store = Store(cfg.state_dir)
    items = [report(n) for n in ("a", "b", "c", "d", "e")]
    store.merge_day(TODAY, items, [], [], "embeddings", at=NOW.isoformat())
    tg.fail["sendMessage"] = 5  # the header and three items go out, the fourth item fails
    first = deliver.deliver_pending(cfg, store, TODAY)[0]
    assert not first.ok and first.sent == 3
    assert len(store.delivered()["telegram"]) == 3
    again = deliver.deliver_pending(cfg, store, TODAY)[0]
    assert again.ok and again.sent == 2
    texts = [p["text"] for p in tg.of("sendMessage")[5:]]
    assert "2 new items" in texts[0]  # a new header for the rest
    assert len(store.delivered()["telegram"]) == 5


def test_a_failed_header_delivers_nothing(tmp_path, tg):
    cfg = telegram_cfg(tmp_path)
    store = Store(cfg.state_dir)
    store.merge_day(TODAY, [report("a")], [], [], "embeddings", at=NOW.isoformat())
    tg.fail["sendMessage"] = 1
    result = deliver.deliver_pending(cfg, store, TODAY)[0]
    assert not result.ok and result.sent == 0 and not store.delivered().get("telegram")


def test_the_pause_between_messages_is_at_least_a_second(tmp_path, tg, monkeypatch):
    waits = []
    monkeypatch.setattr(deliver.time, "sleep", waits.append)
    monkeypatch.setattr(deliver, "TELEGRAM_PAUSE", 1.0)
    cfg = telegram_cfg(tmp_path)
    store = Store(cfg.state_dir)
    store.merge_day(TODAY, [report("a"), report("b")], [], [], "embeddings", at=NOW.isoformat())
    deliver.deliver_pending(cfg, store, TODAY)
    assert waits == [1.0, 1.0]


def test_render_without_items_is_only_a_header():
    messages = render_telegram(Digest(date=TODAY, reports=[]))
    assert len(messages) == 1 and messages[0].sound and messages[0].keyboard is None


# collecting ---------------------------------------------------------------------------------


def shown_store(tmp_path, names=("one", "two")):
    cfg = telegram_cfg(tmp_path)
    db = Db(cfg)
    for name in names:
        key = f"github:alice/{name}"
        db.item(key)
        db.conn.execute("UPDATE item SET state = 'shown' WHERE key = ?", (key,))
        db.conn.execute("UPDATE link SET url = ? WHERE key = ?", (f"https://github.com/alice/{name}", key))
    return cfg, db


def collect(cfg):
    with CandidateStore.open(cfg.state_dir, embed_model=MODEL, now=NOW) as store:
        return deliver.collect_telegram_votes(cfg, store, NOW)


URL_ONE = "https://github.com/alice/one"
URL_TWO = "https://github.com/alice/two"


def test_votes_are_mapped_written_and_confirmed(tmp_path, tg):
    cfg, db = shown_store(tmp_path)
    tg.press(10, URL_ONE, "+", message_id=7)
    tg.press(11, URL_TWO, "-", message_id=8)
    assert collect(cfg) == (2, "")
    rows = db.rows("SELECT key, value, via FROM vote ORDER BY key")
    assert rows == [("github:alice/one", 1, "telegram"), ("github:alice/two", -1, "telegram")]
    confirm = [p for p in tg.of("getUpdates") if "offset" in p]
    assert confirm == [{"offset": 12, "limit": 1}]
    edits = tg.of("editMessageReplyMarkup")
    assert {(e["message_id"], e["reply_markup"]["inline_keyboard"][0][0 if e["message_id"] == 7 else 1]["text"]) for e in edits} == {
        (7, "+ ✓"),
        (8, "- ✓"),
    }
    first = tg.of("getUpdates")[0]
    assert "offset" not in first and "allowed_updates" not in first and first["limit"] == 100


def test_other_chats_bad_data_and_unknown_hashes_are_ignored(tmp_path, tg):
    cfg, db = shown_store(tmp_path)
    tg.press(10, URL_ONE, "+", chat="99")
    tg.press(11, URL_ONE, "+", data="v1+nothex")
    tg.press(12, URL_ONE, "+", data="v2+" + vote_hash(URL_ONE))
    tg.press(13, "https://github.com/alice/unknown", "+")
    tg.updates.append({"update_id": 14, "message": {"chat": {"id": 42}, "text": "hi"}})
    assert collect(cfg) == (0, "")
    assert db.rows("SELECT * FROM vote") == []
    assert tg.of("getUpdates")[-1] == {"offset": 15, "limit": 1}  # all of them are confirmed, none counted


def test_the_last_press_wins_and_a_repeated_press_keeps_the_vote(tmp_path, tg):
    cfg, db = shown_store(tmp_path)
    db.vote("github:alice/one", -1)  # the vote from the interface is replaced
    tg.press(10, URL_ONE, "-")
    tg.press(11, URL_ONE, "+")
    tg.press(12, URL_ONE, "+")
    tg.press(13, URL_TWO, "+")
    tg.press(14, URL_TWO, "-")
    assert collect(cfg)[0] == 2
    assert db.rows("SELECT key, value FROM vote ORDER BY key") == [("github:alice/one", 1), ("github:alice/two", -1)]


def test_updates_are_confirmed_only_after_the_commit(tmp_path, tg, monkeypatch):
    cfg, db = shown_store(tmp_path)
    tg.press(10, URL_ONE, "+")
    order = []
    original = CandidateStore.set_votes

    def spy(self, *a, **k):
        original(self, *a, **k)
        order.append("commit")

    monkeypatch.setattr(CandidateStore, "set_votes", spy)
    real = tg.handle

    def watch(request):
        if b'"offset"' in request.content:
            order.append("confirm")
        return real(request)

    monkeypatch.setattr(deliver, "_telegram_client", lambda: httpx.Client(transport=httpx.MockTransport(watch)))
    collect(cfg)
    assert order == ["commit", "confirm"]


def test_a_failed_confirm_writes_the_same_votes_again(tmp_path, tg):
    cfg, db = shown_store(tmp_path)
    tg.press(10, URL_ONE, "+")
    tg.fail["getUpdates"] = 2  # the read works, the confirming call is refused
    count, warning = collect(cfg)
    assert count == 0 and "Could not collect" in warning and TOKEN not in warning
    assert db.rows("SELECT key, value FROM vote") == [("github:alice/one", 1)]  # written before the confirm
    assert tg.of("editMessageReplyMarkup") == []
    tg.fail.clear()
    assert collect(cfg) == (1, "")  # the update is still there and gives the same vote
    assert db.rows("SELECT key, value FROM vote") == [("github:alice/one", 1)]


def test_a_failed_edit_is_ignored(tmp_path, tg):
    cfg, db = shown_store(tmp_path)
    tg.press(10, URL_ONE, "+")
    tg.press(11, URL_TWO, "+")
    tg.fail["editMessageReplyMarkup"] = True
    assert collect(cfg) == (2, "")
    assert len(db.rows("SELECT * FROM vote")) == 2


def test_a_failed_read_writes_nothing_and_does_not_raise(tmp_path, tg):
    cfg, db = shown_store(tmp_path)
    tg.fail["getUpdates"] = True
    count, warning = collect(cfg)
    assert count == 0 and warning.startswith("Could not collect the Telegram votes")
    assert db.rows("SELECT * FROM vote") == []


def test_no_store_file_means_no_call(tmp_path, tg):
    cfg = telegram_cfg(tmp_path)
    with CandidateStore.open(cfg.state_dir, embed_model=MODEL, now=NOW) as store:  # built in memory
        assert deliver.collect_telegram_votes(cfg, store, NOW) == (0, "")
    assert tg.calls == []
    with CandidateStore.open(cfg.state_dir, embed_model=MODEL, now=NOW, write=False) as store:
        assert deliver.collect_telegram_votes(cfg, store, NOW) == (0, "")
    assert tg.calls == []


def test_a_full_page_is_followed_by_the_next(tmp_path, tg):
    cfg, db = shown_store(tmp_path)
    for i in range(100):
        tg.updates.append({"update_id": i + 1, "message": {"chat": {"id": 42}}})
    tg.press(101, URL_ONE, "+")
    assert collect(cfg) == (1, "")
    pages = [p for p in tg.of("getUpdates") if p.get("limit") == 100]
    assert len(pages) == 2 and pages[1]["offset"] == 101


# collecting in a skipped scheduled scan ---------------------------------------------------------


def test_a_skipped_scheduled_scan_still_collects_votes(tmp_path, tg, monkeypatch, capsys):
    home = tmp_path / "home"
    init_home(home)
    cfg = telegram_cfg(tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli, "_load", lambda home_arg: cfg)
    monkeypatch.setattr(cli, "_ollama_names", lambda c: None)
    monkeypatch.setattr(cli, "resolve_home", lambda arg: home)
    db = Db(cfg)
    db.item("github:alice/one")
    db.conn.execute("UPDATE item SET state = 'shown' WHERE key = 'github:alice/one'")
    db.conn.execute("UPDATE link SET url = ? WHERE key = 'github:alice/one'", (URL_ONE,))
    tg.press(10, URL_ONE, "+")
    code = cli.cmd_scheduled_scan(None, None, False, False, sleep=lambda s: None)
    assert code == cli.EXIT_SKIPPED
    assert db.rows("SELECT key, value, via FROM vote") == [("github:alice/one", 1, "telegram")]
    log = (home / "logs" / "scan.log").read_text(encoding="utf-8")
    assert "Telegram votes recorded: 1" in log


def test_a_skipped_scan_without_a_store_makes_no_telegram_call(tmp_path, tg, monkeypatch):
    home = tmp_path / "home"
    init_home(home)
    cfg = telegram_cfg(tmp_path)
    monkeypatch.setattr(cli, "_load", lambda home_arg: cfg)
    monkeypatch.setattr(cli, "_ollama_names", lambda c: None)
    monkeypatch.setattr(cli, "resolve_home", lambda arg: home)
    assert cli.cmd_scheduled_scan(None, None, False, False, sleep=lambda s: None) == cli.EXIT_SKIPPED
    assert tg.calls == []
