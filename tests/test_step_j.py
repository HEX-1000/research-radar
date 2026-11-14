"""Step J: Telegram pages committed one by one (decision 22), `-`-voted repositories asked about
and listed by name (decision 23)."""

from __future__ import annotations

import httpx
import pytest

from research_radar import deliver
from research_radar.candidates import CandidateStore, StoreError
from research_radar.labelling import format_replay, replay, run_label
from test_step_g import NOW, AxisOllama, Db, make_cfg, pool_of, scripted
from test_step_h import collect, shown_store, tg  # noqa: F401  (tg is a fixture)

URLS = {n: f"https://github.com/alice/{n}" for n in ("one", "two", "three", "four", "five")}


@pytest.fixture(autouse=True)
def small_pages(monkeypatch):
    monkeypatch.setattr(deliver, "UPDATE_PAGE", 2)


def offsets(tg):
    return [(p.get("offset"), p["limit"]) for p in tg.of("getUpdates")]


# Telegram pages ---------------------------------------------------------------------------------


def test_two_pages_are_committed_before_the_calls_that_confirm_them(tmp_path, tg, monkeypatch):
    cfg, db = shown_store(tmp_path, ("one", "two", "three"))
    tg.press(10, URLS["one"], "+")
    tg.press(11, URLS["two"], "-", message_id=8)
    tg.press(12, URLS["three"], "+", message_id=9)  # second, short page
    order = []
    real = CandidateStore.set_votes

    def commit(self, *args, **kwargs):
        order.append("commit")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(CandidateStore, "set_votes", commit)
    handle = tg.handle

    def watch(request):
        if b'"offset"' in request.content:
            order.append("call with offset")
        return handle(request)

    monkeypatch.setattr(deliver, "_telegram_client", lambda: httpx.Client(transport=httpx.MockTransport(watch)))
    assert collect(cfg) == (3, "")
    assert offsets(tg) == [(None, 2), (12, 2), (13, 1)]
    assert order == ["commit", "call with offset", "commit", "call with offset"]
    assert len(db.rows("SELECT key FROM vote")) == 3
    assert len(tg.of("editMessageReplyMarkup")) == 3


def test_a_failed_commit_on_the_second_page_keeps_the_first_page(tmp_path, tg, monkeypatch):
    cfg, db = shown_store(tmp_path, ("one", "two", "three"))
    tg.press(10, URLS["one"], "+")
    tg.press(11, URLS["two"], "-", message_id=8)
    tg.press(12, URLS["three"], "+", message_id=9)
    real = CandidateStore.set_votes
    calls = []

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise StoreError("disk full")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(CandidateStore, "set_votes", flaky)
    count, warning = collect(cfg)
    assert count == 2 and "Could not collect" in warning
    assert sorted(db.rows("SELECT key FROM vote")) == [("github:alice/one",), ("github:alice/two",)]
    assert offsets(tg) == [(None, 2), (12, 2)]  # no call carries the second page's offset (13)
    assert len(tg.of("editMessageReplyMarkup")) == 2  # edits for the committed page only
    assert [u["update_id"] for u in tg.updates] == [12]  # the second page is read again next time


def test_an_item_pressed_on_both_pages_ends_with_the_second_vote(tmp_path, tg):
    cfg, db = shown_store(tmp_path, ("one", "two"))
    tg.press(10, URLS["one"], "+")
    tg.press(11, URLS["two"], "+", message_id=8)
    tg.press(12, URLS["one"], "-")
    assert collect(cfg) == (2, "")
    assert db.rows("SELECT key, value FROM vote ORDER BY key") == [("github:alice/one", -1), ("github:alice/two", 1)]
    edits = [e for e in tg.of("editMessageReplyMarkup") if e["message_id"] == 7]
    assert edits[-1]["reply_markup"]["inline_keyboard"][0][1]["text"] == "- ✓"


def test_a_failed_final_confirm_marks_the_vote_the_store_holds(tmp_path, tg):
    cfg, db = shown_store(tmp_path, ("one", "two"))
    tg.press(10, URLS["one"], "+")
    tg.press(11, URLS["two"], "+", message_id=8)
    tg.press(12, URLS["one"], "-")  # the last page
    tg.fail["getUpdates"] = 3  # the call that confirms the last page is refused
    count, warning = collect(cfg)
    assert count == 2 and "Could not collect" in warning
    assert db.rows("SELECT key, value FROM vote ORDER BY key") == [("github:alice/one", -1), ("github:alice/two", 1)]
    edits = [e for e in tg.of("editMessageReplyMarkup") if e["message_id"] == 7]
    assert [e["reply_markup"]["inline_keyboard"][0][1]["text"] for e in edits] == ["- ✓"]
    assert [u["update_id"] for u in tg.updates] == [12]  # the last page is read again next time


def test_ten_full_pages_end_with_the_confirm_call(tmp_path, tg):
    cfg, db = shown_store(tmp_path, ("one",))
    for update_id in range(1, 21):  # ten full pages of two, none of them a vote
        tg.updates.append({"update_id": update_id, "message": {"text": "hi"}})
    tg.updates.append({"update_id": 21, "message": {"text": "left for the next run"}})
    assert collect(cfg) == (0, "")
    calls = offsets(tg)
    assert len(calls) == 11
    assert calls[0] == (None, 2) and calls[9] == (19, 2)
    assert calls[-1] == (21, 1)  # confirms page ten and nothing after it
    assert [u["update_id"] for u in tg.updates] == [21]


def test_an_empty_page_after_an_offset_ends_the_collection(tmp_path, tg):
    cfg, db = shown_store(tmp_path, ("one", "two"))
    tg.press(10, URLS["one"], "+")
    tg.press(11, URLS["two"], "-", message_id=8)  # exactly one full page
    assert collect(cfg) == (2, "")
    assert offsets(tg) == [(None, 2), (12, 2)]


# the label pool ---------------------------------------------------------------------------------


def minus_pool(tmp_path):
    cfg = make_cfg(tmp_path)  # Alpha bar 0.40
    db = Db(cfg)
    for key, topic, score, kind in [
        ("github:minus", "Alpha", 0.45, "repo"),
        ("github:minus-low", "Alpha", 0.30, "repo"),  # below the band
        ("github:plus", "Alpha", 0.45, "repo"),
        ("github:labelled", "Alpha", 0.45, "repo"),
        ("github:labelled-other", "Beta", 0.55, "repo"),
        ("arxiv:minus", "Alpha", 0.45, "paper"),
    ]:
        db.item(key, kind=kind)
        db.decide(key, topic, score, "shown")
    db.item("github:minus-old")
    db.decide("github:minus-old", "Alpha", 0.45, "shown", scan=1)  # outside the days
    for key, vote in [
        ("github:minus", -1),
        ("github:minus-low", -1),
        ("github:plus", 1),
        ("github:labelled", -1),
        ("github:labelled-other", -1),
        ("arxiv:minus", -1),
        ("github:minus-old", -1),
    ]:
        db.vote(key, vote)
    db.label("github:labelled", "Alpha", "no")
    db.label("github:labelled-other", "Beta", "fits")  # a label for any topic keeps it out
    return cfg, db


def test_only_a_minus_voted_repository_without_labels_joins_the_pool(tmp_path):
    cfg, db = minus_pool(tmp_path)
    pool = pool_of(cfg)
    assert [c.key for c in pool] == ["github:minus"]  # days, band, +, other sources and labelled stay out
    assert pool[0].vote == -1 and pool[0].topic == "Alpha" and pool[0].outcome == "shown"


def test_the_header_of_a_minus_voted_repository_ends_with_vote(tmp_path):
    cfg, db = minus_pool(tmp_path)
    shown = []
    run_label(cfg, ask=scripted(["q"]), show=shown.append, now=NOW)
    header = next(line for line in shown if "left]" in line)
    assert header.endswith("· shown · vote -")
    other = make_cfg(tmp_path / "other")
    plain = Db(other)
    plain.item("github:plain")
    plain.decide("github:plain", "Alpha", 0.45, "below_bar")
    shown = []
    run_label(other, ask=scripted(["q"]), show=shown.append, now=NOW)
    assert not next(line for line in shown if "left]" in line).endswith("vote -")


@pytest.mark.parametrize("answer, row", [("l", "lures"), ("n", "others")])
def test_l_and_n_move_a_minus_voted_repository_in_replay(tmp_path, answer, row):
    cfg, db = minus_pool(tmp_path)
    db.observe("github:minus", 4, 50, 0)
    before = replay(cfg, AxisOllama()).trust
    assert before.minus_only == 3 and before.lures["caught"] == 0 and before.others["caught"] == 0  # minus, minus-low, minus-old
    run_label(cfg, ask=scripted([answer]), show=lambda _: None, now=NOW)
    trust = replay(cfg, AxisOllama()).trust
    assert getattr(trust, row)["caught"] == 1
    assert trust.minus_only == 2  # the repository left the - line
    assert pool_of(cfg) == []


# names under the trust table --------------------------------------------------------------------


def test_names_of_others_caught_and_lures_not_caught(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    for key, kind, observation in [
        ("github:zed/other-caught", "fits", (126, 41, 0)),
        ("github:abe/other-unknown", "fits", (4, 50, None)),
        ("github:abe/other-safe", "fits", (5000, 400, 80)),
        ("github:lure-missed", "lure", (900, 50, 7)),
        ("github:lure-caught", "lure", (4, 50, 0)),
        ("github:lure-unknown", "lure", (4, 50, None)),
    ]:
        db.item(key)
        db.label(key, "*" if kind == "lure" else "Alpha", kind)
        db.observe(key, *observation)
    lines = format_replay(replay(cfg, AxisOllama()))
    first = lines.index("Others caught:")
    assert lines[first + 1] == "  zed/other-caught  126 KB, 41 stars, 0 followers"
    assert lines[first + 2] == "" and lines[first + 3] == "Lures not caught:"
    assert lines[first + 4] == "  lure-missed  900 KB, 50 stars, 7 followers"
    assert not any("other-unknown" in line or "lure-unknown" in line for line in lines)


def test_followers_never_asked_and_empty_lists_are_absent(tmp_path):
    cfg = make_cfg(tmp_path)
    db = Db(cfg)
    db.item("github:big-lure")
    db.label("github:big-lure", "*", "lure")
    db.observe("github:big-lure", 900, 50, None)  # too big for the rule: not caught, followers never asked
    lines = format_replay(replay(cfg, AxisOllama()))
    assert "Others caught:" not in lines
    assert "Lures not caught:" in lines
    assert "  big-lure  900 KB, 50 stars, followers not asked" in lines
    empty = make_cfg(tmp_path / "empty")
    Db(empty)
    lines = format_replay(replay(empty, AxisOllama()))
    assert not any(line in ("Others caught:", "Lures not caught:") for line in lines)
