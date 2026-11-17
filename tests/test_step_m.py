"""Step M: one message after two days without a saved scan (decision 43)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from research_radar import cli, deliver, pipeline
from research_radar.lock import scan_lock
from research_radar.store import Store

from conftest import FakeOllama
from test_digest_and_delivery import smtp  # noqa: F401  (smtp is a fixture)
from test_step_h import CHAT, tg  # noqa: F401  (tg is a fixture)

TODAY = date(2026, 11, 14)
DOWN = (
    "Ollama is not reachable at http://localhost:11434, so a scan can only match by keywords. "
    "Start Ollama (on Windows: from the Start menu), then run the scan again."
)
MODELS = ["nomic-embed-text:latest", "qwen3.5:9b"]


def both_channels(make_cfg, **settings):
    cfg = make_cfg(**settings)
    cfg.delivery["email"].update({"enabled": True, "username": "me@example.com", "to": "me@example.com"})
    cfg.delivery["telegram"].update({"enabled": True, "chat_id": CHAT})
    return cfg


def at_noon(day: date) -> str:
    """Noon of that local day, written in UTC as a scan writes it."""
    return datetime(day.year, day.month, day.day, 12, 0).astimezone().astimezone(timezone.utc).isoformat()


def saved_scan(cfg, *days: date) -> Store:
    """last_run.json as scans on those local days left it, one source per day."""
    store = Store(cfg.state_dir)
    store.save_last_runs({f"source-{n}": at_noon(day) for n, day in enumerate(days)})
    return store


def notice(cfg, today=TODAY, reason=DOWN):
    return deliver.send_skip_notice(cfg, Store(cfg.state_dir), today, reason)


def texts(tg) -> list[str]:  # noqa: F811
    return [p["text"] for p in tg.of("sendMessage")]


# when a message goes out -----------------------------------------------------------------------


def test_nothing_is_sent_when_no_scan_was_ever_saved(make_cfg, smtp, tg):  # noqa: F811
    cfg = both_channels(make_cfg)
    assert notice(cfg) == []
    assert smtp.outbox == [] and tg.calls == []
    assert not (cfg.state_dir / "delivered.json").exists()


def test_nothing_is_sent_one_day_after_a_saved_scan(make_cfg, smtp, tg):  # noqa: F811
    cfg = both_channels(make_cfg)
    saved_scan(cfg, date(2026, 11, 13))
    assert notice(cfg) == []
    assert smtp.outbox == [] and tg.calls == []


def test_two_days_after_a_saved_scan_each_enabled_channel_gets_one_message(make_cfg, smtp, tg):  # noqa: F811
    cfg = both_channels(make_cfg)
    saved_scan(cfg, date(2026, 11, 12))
    results = notice(cfg)
    assert [(r.channel, r.ok) for r in results] == [("email", True), ("telegram", True)]
    expected = (
        f"Research Radar has not saved a scan since 12 November. Today: {DOWN} "
        "The next scan that runs collects what was missed, up to 14 days back."
    )
    assert len(smtp.outbox) == 1
    mail = smtp.outbox[0]
    assert mail["Subject"] == "Research Radar: no scan since 12 November"
    assert mail.get_body(("plain",)).get_content().strip() == expected
    sent = tg.of("sendMessage")
    assert len(sent) == 1
    assert sent[0]["text"] == expected and sent[0]["chat_id"] == CHAT
    assert "disable_notification" not in sent[0]  # with sound
    assert Store(cfg.state_dir).delivered() == {"email": ["skip:2026-11-12"], "telegram": ["skip:2026-11-12"]}
    assert deliver.describe(results) == [
        "email: told you that no scan was saved since 12 November",
        "telegram: told you that no scan was saved since 12 November",
    ]


def test_a_channel_that_is_off_gets_nothing(make_cfg, smtp, tg):  # noqa: F811
    cfg = both_channels(make_cfg)
    cfg.delivery["email"]["enabled"] = False
    saved_scan(cfg, date(2026, 11, 12))
    assert [r.channel for r in notice(cfg)] == ["telegram"]
    assert smtp.outbox == [] and len(texts(tg)) == 1


def test_the_newest_source_time_counts_and_its_local_date_is_named(make_cfg, tg):  # noqa: F811
    cfg = both_channels(make_cfg, max_catchup_days=7)
    cfg.delivery["email"]["enabled"] = False
    just_after_midnight = datetime(2026, 11, 12, 0, 30).astimezone().astimezone(timezone.utc).isoformat()
    Store(cfg.state_dir).save_last_runs({"github": at_noon(date(2026, 11, 3)), "arxiv": just_after_midnight, "feeds": ""})
    notice(cfg, reason="Another scan has been running since 19:40.\nA second line.")
    assert texts(tg) == [
        "Research Radar has not saved a scan since 12 November. Today: Another scan has been running since 19:40. "
        "The next scan that runs collects what was missed, up to 7 days back."
    ]


def test_markup_in_the_reason_is_escaped_for_telegram(make_cfg, tg):  # noqa: F811
    cfg = both_channels(make_cfg)
    cfg.delivery["email"]["enabled"] = False
    saved_scan(cfg, date(2026, 11, 12))
    notice(cfg, reason="Ollama is not reachable at http://host/?a=1&b=<2>")
    assert "http://host/?a=1&amp;b=&lt;2&gt;." in texts(tg)[0]


# once per gap and channel ----------------------------------------------------------------------


def test_a_second_skipped_run_sends_nothing(make_cfg, smtp, tg):  # noqa: F811
    cfg = both_channels(make_cfg)
    saved_scan(cfg, date(2026, 11, 12))
    notice(cfg)
    again = notice(cfg)
    next_day = notice(cfg, today=TODAY + timedelta(days=1))
    assert len(smtp.outbox) == 1 and len(texts(tg)) == 1
    for results in (again, next_day):
        assert deliver.describe(results) == [
            "email: already told you that no scan was saved since 12 November",
            "telegram: already told you that no scan was saved since 12 November",
        ]


def test_after_a_saved_scan_a_new_gap_sends_again(make_cfg, smtp, tg):  # noqa: F811
    cfg = both_channels(make_cfg)
    saved_scan(cfg, date(2026, 11, 12))
    notice(cfg)
    saved_scan(cfg, date(2026, 11, 15))
    assert notice(cfg, today=date(2026, 11, 16)) == []  # one day into the new gap
    assert len(smtp.outbox) == 1 and len(texts(tg)) == 1
    notice(cfg, today=date(2026, 11, 17))
    assert len(smtp.outbox) == 2 and len(texts(tg)) == 2
    assert smtp.outbox[1]["Subject"] == "Research Radar: no scan since 15 November"
    assert "since 15 November" in texts(tg)[1]
    keys = ["skip:2026-11-12", "skip:2026-11-15"]
    assert Store(cfg.state_dir).delivered() == {"email": keys, "telegram": keys}


def test_a_failed_send_leaves_no_key_and_the_next_skipped_run_tries_again(make_cfg, smtp, tg):  # noqa: F811
    cfg = both_channels(make_cfg)
    saved_scan(cfg, date(2026, 11, 12))
    tg.fail["sendMessage"] = True
    first = notice(cfg)
    assert [(r.channel, r.ok) for r in first] == [("email", True), ("telegram", False)]
    assert deliver.describe(first)[1].startswith("telegram: not sent, Telegram refused the request")
    assert Store(cfg.state_dir).delivered() == {"email": ["skip:2026-11-12"]}
    tg.fail.clear()
    second = notice(cfg, today=TODAY + timedelta(days=1))
    assert all(r.ok for r in second)
    assert len(smtp.outbox) == 1  # the mail went out the first time and is not repeated
    assert Store(cfg.state_dir).delivered()["telegram"] == ["skip:2026-11-12"]


# the scheduled scan ----------------------------------------------------------------------------


def scan_saved_days_ago(cfg, days=5) -> None:
    Store(cfg.state_dir).save_last_runs({"github": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()})


def scheduled(cfg, monkeypatch, names=MODELS) -> int:
    monkeypatch.setattr(cli, "_load", lambda home_arg: cfg)
    monkeypatch.setattr(cli, "_ollama_names", lambda c: names)
    return cli.cmd_scheduled_scan(str(cfg.home), None, False, False, sleep=lambda s: None)


def log_of(cfg) -> str:
    return (cfg.home / "logs" / "scan.log").read_text(encoding="utf-8")


@pytest.fixture
def telegram_only(make_cfg):
    cfg = both_channels(make_cfg)
    cfg.delivery["email"]["enabled"] = False
    scan_saved_days_ago(cfg)
    return cfg


def test_a_skipped_run_names_ollama_being_down(telegram_only, tg, monkeypatch):  # noqa: F811
    assert scheduled(telegram_only, monkeypatch, names=None) == cli.EXIT_SKIPPED
    assert len(texts(tg)) == 1 and "Today: Ollama is not " in texts(tg)[0]
    log = log_of(telegram_only)
    assert "telegram: told you that no scan was saved since" in log and log.rstrip().endswith(f"exit code {cli.EXIT_SKIPPED}")


def test_a_skipped_run_names_the_missing_embedding_model(telegram_only, tg, monkeypatch):  # noqa: F811
    assert scheduled(telegram_only, monkeypatch, names=["qwen3.5:9b"]) == cli.EXIT_SKIPPED
    assert "Today: The embedding model nomic-embed-text is not downloaded" in texts(tg)[0]


def test_a_skipped_run_names_the_keyword_preview(telegram_only, tg, http_factory, monkeypatch):  # noqa: F811
    monkeypatch.setattr(pipeline, "Http", lambda **k: http_factory())
    monkeypatch.setattr(pipeline, "Ollama", lambda *a, **k: FakeOllama(up=False))
    assert scheduled(telegram_only, monkeypatch) == cli.EXIT_SKIPPED
    assert "Today: Keyword preview: nothing was saved or sent." in texts(tg)[0]


def test_a_skipped_run_names_the_other_scan(telegram_only, tg, http_factory, monkeypatch):  # noqa: F811
    monkeypatch.setattr(pipeline, "Http", lambda **k: http_factory())
    monkeypatch.setattr(pipeline, "Ollama", lambda *a, **k: FakeOllama())
    with scan_lock(telegram_only.state_dir):
        assert scheduled(telegram_only, monkeypatch) == cli.EXIT_SKIPPED
    assert "Today: Another scan has been running since " in texts(tg)[0]


def test_a_run_that_was_not_skipped_sends_nothing(telegram_only, tg, monkeypatch):  # noqa: F811
    for code in (0, 1):
        monkeypatch.setattr(cli, "cmd_scan", lambda *a, code=code, **k: code)
        assert scheduled(telegram_only, monkeypatch) == code
    assert tg.calls == []


def test_a_failed_send_keeps_the_exit_code_and_goes_to_the_log(telegram_only, tg, monkeypatch):  # noqa: F811
    tg.fail["sendMessage"] = True
    assert scheduled(telegram_only, monkeypatch, names=None) == cli.EXIT_SKIPPED
    assert "telegram: not sent, Telegram refused the request" in log_of(telegram_only)
    assert Store(telegram_only.state_dir).delivered() == {}

    def boom(*a, **k):
        raise RuntimeError("state folder exploded")

    monkeypatch.setattr(deliver, "send_skip_notice", boom)
    assert scheduled(telegram_only, monkeypatch, names=None) == cli.EXIT_SKIPPED
    log = log_of(telegram_only)
    assert "was not sent (RuntimeError)" in log and log.rstrip().endswith(f"exit code {cli.EXIT_SKIPPED}")
