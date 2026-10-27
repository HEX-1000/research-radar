from __future__ import annotations

import json
import smtplib
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import yaml

from research_radar import cli, deliver
from research_radar.config import ConfigError, example_text, init_home, parse_config
from research_radar.digest import Digest, render_html, render_telegram, render_text
from research_radar.models import Report, plain_title
from research_radar.pipeline import scan
from research_radar.store import Store
from research_radar.summarize import summarize
from research_radar.config import Topic
from research_radar.models import Item

from conftest import FakeOllama

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
TOKEN = "123456:SECRET-TOKEN"


def report(title="alice/tool", verdict="read", topic="Topic A", kind="repo", url=None, score=0.5, **extra) -> Report:
    return Report(
        date="2026-10-27",
        topic=topic,
        score=score,
        source="github" if kind == "repo" else "arxiv",
        kind=kind,
        title=title,
        url=url or f"https://example.org/{title.replace(' ', '-')}",
        what=extra.pop("what", "Does a thing."),
        why=extra.pop("why", "Fits the topic."),
        verdict=verdict,
        meta=extra.pop("meta", {"stars": 10, "license": "MIT", "language": "Python"} if kind == "repo" else {"arxiv_id": "1"}),
        **extra,
    )


# digest ---------------------------------------------------------------------------------


def test_digest_groups_by_topic_puts_actionable_first_and_folds_skips():
    digest = Digest(
        date="2026-10-27",
        topic_order=["Topic B", "Topic A"],
        reports=[
            report("watch-me", "watch", "Topic A"),
            report("try-me", "install", "Topic A"),
            report("never", "skip", "Topic A"),
            report("other", "read", "Topic B", kind="paper"),
        ],
    )
    groups = digest.groups()
    assert [name for name, _ in groups] == ["Topic B", "Topic A"]
    assert [r.title for r in groups[1][1]] == ["try-me", "watch-me"]
    assert [r.title for r in digest.also_matched] == ["never"]
    assert digest.worth_your_time == 2
    assert digest.subject() == "Research Radar 2026-10-27: 4 new items, 2 worth your time"
    text = render_text(digest)
    assert text.index("try-me") < text.index("watch-me") < text.index("Also matched")
    assert "- never" in text


def test_subject_for_an_empty_and_a_single_item_digest():
    assert Digest(date="2026-10-27", reports=[]).subject() == "Research Radar 2026-10-27: nothing new"
    assert "Nothing new today." in render_html(Digest(date="2026-10-27", reports=[]))
    one = Digest(date="2026-10-27", reports=[report(verdict="watch")])
    assert one.subject() == "Research Radar 2026-10-27: 1 new item"


def test_html_escapes_outside_text_and_only_links_http():
    evil = report(
        title='<script>alert(1)</script> & "quotes"',
        what="<img src=x onerror=alert(1)>",
        url="javascript:alert(1)",
    )
    page = render_html(Digest(date="2026-10-27", reports=[evil]))
    assert "<script>" not in page and "<img src=x" not in page
    assert "&lt;script&gt;" in page
    assert "javascript:" not in page
    good = render_html(Digest(date="2026-10-27", reports=[report(url="https://example.org/a?x=1&y=2")]))
    assert 'href="https://example.org/a?x=1&amp;y=2"' in good


def test_html_has_both_themes_and_a_readable_date():
    page = render_html(Digest(date="2026-10-27", reports=[report()]))
    assert "prefers-color-scheme: dark" in page
    assert "Tuesday, 27 October 2026" in page
    assert "<title>Research Radar 2026-10-27: 1 new item" in page


def test_rising_item_is_described_by_its_growth():
    rising = report(verdict="watch", rising=True, meta={"stars_before": 40, "stars_now": 300})
    text = render_text(Digest(date="2026-10-27", reports=[rising]))
    assert "Rising: from 40 to 300 stars" in text
    assert "RISING" in render_html(Digest(date="2026-10-27", reports=[rising]))


def test_telegram_messages_stay_under_the_limit_and_keep_items_whole():
    reports = [report(f"repo-{i}", what="word " * 60, why="reason " * 40, url=f"https://example.org/{i}") for i in range(40)]
    messages = render_telegram(Digest(date="2026-10-27", reports=reports))
    assert len(messages) > 1
    assert all(len(m) <= 4000 for m in messages)
    for m in messages:
        assert m.count("<a ") == m.count("</a>")
    assert sum(m.count("repo-") for m in messages) == 40


def test_telegram_escapes_markup_in_titles():
    messages = render_telegram(Digest(date="2026-10-27", reports=[report(title="a <b> & c")]))
    assert "a &lt;b&gt; &amp; c" in messages[0]


def test_latex_in_titles_is_shown_as_plain_text():
    assert plain_title("Real-World $\\times 10$ Cross-Sensor") == "Real-World x10 Cross-Sensor"
    assert plain_title("A $\\alpha$ test") == "A alpha test"


def test_no_em_dash_in_any_rendering():
    digest = Digest(date="2026-10-27", reports=[report(), report("p", "skip", kind="paper")], warnings=["w"])
    for output in (render_html(digest), render_text(digest), "".join(render_telegram(digest))):
        assert chr(0x2014) not in output


def test_the_prompt_never_offers_install_for_a_paper():
    seen: list[str] = []

    class Spy(FakeOllama):
        def chat_json(self, system, user):
            seen.append(user)
            return super().chat_json(system, user)

    topic = Topic(name="T", description="things")
    summarize(Spy(), Item("arxiv", "paper", "A paper", "https://x/1", "text"), topic, "text", "English")
    summarize(Spy(), Item("github", "repo", "a/b", "https://x/2", "text"), topic, "text", "English")
    assert "install =" not in seen[0] and "read =" in seen[0]
    assert "install =" in seen[1]
    assert "empty string" not in seen[0]  # the model is no longer invited to leave "why" blank


# delivery: email ---------------------------------------------------------------------------


class FakeSMTP:
    outbox: list = []
    fail_login = False
    fail_send = False

    def login(self, user, password):
        if FakeSMTP.fail_login:
            raise smtplib.SMTPAuthenticationError(535, b"nope")
        self.login_args = (user, password)

    def send_message(self, message):
        if FakeSMTP.fail_send:
            raise smtplib.SMTPServerDisconnected("gone")
        FakeSMTP.outbox.append(message)

    def quit(self):
        return None


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.outbox = []
    FakeSMTP.fail_login = False
    FakeSMTP.fail_send = False
    monkeypatch.setattr(deliver, "_open_smtp", lambda host, port, security: FakeSMTP())
    monkeypatch.setenv("RADAR_SMTP_PASSWORD", "app-password")
    return FakeSMTP


def email_cfg(make_cfg, **delivery):
    cfg = make_cfg()
    cfg.delivery["email"].update({"enabled": True, "username": "me@example.com", "to": "me@example.com"})
    cfg.delivery.update(delivery)
    return cfg


def run(cfg, http, now=NOW, ollama=None):
    return scan(cfg, http=http, ollama=ollama or FakeOllama(), now=now)


def test_scan_sends_one_mail_with_text_and_html(make_cfg, http_factory, smtp):
    cfg = email_cfg(make_cfg)
    result = run(cfg, http_factory())
    assert [d.channel for d in result.deliveries] == ["email"]
    assert result.deliveries[0].sent == len(result.new_reports) > 0
    assert len(smtp.outbox) == 1
    message = smtp.outbox[0]
    assert message["To"] == "me@example.com" and message["Subject"].startswith("Research Radar ")
    assert "alice/local-llm-kit" in message.get_body(("plain",)).get_content()
    assert "alice/local-llm-kit" in message.get_body(("html",)).get_content()
    assert result.digest_path and result.digest_path.exists()


def test_nothing_is_sent_twice(make_cfg, http_factory, smtp):
    cfg = email_cfg(make_cfg)
    run(cfg, http_factory())
    second = run(cfg, http_factory(), now=NOW + timedelta(hours=1))
    assert len(smtp.outbox) == 1
    assert second.deliveries[0].sent == 0 and second.deliveries[0].ok


def test_a_failed_send_is_retried_by_the_next_scan(make_cfg, http_factory, smtp):
    cfg = email_cfg(make_cfg)
    smtp.fail_send = True
    first = run(cfg, http_factory())
    assert first.new_reports and not first.deliveries[0].ok and smtp.outbox == []
    smtp.fail_send = False
    second = run(cfg, http_factory(), now=NOW + timedelta(hours=1))
    assert second.new_reports == []  # the items were already reported
    assert second.deliveries[0].sent == len(first.new_reports)
    assert len(smtp.outbox) == 1


def test_missing_password_is_a_clear_error_and_does_not_break_the_scan(make_cfg, http_factory, smtp, monkeypatch):
    monkeypatch.delenv("RADAR_SMTP_PASSWORD")
    result = run(email_cfg(make_cfg), http_factory())
    assert result.new_reports and result.note_path.exists()
    assert "RADAR_SMTP_PASSWORD is not set" in result.deliveries[0].error


def test_refused_login_points_at_app_passwords(make_cfg, http_factory, smtp):
    smtp.fail_login = True
    result = run(email_cfg(make_cfg), http_factory())
    assert "app password" in result.deliveries[0].error


def test_misconfigured_email_names_the_missing_setting(make_cfg, http_factory, smtp):
    cfg = make_cfg()
    cfg.delivery["email"].update({"enabled": True, "username": "me@example.com", "to": ""})
    assert "delivery.email.to" in run(cfg, http_factory()).deliveries[0].error


def test_send_when_empty_tells_you_once_a_day(make_cfg, http_factory, smtp):
    cfg = email_cfg(make_cfg, send_when_empty=True, html_file=False)
    quiet = FakeOllama()
    first = run(make_cfg_quiet(cfg), http_factory(), ollama=quiet)
    second = run(make_cfg_quiet(cfg), http_factory(), now=NOW + timedelta(hours=1), ollama=quiet)
    assert first.new_reports == [] and len(smtp.outbox) == 1
    assert smtp.outbox[0]["Subject"].endswith("nothing new")
    assert second.deliveries[0].note == "already told you there was nothing new"


def make_cfg_quiet(cfg):
    cfg.settings["min_score"] = 0.99
    return cfg


def test_no_mail_when_nothing_is_new_by_default(make_cfg, http_factory, smtp):
    cfg = email_cfg(make_cfg)
    cfg.settings["min_score"] = 0.99
    result = run(cfg, http_factory())
    assert smtp.outbox == [] and result.deliveries[0].note == "nothing new to send"


def test_resending_a_whole_day(make_cfg, http_factory, smtp):
    cfg = email_cfg(make_cfg)
    first = run(cfg, http_factory())
    store = Store(cfg.state_dir)
    again = deliver.deliver_pending(cfg, store, first.date, only_date=first.date)
    assert again[0].sent == len(first.all_reports) and len(smtp.outbox) == 2


def test_dry_run_sends_nothing_and_remembers_nothing(make_cfg, http_factory, smtp):
    cfg = email_cfg(make_cfg)
    smtp.fail_send = True
    first = run(cfg, http_factory())  # fails, so everything stays pending
    smtp.fail_send = False
    store = Store(cfg.state_dir)
    dry = deliver.deliver_pending(cfg, store, first.date, dry_run=True)
    assert dry[0].sent == len(first.new_reports) and smtp.outbox == []
    real = deliver.deliver_pending(cfg, store, first.date)
    assert real[0].sent == len(first.new_reports) and len(smtp.outbox) == 1


def test_secrets_never_reach_the_state_files(make_cfg, http_factory, smtp):
    cfg = email_cfg(make_cfg)
    run(cfg, http_factory())
    for path in cfg.home.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".html"}:
            assert "app-password" not in path.read_text(encoding="utf-8")


# delivery: telegram -------------------------------------------------------------------------


def telegram_cfg(make_cfg):
    cfg = make_cfg()
    cfg.delivery["telegram"].update({"enabled": True, "chat_id": "42"})
    return cfg


@pytest.fixture
def telegram(monkeypatch):
    calls: list[httpx.Request] = []
    state = {"status": 200, "body": {"ok": True, "result": {"username": "radar_bot"}}, "raise": False}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if state["raise"]:
            raise httpx.ConnectError(f"cannot connect to {request.url}")
        return httpx.Response(state["status"], json=state["body"])

    monkeypatch.setenv("RADAR_TELEGRAM_TOKEN", TOKEN)
    monkeypatch.setattr(deliver, "_telegram_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    return calls, state


def test_telegram_message_is_sent_in_html_mode(make_cfg, http_factory, telegram):
    calls, _ = telegram
    result = run(telegram_cfg(make_cfg), http_factory())
    assert result.deliveries[0].ok and result.deliveries[0].sent == len(result.new_reports)
    sent = [c for c in calls if c.url.path.endswith("/sendMessage")]
    body = json.loads(sent[0].content)
    assert body["chat_id"] == "42" and body["parse_mode"] == "HTML" and body["disable_web_page_preview"] is True
    assert "alice/local-llm-kit" in body["text"]
    assert sent[0].url.path == f"/bot{TOKEN}/sendMessage"


def test_telegram_errors_do_not_leak_the_token(make_cfg, http_factory, telegram):
    calls, state = telegram
    state["status"], state["body"] = 401, {"ok": False, "description": "Unauthorized"}
    refused = run(telegram_cfg(make_cfg), http_factory())
    assert "Unauthorized" in refused.deliveries[0].error and TOKEN not in refused.deliveries[0].error

    state["raise"] = True
    cfg = telegram_cfg(make_cfg)
    unreachable = run(cfg, http_factory(), now=NOW + timedelta(hours=1))
    error = unreachable.deliveries[0].error
    assert "cannot reach Telegram" in error and TOKEN not in error and "SECRET" not in error


def test_telegram_needs_token_and_chat_id(make_cfg, http_factory, telegram, monkeypatch):
    cfg = make_cfg()
    cfg.delivery["telegram"]["enabled"] = True
    assert "chat_id" in run(cfg, http_factory()).deliveries[0].error
    monkeypatch.delenv("RADAR_TELEGRAM_TOKEN")
    cfg.delivery["telegram"]["chat_id"] = "42"
    assert "RADAR_TELEGRAM_TOKEN is not set" in run(cfg, http_factory(), now=NOW + timedelta(hours=1)).deliveries[0].error


def test_two_channels_keep_separate_books(make_cfg, http_factory, smtp, telegram):
    _, state = telegram
    cfg = email_cfg(make_cfg)
    cfg.delivery["telegram"].update({"enabled": True, "chat_id": "42"})
    state["status"], state["body"] = 500, {"ok": False, "description": "down"}
    first = run(cfg, http_factory())
    assert {d.channel: d.ok for d in first.deliveries} == {"email": True, "telegram": False}
    state["status"], state["body"] = 200, {"ok": True, "result": {}}
    second = run(cfg, http_factory(), now=NOW + timedelta(hours=1))
    by_channel = {d.channel: d for d in second.deliveries}
    assert by_channel["email"].sent == 0 and by_channel["telegram"].sent == len(first.new_reports)


def test_finding_the_chat_id(make_cfg, telegram):
    _, state = telegram
    state["body"] = {
        "ok": True,
        "result": [
            {"message": {"chat": {"id": 42, "first_name": "Arca", "last_name": "A"}}},
            {"message": {"chat": {"id": 42, "first_name": "Arca"}}},
            {"my_chat_member": {"chat": {"id": -100, "title": "Radar group"}}},
        ],
    }
    assert deliver.find_telegram_chats(make_cfg()) == [("42", "Arca"), ("-100", "Radar group")]


def test_test_message_goes_through_each_enabled_channel(make_cfg, smtp, telegram):
    calls, _ = telegram
    cfg = email_cfg(make_cfg)
    cfg.delivery["telegram"].update({"enabled": True, "chat_id": "42"})
    results = deliver.send_test(cfg)
    assert [r.ok for r in results] == [True, True]
    assert smtp.outbox[0]["Subject"] == "Research Radar test message"
    assert any("test message" in json.loads(c.content).get("text", "") for c in calls)


def test_doctor_checks_use_the_real_login_and_bot_lookup(make_cfg, smtp, telegram):
    cfg = email_cfg(make_cfg)
    cfg.delivery["telegram"].update({"enabled": True, "chat_id": "42"})
    assert "login ok" in deliver.check_email(cfg)
    assert "@radar_bot" in deliver.check_telegram(cfg)
    smtp.fail_login = True
    with pytest.raises(deliver.DeliveryError):
        deliver.check_email(cfg)


# config -----------------------------------------------------------------------------------------


def test_delivery_defaults_and_shorthand():
    data = yaml.safe_load(example_text())
    data.pop("delivery")
    cfg = parse_config(data, __import__("pathlib").Path("x"))
    assert cfg.delivery["html_file"] is True and not cfg.channel_enabled("email")
    data["delivery"] = {"telegram": True}
    assert parse_config(data, __import__("pathlib").Path("x")).channel_enabled("telegram")


def test_unknown_delivery_options_are_rejected():
    data = yaml.safe_load(example_text())
    data["delivery"] = {"emial": {"enabled": True}}
    with pytest.raises(ConfigError, match="emial"):
        parse_config(data, __import__("pathlib").Path("x"))
    data["delivery"] = {"email": {"enabled": True, "smtp_hots": "x"}}
    with pytest.raises(ConfigError, match="smtp_hots"):
        parse_config(data, __import__("pathlib").Path("x"))


# command line and interface ---------------------------------------------------------------------


def seed_home(tmp_path, channel_enabled=True):
    home = tmp_path / "h"
    path = init_home(home)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["delivery"]["telegram"].update({"enabled": channel_enabled, "chat_id": "42"})
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    today = datetime.now().astimezone().date().isoformat()
    day = report(title="alice/tool", verdict="read")
    day.date = today
    Store(home / "state").save_day(today, [day], [], [], "embeddings")
    return home, today


def test_send_without_a_channel_explains_what_to_do(tmp_path, capsys):
    home, _ = seed_home(tmp_path, channel_enabled=False)
    assert cli.main(["--home", str(home), "send"]) == 1
    assert "delivery section" in capsys.readouterr().out


def test_send_dry_run_writes_the_page_and_sends_nothing(tmp_path, capsys, monkeypatch):
    home, today = seed_home(tmp_path)
    monkeypatch.setattr(deliver, "_telegram_client", lambda: (_ for _ in ()).throw(AssertionError("must not send")))
    assert cli.main(["--home", str(home), "send", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run, would send 1 item" in out and (home / "digests" / f"{today}.html").exists()
    assert Store(home / "state").delivered() == {}


def test_send_delivers_pending_items_and_a_second_call_has_nothing_left(tmp_path, capsys, telegram, monkeypatch):
    home, _ = seed_home(tmp_path)
    assert cli.main(["--home", str(home), "send"]) == 0
    assert "telegram: sent 1 item" in capsys.readouterr().out
    assert cli.main(["--home", str(home), "send"]) == 0
    assert "nothing new to send" in capsys.readouterr().out


def test_telegram_chat_command(tmp_path, capsys, telegram):
    _, state = telegram
    home, _ = seed_home(tmp_path)
    state["body"] = {"ok": True, "result": [{"message": {"chat": {"id": 42, "first_name": "Arca"}}}]}
    assert cli.main(["--home", str(home), "telegram-chat"]) == 0
    assert "42  Arca" in capsys.readouterr().out
    state["body"] = {"ok": True, "result": []}
    assert cli.main(["--home", str(home), "telegram-chat"]) == 1


async def test_tui_digest_key_writes_and_opens_the_page(make_cfg, monkeypatch):
    from research_radar.tui import RadarApp

    opened: list[str] = []
    monkeypatch.setattr("research_radar.tui.webbrowser.open", lambda url: opened.append(url))
    cfg = make_cfg(auto_scan_on_launch=False)
    app = RadarApp(cfg, auto_scan=False)
    async with app.run_test() as pilot:
        await pilot.press("d")
        await pilot.pause()
    assert len(opened) == 1 and opened[0].startswith("file:") and opened[0].endswith(".html")
