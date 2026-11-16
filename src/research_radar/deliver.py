"""Getting the digest to the person: an HTML file, an email, a Telegram message.

Secrets (the SMTP password, the Telegram bot token) are read from environment
variables named in interests.yaml. They are never written to a file here and
never put into an error message.

A scan never fails because a message could not be sent. Each channel remembers
which items it has already delivered, so a failed send is retried by the next
scan and a successful one is never repeated.
"""

from __future__ import annotations

import os
import re
import smtplib
import ssl
import time
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any

import httpx

from . import __version__
from .candidates import CandidateStore, StoreError
from .config import Config
from .digest import (
    Digest,
    TelegramMessage,
    keyboard_for_hash,
    render_html,
    render_telegram,
    render_text,
    vote_hash,
)
from .models import Report
from .store import Store

CHANNELS = ("email", "telegram")
TELEGRAM_API = "https://api.telegram.org"
TELEGRAM_PAUSE = 1.0  # seconds between two messages of one digest
VOTE_DATA = re.compile(r"^v1([+-])([0-9a-f]{16})$")
UPDATE_PAGE = 100
UPDATE_CALLS = 10


class DeliveryError(Exception):
    """A channel is misconfigured or the message could not be sent."""


@dataclass
class DeliveryResult:
    channel: str
    sent: int = 0  # items delivered
    error: str = ""
    note: str = ""  # why nothing was sent, when that is not an error
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return not self.error


def describe(results: list[DeliveryResult]) -> list[str]:
    lines = []
    for r in results:
        if r.error:
            lines.append(f"{r.channel}: not sent, {r.error}")
        elif r.dry_run:
            lines.append(f"{r.channel}: dry run, would send {r.sent} item{'s' if r.sent != 1 else ''}")
        elif r.sent:
            lines.append(f"{r.channel}: sent {r.sent} item{'s' if r.sent != 1 else ''}")
        else:
            lines.append(f"{r.channel}: {r.note or 'nothing to send'}")
    return lines


# which reports still need sending -------------------------------------------------------


def delivery_key(report: Report) -> str:
    return f"{report.date}|{report.url}|{int(report.rising)}"


def _empty_key(date: str) -> str:
    return f"{date}|empty"


def _alert_key(date: str) -> str:
    return f"{date}|alert"


def pending_reports(cfg: Config, store: Store, channel: str, today: str) -> list[Report]:
    """Reports from the last few days that this channel has not delivered yet."""
    done = set(store.delivered().get(channel, []))
    try:
        horizon = (date_type.fromisoformat(today) - timedelta(days=int(cfg.delivery["retry_days"]))).isoformat()
    except ValueError:
        horizon = ""
    reports: list[Report] = []
    for date in store.report_dates():
        if date < horizon or date > today:
            continue
        reports.extend(r for r in store.load_day(date)["reports"] if delivery_key(r) not in done)
    return reports


def build_digest(cfg: Config, store: Store, date: str, reports: list[Report]) -> Digest:
    day = store.load_day(date)
    return Digest(
        date=date,
        reports=reports,
        topic_order=[t.name for t in cfg.topics],
        skipped=day["skipped"],
        warnings=day["warnings"],
        ranking=day["ranking"] or "embeddings",
        alert=cfg.alert,
    )


def write_digest_file(cfg: Config, date: str, *, store: Store | None = None) -> Path:
    """The whole day as one HTML page, whether or not anything was sent."""
    store = store or Store(cfg.state_dir)
    digest = build_digest(cfg, store, date, store.load_day(date)["reports"])
    path = cfg.digests_dir / f"{date}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(digest), encoding="utf-8")
    return path


# email ----------------------------------------------------------------------------------


def _addresses(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.replace(";", ",").split(",")
    return [str(v).strip() for v in (value or []) if str(v).strip()]


def email_settings(cfg: Config) -> dict[str, Any]:
    opts = cfg.delivery["email"]
    username = str(opts.get("username") or "").strip()
    recipients = _addresses(opts.get("to"))
    sender = str(opts.get("from") or username).strip()
    if not str(opts.get("smtp_host") or "").strip():
        raise DeliveryError("delivery.email.smtp_host is empty")
    if not recipients:
        raise DeliveryError("delivery.email.to is empty")
    if not sender:
        raise DeliveryError("set delivery.email.username (or delivery.email.from)")
    if any("@" not in a for a in recipients + [sender]):
        raise DeliveryError("delivery.email addresses must contain an @")
    security = str(opts.get("security") or "starttls").lower()
    if security not in ("starttls", "ssl", "none"):
        raise DeliveryError("delivery.email.security must be starttls, ssl or none")
    env = str(opts.get("password_env") or "")
    password = os.environ.get(env, "") if env else ""
    if username and not password:
        raise DeliveryError(f"the environment variable {env or '(password_env)'} is not set")
    return {
        "host": str(opts["smtp_host"]).strip(),
        "port": int(opts.get("smtp_port") or 587),
        "security": security,
        "username": username,
        "password": password,
        "sender": sender,
        "recipients": recipients,
    }


def _open_smtp(host: str, port: int, security: str) -> smtplib.SMTP:
    context = ssl.create_default_context()
    if security == "ssl":
        return smtplib.SMTP_SSL(host, port, timeout=30, context=context)
    server = smtplib.SMTP(host, port, timeout=30)
    if security == "starttls":
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
    return server


def _smtp_session(settings: dict[str, Any]):
    try:
        server = _open_smtp(settings["host"], settings["port"], settings["security"])
        if settings["username"]:
            server.login(settings["username"], settings["password"])
    except smtplib.SMTPAuthenticationError as exc:
        raise DeliveryError(
            "the mail server refused the login. For Gmail, use an app password, not the account password"
        ) from exc
    except (smtplib.SMTPException, OSError) as exc:
        raise DeliveryError(f"cannot use the mail server {settings['host']}:{settings['port']} ({type(exc).__name__})") from exc
    return server


def build_email(settings: dict[str, Any], subject: str, text: str, page: str) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings["sender"]
    message["To"] = ", ".join(settings["recipients"])
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="research-radar.local")
    message.set_content(text)
    message.add_alternative(page, subtype="html")
    return message


def send_email(cfg: Config, subject: str, text: str, page: str) -> None:
    settings = email_settings(cfg)
    message = build_email(settings, subject, text, page)
    server = _smtp_session(settings)
    try:
        server.send_message(message)
    except (smtplib.SMTPException, OSError) as exc:
        raise DeliveryError(f"the mail server did not accept the message ({type(exc).__name__})") from exc
    finally:
        try:
            server.quit()
        except (smtplib.SMTPException, OSError):
            pass


def mask_address(address: str) -> str:
    """a***@example.org: the first character, `***`, then the domain."""
    address = address.strip()
    local, at, domain = address.partition("@")
    return f"{local[:1]}***{at}{domain}"


def mask_chat(chat: str) -> str:
    """...123: the last three characters; a shorter id is hidden."""
    chat = chat.strip()
    return f"...{chat[-3:]}" if len(chat) > 3 else "***"


def check_email(cfg: Config) -> str:
    settings = email_settings(cfg)
    server = _smtp_session(settings)
    try:
        server.quit()
    except (smtplib.SMTPException, OSError):
        pass
    return f"login ok on {settings['host']}, mail goes to {', '.join(mask_address(r) for r in settings['recipients'])}"


# telegram -------------------------------------------------------------------------------


def _telegram_client() -> httpx.Client:
    return httpx.Client(timeout=30.0, headers={"User-Agent": f"research-radar/{__version__}"})


def telegram_token(cfg: Config) -> str:
    env = str(cfg.delivery["telegram"].get("token_env") or "")
    token = os.environ.get(env, "") if env else ""
    if not token:
        raise DeliveryError(f"the environment variable {env or '(token_env)'} is not set")
    return token


def _telegram_call(token: str, method: str, payload: dict[str, Any] | None = None) -> Any:
    # Errors never include the request URL, because it contains the bot token.
    client = _telegram_client()
    try:
        response = client.post(f"{TELEGRAM_API}/bot{token}/{method}", json=payload or {})
    except httpx.HTTPError as exc:
        raise DeliveryError(f"cannot reach Telegram ({type(exc).__name__})") from exc
    finally:
        client.close()
    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.status_code != 200 or not data.get("ok"):
        reason = str(data.get("description") or f"HTTP {response.status_code}")
        raise DeliveryError(f"Telegram refused the request: {reason}")
    return data.get("result")


def telegram_chat_id(cfg: Config) -> str:
    chat = str(cfg.delivery["telegram"].get("chat_id") or "").strip()
    if not chat:
        raise DeliveryError("delivery.telegram.chat_id is empty (run `radar telegram-chat` to find it)")
    return chat


def _send_message(token: str, chat: str, message: TelegramMessage) -> None:
    payload: dict[str, Any] = {
        "chat_id": chat,
        "text": message.text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if not message.sound:
        payload["disable_notification"] = True
    if message.keyboard:
        payload["reply_markup"] = {"inline_keyboard": message.keyboard}
    _telegram_call(token, "sendMessage", payload)


def send_telegram(cfg: Config, messages: list[TelegramMessage], on_sent: Any = None) -> None:
    """Send the messages in order, at least TELEGRAM_PAUSE apart. `on_sent(message)` runs after
    each one, so the caller can record what was delivered before a later message fails."""
    token = telegram_token(cfg)
    chat = telegram_chat_id(cfg)
    for index, message in enumerate(messages):
        if index:
            time.sleep(TELEGRAM_PAUSE)
        _send_message(token, chat, message)
        if on_sent:
            on_sent(message)


def check_telegram(cfg: Config) -> str:
    token = telegram_token(cfg)
    chat = telegram_chat_id(cfg)
    bot = _telegram_call(token, "getMe") or {}
    name = str(bot.get("username") or "")
    shown = f"@{name[:3]}***" if name else "@?"
    return f"bot {shown}, messages go to chat {mask_chat(chat)}"


def find_telegram_chats(cfg: Config) -> list[tuple[str, str]]:
    """Chats that have written to the bot recently: (chat id, readable name)."""
    updates = _telegram_call(telegram_token(cfg), "getUpdates") or []
    found: dict[str, str] = {}
    for update in updates:
        message = update.get("message") or update.get("channel_post") or update.get("my_chat_member") or {}
        chat = message.get("chat") or {}
        if "id" in chat:
            name = chat.get("title") or " ".join(
                p for p in (chat.get("first_name"), chat.get("last_name")) if p
            ) or chat.get("username") or "unnamed"
            found[str(chat["id"])] = name
    return list(found.items())


# votes -----------------------------------------------------------------------------------


def collect_telegram_votes(cfg: Config, candidates: CandidateStore, now: datetime) -> tuple[int, str]:
    """Record the votes pressed on the buttons of earlier digests: (votes written, a warning or "").

    Run under the scan lock with a store opened for writing and already on disk. Nothing here
    raises: what cannot be read now is read again by the next run.

    Pages of `getUpdates` are committed one by one: the call that asks for the next page (or the
    final call with `limit` 1) carries the `offset` that confirms the page before it, so a page is
    written to the store before that call is made. A committed vote is counted and gets its check
    mark also when that call fails; the next run then writes it again. Without `offset` and
    `allowed_updates`, so that `radar telegram-chat` still sees its messages."""
    if not candidates.write or candidates.create or candidates.schema < 2:
        return 0, ""
    try:
        token = telegram_token(cfg)
        chat = telegram_chat_id(cfg)
    except DeliveryError:
        return 0, ""  # delivery reports a missing token or chat itself
    pressed: dict[str, tuple[int, str, int]] = {}  # item key -> (vote, button hash, message id)
    warning = ""
    try:
        urls: dict[str, str] = {}  # hash of a link URL -> item key
        for key, url in candidates.conn.execute(
            "SELECT item.key, link.url FROM item JOIN link ON link.key = item.key"
            " WHERE item.state IN ('shown', 'gone')"
        ):
            urls[vote_hash(url)] = key
        payload: dict[str, Any] = {"limit": UPDATE_PAGE}
        for pages in range(1, UPDATE_CALLS + 1):
            page = _telegram_call(token, "getUpdates", payload) or []
            if not page:
                break  # an empty page after an offset: the pages before it are confirmed
            on_page: dict[str, tuple[int, str, int]] = {}
            for update in sorted(page, key=lambda u: int(u.get("update_id", 0))):
                query = update.get("callback_query") or {}
                message = query.get("message") or {}
                if str((message.get("chat") or {}).get("id", "")) != chat:
                    continue
                match = VOTE_DATA.match(str(query.get("data") or ""))
                key = urls.get(match.group(2)) if match else None
                if not match or not key or "message_id" not in message:
                    continue
                on_page[key] = (1 if match.group(1) == "+" else -1, match.group(2), int(message["message_id"]))
            if on_page:
                candidates.set_votes({k: v[0] for k, v in on_page.items()}, "telegram", now)
                pressed.update(on_page)  # a later page writes over an earlier one
            offset = max(int(u["update_id"]) for u in page) + 1
            if len(page) == UPDATE_PAGE and pages < UPDATE_CALLS:
                payload = {"limit": UPDATE_PAGE, "offset": offset}  # confirms this page, reads the next
                continue
            # confirms the last page; when it fails, the next run reads that page again
            _telegram_call(token, "getUpdates", {"offset": offset, "limit": 1})
            break
    except (DeliveryError, StoreError) as exc:
        warning = f"Could not collect the Telegram votes: {exc}"
    except Exception as exc:  # a scan must survive anything the collecting code throws
        warning = f"Could not collect the Telegram votes ({type(exc).__name__})."
    # the count and the edits cover every committed page, also when a later call failed (decision 24)
    for vote, h, message_id in pressed.values():
        try:
            _telegram_call(
                token,
                "editMessageReplyMarkup",
                {
                    "chat_id": chat,
                    "message_id": message_id,
                    "reply_markup": {"inline_keyboard": keyboard_for_hash(h, vote)},
                },
            )
        except Exception:
            pass  # a failed edit is ignored
    return len(pressed), warning


# sending ---------------------------------------------------------------------------------


def deliver_pending(
    cfg: Config,
    store: Store,
    today: str,
    *,
    only_date: str | None = None,
    dry_run: bool = False,
) -> list[DeliveryResult]:
    """Send what each enabled channel has not delivered yet.

    With only_date, resend that whole day to every enabled channel.
    With dry_run, change nothing and send nothing; report what would go out.
    """
    results: list[DeliveryResult] = []
    for channel in CHANNELS:
        if not cfg.channel_enabled(channel):
            continue
        result = DeliveryResult(channel)
        results.append(result)
        try:
            if only_date:
                reports = store.load_day(only_date)["reports"]
                date = only_date
            else:
                reports = pending_reports(cfg, store, channel, today)
                date = today
            delivered = store.delivered().get(channel, [])
            empty_note = False
            # A broken interests.yaml is reported once a day per channel, even when nothing is new.
            alert_due = bool(cfg.alert) and not only_date and _alert_key(today) not in delivered
            if not reports and not alert_due:
                if only_date or not cfg.delivery["send_when_empty"]:
                    result.note = "nothing new to send"
                    continue
                if _empty_key(today) in delivered:
                    result.note = "already told you there was nothing new"
                    continue
                empty_note = True
            if dry_run:
                result.sent = len(reports)
                result.dry_run = True
                continue
            digest = build_digest(cfg, store, date, reports)
            extra = ([_empty_key(today)] if empty_note else []) + ([_alert_key(today)] if alert_due else [])
            if channel == "email":
                send_email(cfg, digest.subject(), render_text(digest), render_html(digest))
                store.mark_delivered(channel, [delivery_key(r) for r in reports] + extra)
                result.sent = len(reports)
            else:

                def sent(message: TelegramMessage) -> None:
                    # each item is marked as soon as its message is out; the header carries the day marks
                    keys = [delivery_key(r) for r in message.reports] + (extra if message.sound else [])
                    store.mark_delivered(channel, keys)
                    result.sent += len(message.reports)

                send_telegram(cfg, render_telegram(digest), sent)
            if not reports:
                result.note = "sent the interests.yaml warning"
        except DeliveryError as exc:
            result.error = str(exc)
        except Exception as exc:  # a scan must survive anything the sending code throws
            result.error = f"unexpected failure ({type(exc).__name__})"
    return results


def send_test(cfg: Config) -> list[DeliveryResult]:
    """A short message through every enabled channel, to check the setup end to end."""
    results: list[DeliveryResult] = []
    text = "This is a test message from Research Radar. If you can read it, delivery works."
    for channel in CHANNELS:
        if not cfg.channel_enabled(channel):
            continue
        result = DeliveryResult(channel)
        results.append(result)
        try:
            if channel == "email":
                send_email(cfg, "Research Radar test message", text + "\n", f"<p>{text}</p>")
            else:
                send_telegram(cfg, [TelegramMessage(text, sound=True)])
            result.note = "test message sent"
        except DeliveryError as exc:
            result.error = str(exc)
    return results
