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
import smtplib
import ssl
from dataclasses import dataclass
from datetime import date as date_type
from datetime import timedelta
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any

import httpx

from . import __version__
from .config import Config
from .digest import Digest, render_html, render_telegram, render_text
from .models import Report
from .store import Store

CHANNELS = ("email", "telegram")
TELEGRAM_API = "https://api.telegram.org"


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


def check_email(cfg: Config) -> str:
    settings = email_settings(cfg)
    server = _smtp_session(settings)
    try:
        server.quit()
    except (smtplib.SMTPException, OSError):
        pass
    return f"login ok on {settings['host']}, mail goes to {', '.join(settings['recipients'])}"


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


def send_telegram(cfg: Config, messages: list[str]) -> None:
    token = telegram_token(cfg)
    chat = telegram_chat_id(cfg)
    for text in messages:
        _telegram_call(
            token,
            "sendMessage",
            {"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
        )


def check_telegram(cfg: Config) -> str:
    token = telegram_token(cfg)
    chat = telegram_chat_id(cfg)
    bot = _telegram_call(token, "getMe") or {}
    return f"bot @{bot.get('username', '?')}, messages go to chat {chat}"


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


# sending ---------------------------------------------------------------------------------


def _send(cfg: Config, channel: str, digest: Digest) -> None:
    if channel == "email":
        send_email(cfg, digest.subject(), render_text(digest), render_html(digest))
    else:
        send_telegram(cfg, render_telegram(digest))


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
            empty_note = False
            if not reports:
                if only_date or not cfg.delivery["send_when_empty"]:
                    result.note = "nothing new to send"
                    continue
                if _empty_key(today) in store.delivered().get(channel, []):
                    result.note = "already told you there was nothing new"
                    continue
                empty_note = True
            if dry_run:
                result.sent = len(reports)
                result.dry_run = True
                continue
            _send(cfg, channel, build_digest(cfg, store, date, reports))
            store.mark_delivered(channel, [delivery_key(r) for r in reports] + ([_empty_key(today)] if empty_note else []))
            result.sent = len(reports)
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
                send_telegram(cfg, [text])
            result.note = "test message sent"
        except DeliveryError as exc:
            result.error = str(exc)
    return results
