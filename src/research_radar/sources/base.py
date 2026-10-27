"""What every source module gets and what it must do."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..config import Config
from ..http import Http, SourceError


@dataclass
class FetchContext:
    config: Config
    http: Http
    since: datetime  # UTC, timezone aware
    until: datetime
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def settle(ctx: FetchContext, name: str, ok_requests: int, errors: list[str]) -> None:
    """Decide whether a source failed or only had a few bad requests.

    If no request worked, the source failed and the scan reports it as skipped.
    If some worked, the bad ones become warnings and the items we got are kept.
    """
    if errors and ok_requests == 0:
        raise SourceError(errors[0])
    for message in errors:
        ctx.warn(f"{name}: {message}")


def clean_text(value: object) -> str:
    """Collapse whitespace. Sources return text with line breaks and padding."""
    return " ".join(str(value or "").split())
