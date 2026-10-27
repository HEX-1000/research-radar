"""Plain data types shared by the sources, the ranker and the UI."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

VERDICTS = ("install", "read", "watch", "skip")


def plain_title(title: str) -> str:
    """Titles from arXiv sometimes carry LaTeX. Show it as plain text."""
    text = re.sub(r"\\times\s*", "x", title)
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = text.replace("$", "")
    return " ".join(text.split())


@dataclass
class Item:
    """One thing a source found: a repository, a paper or a model."""

    source: str  # github, arxiv, huggingface-papers, huggingface-models, forgejo
    kind: str  # repo, paper or model
    title: str
    url: str
    text: str = ""
    published: str | None = None  # ISO date or datetime when known
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.url


@dataclass
class Report:
    """An item that was picked for a day's note, with its summary."""

    date: str
    topic: str
    score: float
    source: str
    kind: str
    title: str
    url: str
    what: str = ""
    why: str = ""
    verdict: str = "read"
    meta: dict[str, Any] = field(default_factory=dict)
    rising: bool = False
    vote: int = 0  # +1, -1 or 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Report":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


@dataclass
class SkippedSource:
    source: str
    reason: str
