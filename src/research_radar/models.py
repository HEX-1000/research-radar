"""Plain data types shared by the sources, the ranker and the UI."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

VERDICTS = ("install", "read", "watch", "skip")

# Path parts of huggingface.co that are not an organisation or user, so not a model.
_HF_NOT_MODELS = {"papers", "datasets", "spaces", "docs", "blog", "api", "collections", "models", "organizations"}


def _arxiv_id(value: str) -> str:
    """An arXiv id without its version: 2610.12345v2 becomes 2610.12345."""
    return re.sub(r"v\d+$", "", value.strip().removesuffix(".pdf"))


def key_from_url(url: str, source: str | None = None) -> str:
    """One key per item across sources, from the URL a source gave for it.

    An arXiv paper is the same item on arXiv and on Hugging Face papers. Repository and model
    names are compared in lower case, as GitHub and Hugging Face do. Forgejo servers can live
    on any host, so a Forgejo URL needs its source to be told apart from an ordinary link."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = [p for p in parts.path.split("/") if p]
    if host == "github.com" and len(path) >= 2:
        return f"github:{path[0]}/{path[1]}".lower()
    if host in ("arxiv.org", "export.arxiv.org") and len(path) >= 2 and path[0] in ("abs", "pdf"):
        return "arxiv:" + _arxiv_id("/".join(path[1:]))  # old ids have a slash: hep-th/9901001
    if host == "huggingface.co" and path:
        if path[0] == "papers" and len(path) >= 2:
            return "arxiv:" + _arxiv_id(path[1])
        if path[0] not in _HF_NOT_MODELS:
            return "hf:" + "/".join(path[:2]).lower()
    if source == "forgejo" and host and len(path) >= 2:
        return f"forgejo:{host}/{path[0]}/{path[1]}".lower()
    return f"url:{url}"


def plain_title(title: str) -> str:
    """Titles from arXiv sometimes carry LaTeX. Show it as plain text."""
    text = re.sub(r"\\times\s*", "x", title)
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = text.replace("$", "")
    return " ".join(text.split())


@dataclass
class Item:
    """One thing a source found: a repository, a paper or a model."""

    source: str  # github, arxiv, huggingface-papers, huggingface-models, forgejo, feeds
    kind: str  # repo, paper, model, release (a watched feed) or article (a feed's post)
    title: str
    url: str
    text: str = ""
    published: str | None = None  # ISO date or datetime when known
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.url

    @property
    def key(self) -> str:
        """The item's key in the candidate store: the same paper from arXiv and from Hugging Face
        has one key, see key_from_url. An entry of a feed has the key of its own id: the URL of a
        release such as github.com/owner/name/releases/tag/v1 would otherwise give the key of the
        repository, and every later release would count as already shown."""
        entry_id = self.meta.get("entry_id")
        if self.source == "feeds" and isinstance(entry_id, str) and entry_id.strip():
            return "feed:" + entry_id.strip()
        arxiv_id = self.meta.get("arxiv_id") if self.kind == "paper" else None
        if isinstance(arxiv_id, str) and arxiv_id.strip():
            return "arxiv:" + _arxiv_id(arxiv_id)
        return key_from_url(self.url, self.source)


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
    ranking: str = ""  # how the scan chose it: embeddings or keywords; empty for rising items and older reports

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
