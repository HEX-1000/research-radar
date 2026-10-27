"""Source registry. Each source is a function: FetchContext in, list of Item out."""

from __future__ import annotations

from typing import Callable

from ..models import Item
from . import arxiv, forgejo, github, huggingface
from .base import FetchContext

SOURCES: dict[str, Callable[[FetchContext], list[Item]]] = {
    "github": github.fetch,
    "arxiv": arxiv.fetch,
    "huggingface-papers": huggingface.fetch_papers,
    "huggingface-models": huggingface.fetch_models,
    "forgejo": forgejo.fetch,
}

__all__ = ["SOURCES", "FetchContext"]
