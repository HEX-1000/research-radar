"""Shared fixtures: fake sources behind httpx.MockTransport and a fake Ollama."""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import httpx
import pytest
import yaml

from research_radar.config import Config, example_text, parse_config
from research_radar.http import Http
from research_radar.llm import OllamaError

GITHUB_SEARCH = {
    "total_count": 2,
    "items": [
        {
            "full_name": "alice/local-llm-kit",
            "html_url": "https://github.com/alice/local-llm-kit",
            "description": "Run a local llm with embeddings and retrieval, no cloud needed",
            "stargazers_count": 120,
            "language": "Python",
            "created_at": "2026-10-26T08:00:00Z",
            "pushed_at": "2026-10-27T08:00:00Z",
            "topics": ["ollama", "retrieval"],
            "license": {"spdx_id": "MIT"},
        },
        {
            "full_name": "bob/cat-photo-app",
            "html_url": "https://github.com/bob/cat-photo-app",
            "description": "Share pictures of cats with friends",
            "stargazers_count": 40,
            "language": "Kotlin",
            "created_at": "2026-10-26T09:00:00Z",
            "topics": [],
            "license": None,
        },
    ],
}

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2610.12345v2</id>
    <published>2026-10-26T17:00:00Z</published>
    <title>Retrieval with
      small local embedding models</title>
    <summary>We study retrieval and context compression using embeddings that run locally.</summary>
    <author><name>A. Author</name></author>
    <category term="cs.IR"/>
  </entry>
</feed>
"""

HF_PAPERS = [
    {
        "paper": {
            "id": "2610.55555",
            "title": "Segmenting satellite images of terrain",
            "summary": "A remote sensing method for segmentation of sentinel-2 imagery.",
            "publishedAt": "2026-10-26T10:00:00.000Z",
            "upvotes": 12,
        }
    }
]

HF_MODELS = [
    {
        "id": "org/terrain-segmenter",
        "pipeline_tag": "image-segmentation",
        "tags": ["remote-sensing", "region:us", "license:apache-2.0"],
        "likes": 30,
        "downloads": 1000,
        "createdAt": "2026-10-20T00:00:00.000Z",
    }
]

FORGEJO = {
    "ok": True,
    "data": [
        {
            "full_name": "carol/stac-tools",
            "html_url": "https://codeberg.org/carol/stac-tools",
            "description": "Small tools for STAC catalogs and satellite imagery",
            "stars_count": 8,
            "created_at": "2026-10-26T12:00:00Z",
            "language": "Python",
            "fork": False,
            "archived": False,
        },
        {
            "full_name": "old/ancient-repo",
            "html_url": "https://codeberg.org/old/ancient-repo",
            "description": "created long before the window",
            "stars_count": 99,
            "created_at": "2020-01-01T00:00:00Z",
            "fork": False,
            "archived": False,
        },
    ],
}


def _bow(text: str, dims: int = 256) -> list[float]:
    vec = [0.0] * dims
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[zlib.crc32(word.encode()) % dims] += 1.0
    return vec


class FakeOllama:
    """Deterministic stand-in for Ollama: a hashed bag-of-words embedder."""

    def __init__(self, up: bool = True, chat_answer: dict | None = None) -> None:
        self.up = up
        self.chat_model = "fake-chat"
        self.chat_calls = 0
        self.chat_answer = chat_answer or {
            "what": "A toolkit for running models locally.",
            "why": "It matches the local-first interest.",
            "verdict": "install",
        }

    def ping(self) -> bool:
        return self.up

    def close(self) -> None:
        return None

    def resolve_chat_model(self) -> str:
        return self.chat_model

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        if not self.up:
            raise OllamaError("down")
        return [_bow(t) for t in texts]

    def chat_json(self, system: str, user: str):
        if not self.up:
            raise OllamaError("down")
        self.chat_calls += 1
        return self.chat_answer


def make_handler(calls: list[str] | None = None, fail: set[str] | None = None, repo_stars: int = 200):
    fail = fail or set()

    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if calls is not None:
            calls.append(f"{host}{path}")
        if host in fail:
            return httpx.Response(503, text="down")
        if host == "api.github.com" and path == "/search/repositories":
            return httpx.Response(200, json=GITHUB_SEARCH)
        if host == "api.github.com" and path.endswith("/readme"):
            return httpx.Response(200, text="# Tool\n![badge](x.png)\nA tool to run a local llm.")
        if host == "api.github.com" and path.startswith("/repos/"):
            return httpx.Response(200, json={"stargazers_count": repo_stars})
        if host == "export.arxiv.org":
            return httpx.Response(200, text=ARXIV_ATOM)
        if host == "huggingface.co" and path == "/api/daily_papers":
            return httpx.Response(200, json=HF_PAPERS)
        if host == "huggingface.co" and path == "/api/models":
            return httpx.Response(200, json=HF_MODELS)
        if host == "codeberg.org":
            return httpx.Response(200, json=FORGEJO)
        return httpx.Response(404, text="unexpected " + str(request.url))

    return handler


@pytest.fixture(autouse=True)
def ollama_has_every_model(monkeypatch):
    """The interface asks Ollama for its models on start. Tests never reach a real Ollama."""
    from research_radar import tui

    monkeypatch.setattr(tui, "default_probe", lambda cfg: ["nomic-embed-text:latest", "fake-chat:latest"])


@pytest.fixture
def http_factory():
    clients: list[Http] = []

    def build(**kwargs) -> Http:
        client = Http(
            transport=httpx.MockTransport(make_handler(**kwargs)),
            sleep=lambda seconds: None,
        )
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.close()


@pytest.fixture
def make_cfg(tmp_path: Path):
    def build(**settings) -> Config:
        data = yaml.safe_load(example_text())
        data["settings"].update({"min_score": 0.2, "feedback_weight": 0.2})
        data["settings"].update(settings)
        return parse_config(data, tmp_path / "home")

    return build
