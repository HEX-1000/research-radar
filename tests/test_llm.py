from __future__ import annotations

import json

import httpx
import pytest

from research_radar.llm import Ollama, OllamaError


def client(handler, **kwargs) -> Ollama:
    return Ollama("http://localhost:11434", kwargs.pop("embed_model", "nomic-embed-text"), kwargs.pop("chat_model", "m"),
                  transport=httpx.MockTransport(handler))


def test_embedding_prefixes_depend_on_the_model():
    sent: list[list[str]] = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body["input"])
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in body["input"]]})

    client(handler, embed_model="embeddinggemma:latest").embed(["a"], "query")
    client(handler, embed_model="embeddinggemma:latest").embed(["a"], "document")
    client(handler, embed_model="nomic-embed-text").embed(["a"], "document")
    client(handler, embed_model="other-model").embed(["a"], "document")
    assert sent == [
        ["task: search result | query: a"],
        ["title: none | text: a"],
        ["search_document: a"],
        ["a"],
    ]


def test_embedding_batches_of_sixteen():
    sizes: list[int] = []

    def handler(request):
        body = json.loads(request.content)
        sizes.append(len(body["input"]))
        return httpx.Response(200, json={"embeddings": [[1.0] for _ in body["input"]]})

    assert len(client(handler).embed([str(i) for i in range(40)])) == 40
    assert sizes == [16, 16, 8]


def test_missing_embedding_model_has_a_helpful_error():
    handler = lambda request: httpx.Response(200, json={"embeddings": []})  # noqa: E731
    with pytest.raises(OllamaError, match="ollama pull"):
        client(handler).embed(["x"])


def test_chat_turns_thinking_off_and_strips_think_blocks():
    seen: dict = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '<think>hmm</think>{"what": "ok"}'}})

    assert client(handler).chat_json("s", "u") == {"what": "ok"}
    assert seen["think"] is False and seen["format"] == "json"


def test_chat_retries_without_think_when_the_option_is_rejected():
    bodies: list[dict] = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, text='{"error":"model does not support think"}')
        return httpx.Response(200, json={"message": {"content": '{"what": "ok"}'}})

    assert client(handler).chat_json("s", "u") == {"what": "ok"}
    assert len(bodies) == 2 and "think" not in bodies[1]


def test_unusable_chat_reply_is_none():
    handler = lambda request: httpx.Response(200, json={"message": {"content": "not json"}})  # noqa: E731
    assert client(handler).chat_json("s", "u") is None


def test_auto_chat_model_skips_embedders():
    handler = lambda request: httpx.Response(  # noqa: E731
        200, json={"models": [{"name": "embeddinggemma:latest"}, {"name": "qwen3.5:4b"}]}
    )
    assert client(handler, chat_model="").resolve_chat_model() == "qwen3.5:4b"
