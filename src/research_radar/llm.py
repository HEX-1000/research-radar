"""The local model side: embeddings and short summaries through Ollama."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from . import __version__

# Suggested when no chat model is installed. Any chat model works.
SUGGESTED_CHAT_MODEL = "qwen3.5:9b"


class OllamaError(Exception):
    """Ollama is not reachable or answered with something unusable."""


def is_embedding_model(name: str) -> bool:
    lowered = name.lower()
    return "embed" in lowered or "bert" in lowered


def has_model(names: list[str], model: str) -> bool:
    """True when Ollama lists the model. A name without a tag means the :latest tag, as in Ollama."""
    if not model:
        return False
    return model in names or (":" not in model and f"{model}:latest" in names)


def ollama_installed() -> bool:
    """True when the ollama program is on this computer (on PATH or in its usual Windows folder)."""
    if shutil.which("ollama"):
        return True
    local = os.environ.get("LOCALAPPDATA")
    return bool(local) and (Path(local) / "Programs" / "Ollama" / "ollama.exe").exists()


def _is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in ("localhost", "127.0.0.1", "::1")


def ollama_advice(
    settings: dict[str, Any],
    names: list[str] | None,
    *,
    retry_hint: str = "press s",
    installed: bool | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """What is missing on the Ollama side, in words a new user can act on.

    names is the list from Ollama's /api/tags, or None when Ollama could not be reached.
    Returns a state and (message, severity) pairs. States:
      ok        everything a scan uses is there
      no-chat   no usable chat model: items are described by their first sentence
      no-embed  the embedding model is missing: a scan can only match by keywords
      down      Ollama is not reachable: a scan can only match by keywords
    """
    url = str(settings.get("ollama_url") or "")
    if names is None:
        if installed is None:
            installed = ollama_installed() if _is_local(url) else True
        if not installed:
            return "down", [(
                "Ollama is not installed, so a scan can only match by keywords. "
                "Get it from https://ollama.com/download",
                "error",
            )]
        return "down", [(
            f"Ollama is not reachable at {url}, so a scan can only match by keywords. "
            f"Start Ollama (on Windows: from the Start menu), then {retry_hint}.",
            "error",
        )]

    messages: list[tuple[str, str]] = []
    state = "ok"
    embed = str(settings.get("embed_model") or "")
    if not has_model(names, embed):
        state = "no-embed"
        messages.append((
            f"The embedding model {embed} is not downloaded, so a scan can only match by keywords. "
            f"To get it, run: ollama pull {embed}",
            "warning",
        ))

    chat = str(settings.get("chat_model") or "")
    installed_chat = [n for n in names if not is_embedding_model(n)]
    if chat and not has_model(names, chat):
        if state == "ok":
            state = "no-chat"
        others = [n for n in installed_chat if n != chat]
        alternative = f"Set chat_model to {others[0]} in interests.yaml, or run" if others else "To get it, run"
        messages.append((
            f"The chat model {chat} is not downloaded, so items are described by their first sentence. "
            f"{alternative}: ollama pull {chat}",
            "warning",
        ))
    elif not chat and not installed_chat:
        if state == "ok":
            state = "no-chat"
        messages.append((
            "No chat model is installed, so items are described by their first sentence. "
            f"Any chat model works, for example: ollama pull {SUGGESTED_CHAT_MODEL}",
            "warning",
        ))
    return state, messages


class Ollama:
    def __init__(
        self,
        url: str,
        embed_model: str,
        chat_model: str = "",
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout,
            headers={"User-Agent": f"research-radar/{__version__}"},
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, f"{self.url}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.url} ({type(exc).__name__})") from exc
        if response.status_code == 404:
            raise OllamaError(f"404 from {path}")
        if response.status_code >= 400:
            detail = response.text[:200].replace("\n", " ")
            raise OllamaError(f"HTTP {response.status_code} from {path}: {detail}")
        try:
            return response.json()
        except ValueError as exc:
            raise OllamaError(f"{path} did not return JSON") from exc

    def models(self) -> list[str]:
        data = self._request("GET", "/api/tags")
        return [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]

    def ping(self) -> bool:
        try:
            self.models()
        except OllamaError:
            return False
        return True

    def resolve_chat_model(self) -> str:
        """The configured chat model, or the first installed model that is not an embedder."""
        if self.chat_model:
            return self.chat_model
        for name in self.models():
            if is_embedding_model(name):
                continue
            self.chat_model = name
            return name
        raise OllamaError("no chat model installed. Run `ollama pull` for one, or set settings.chat_model")

    def _prefix(self, kind: str) -> str:
        # These embedding models were trained with task prefixes and score better with them.
        name = self.embed_model.lower()
        if "nomic" in name:
            return "search_query: " if kind == "query" else "search_document: "
        if "embeddinggemma" in name:
            return "task: search result | query: " if kind == "query" else "title: none | text: "
        return ""

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        if not texts:
            return []
        prefix = self._prefix(kind)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 16):
            batch = [prefix + t for t in texts[start : start + 16]]
            try:
                data = self._request("POST", "/api/embed", json={"model": self.embed_model, "input": batch})
                batch_vectors = data.get("embeddings")
            except OllamaError as exc:
                if "404" not in str(exc):
                    raise
                batch_vectors = [  # older Ollama versions only have the single-text endpoint
                    self._request("POST", "/api/embeddings", json={"model": self.embed_model, "prompt": t}).get("embedding")
                    for t in batch
                ]
            if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch) or not all(batch_vectors):
                raise OllamaError(
                    f"embedding model '{self.embed_model}' returned nothing usable. Is it installed? (ollama pull {self.embed_model})"
                )
            vectors.extend(batch_vectors)
        return vectors

    def chat_json(self, system: str, user: str) -> dict[str, Any] | None:
        """One JSON answer from the chat model, or None when the reply is not usable."""
        model = self.resolve_chat_model()
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "format": "json",
            # Models that think before answering would spend minutes on a one-line note.
            "think": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            data = self._request("POST", "/api/chat", json=payload)
        except OllamaError as exc:
            if "think" not in str(exc).lower():
                raise
            payload.pop("think")  # an Ollama version or model that does not know the option
            data = self._request("POST", "/api/chat", json=payload)
        content = (data.get("message") or {}).get("content", "")
        content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
