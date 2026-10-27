"""The local model side: embeddings and short summaries through Ollama."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from . import __version__


class OllamaError(Exception):
    """Ollama is not reachable or answered with something unusable."""


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
            lowered = name.lower()
            if "embed" in lowered or "bert" in lowered:
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
