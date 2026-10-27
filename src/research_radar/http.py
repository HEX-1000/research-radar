"""A small polite HTTP client: one user agent, per-host spacing, limited retries."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

import httpx

from . import __version__

USER_AGENT = f"research-radar/{__version__} (+https://github.com/HEX-1000/research-radar)"
MAX_GITHUB_WAIT = 90.0  # seconds a scan is willing to wait for a GitHub rate limit to clear


class SourceError(Exception):
    """A source could not be read. The scan goes on without it."""


class Http:
    def __init__(
        self,
        *,
        github_token: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
        timeout: float = 30.0,
    ) -> None:
        self.github_token = github_token
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self._sleep = sleep
        self._clock = clock
        self._last_call: dict[str, float] = {}

    def close(self) -> None:
        self._client.close()

    def _wait(self, bucket: str, min_interval: float) -> None:
        if min_interval <= 0:
            return
        last = self._last_call.get(bucket)
        if last is not None:
            gap = self._clock() - last
            if gap < min_interval:
                self._sleep(min_interval - gap)
        self._last_call[bucket] = self._clock()

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        bucket: str | None = None,
        min_interval: float = 0.0,
        retries: int = 2,
    ) -> httpx.Response:
        host = urlparse(url).netloc
        hdrs = dict(headers or {})
        if self.github_token and host == "api.github.com":
            hdrs.setdefault("Authorization", f"Bearer {self.github_token}")
        attempt = 0
        while True:
            self._wait(bucket or host, min_interval)
            try:
                response = self._client.get(url, params=params, headers=hdrs)
            except httpx.HTTPError as exc:
                if attempt < retries:
                    attempt += 1
                    self._sleep(2.0 * attempt)
                    continue
                raise SourceError(f"{host}: {type(exc).__name__}: {exc}") from exc
            if response.status_code in (403, 429) and host == "api.github.com":
                message = _github_message(response)
                wait = _github_wait(response, message)
                if wait is not None and wait <= MAX_GITHUB_WAIT and attempt < retries:
                    attempt += 1
                    self._sleep(wait)
                    continue
                if wait is not None:
                    detail = f" (GitHub says: {message})" if message else ""
                    raise SourceError(f"the search rate limit is used up, try again later or set GITHUB_TOKEN{detail}")
                if response.status_code == 403:
                    raise SourceError(f"{host}: HTTP 403" + (f" ({message})" if message else ""))
            if response.status_code in (429, 502, 503, 504) and attempt < retries:
                attempt += 1
                self._sleep(_retry_after(response, default=3.0 * attempt))
                continue
            if response.status_code >= 400:
                raise SourceError(f"{host}: HTTP {response.status_code}")
            return response

    def get_json(self, url: str, **kwargs: Any) -> Any:
        response = self.get(url, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise SourceError(f"{urlparse(url).netloc}: response is not JSON") from exc


def _github_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return ""
    return " ".join(str(data.get("message", "")).split())[:160] if isinstance(data, dict) else ""


def _github_wait(response: httpx.Response, message: str) -> float | None:
    """Seconds to wait when GitHub is rate limiting us, or None when this 403 is something else."""
    value = response.headers.get("retry-after")
    if value:
        try:
            return float(value) + 1.0
        except ValueError:
            pass
    if response.headers.get("x-ratelimit-remaining") == "0":
        try:
            return max(1.0, float(response.headers["x-ratelimit-reset"]) - time.time()) + 1.0
        except (KeyError, ValueError):
            return MAX_GITHUB_WAIT + 1.0  # the reset time is unknown, so do not wait for it
    if "rate limit" in message.lower():
        return 60.0  # GitHub's advice for a secondary limit without a Retry-After header
    return None


def _retry_after(response: httpx.Response, default: float) -> float:
    value = response.headers.get("retry-after")
    if value:
        try:
            return min(float(value), 30.0)
        except ValueError:
            pass
    return default
