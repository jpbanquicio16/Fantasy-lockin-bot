"""Immutable raw store and polite HTTP fetching.

Every response is written once, unchanged, as gzip JSON keyed by source, endpoint,
parameters and fetch time, with the SHA-256 of its body. A re-fetch whose body is
identical to the latest version writes nothing (only the check time moves); a changed
body becomes a new version. Nothing is overwritten, so stat corrections and every
injury-status change stay recoverable.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx

from lockin_core.config import UTC, now_utc


def _sha(body: Any) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:80] or "root"


def params_key(params: dict | None) -> str:
    if not params:
        return "noparams"
    blob = json.dumps(params, sort_keys=True)
    return _slug("_".join(f"{k}-{v}" for k, v in sorted(params.items())))[:60] + "_" + (
        hashlib.sha256(blob.encode()).hexdigest()[:8]
    )


@dataclass(frozen=True)
class RawRecord:
    path: Path
    fetched_at: datetime  # first time this exact body was seen
    sha256: str
    body: Any


class RawStore:
    def __init__(self, root: Path):
        self.root = root

    def dir(self, source: str, endpoint: str, params: dict | None) -> Path:
        return self.root / source / _slug(endpoint) / params_key(params)

    def versions(self, source: str, endpoint: str, params: dict | None) -> list[Path]:
        d = self.dir(source, endpoint, params)
        return sorted(d.glob("*.json.gz")) if d.exists() else []

    @staticmethod
    def read(path: Path) -> RawRecord:
        with gzip.open(path, "rt") as f:
            rec = json.load(f)
        return RawRecord(path, datetime.fromisoformat(rec["fetched_at"]), rec["sha256"], rec["body"])

    def latest(self, source: str, endpoint: str, params: dict | None) -> RawRecord | None:
        v = self.versions(source, endpoint, params)
        return self.read(v[-1]) if v else None

    def checked_at(self, source: str, endpoint: str, params: dict | None) -> datetime | None:
        p = self.dir(source, endpoint, params) / "_checked_at"
        return datetime.fromisoformat(p.read_text().strip()) if p.exists() else None

    def put(self, source: str, endpoint: str, params: dict | None, body: Any,
            fetched_at: datetime | None = None) -> tuple[RawRecord, bool]:
        """Store ``body``; returns (record, is_new_version)."""
        fetched_at = fetched_at or now_utc()
        d = self.dir(source, endpoint, params)
        d.mkdir(parents=True, exist_ok=True)
        sha = _sha(body)
        prev = self.latest(source, endpoint, params)
        (d / "_checked_at").write_text(fetched_at.isoformat())
        if prev is not None and prev.sha256 == sha:
            return prev, False
        path = d / f"{fetched_at.strftime('%Y%m%dT%H%M%S%fZ')}_{sha[:8]}.json.gz"
        tmp = path.with_suffix(".tmp")
        rec = {"source": source, "endpoint": endpoint, "params": params,
               "fetched_at": fetched_at.isoformat(), "sha256": sha, "body": body}
        with gzip.open(tmp, "wt") as f:
            json.dump(rec, f)
        tmp.replace(path)
        return RawRecord(path, fetched_at, sha, body), True

    def fetch(self, source: str, endpoint: str, params: dict | None,
              fetcher: Callable[[], Any], max_age_s: float | None) -> RawRecord:
        """Latest stored version if it was checked within ``max_age_s`` (None = forever)."""
        prev = self.latest(source, endpoint, params)
        if prev is not None:
            checked = self.checked_at(source, endpoint, params) or prev.fetched_at
            if max_age_s is None or (now_utc() - checked).total_seconds() < max_age_s:
                return prev
        rec, _ = self.put(source, endpoint, params, fetcher())
        return rec


class RateLimiter:
    """Minimum interval between calls, shared across threads (a one-token bucket)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class FetchError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


class HttpClient:
    """httpx with a rate limiter and retries (429, 5xx and timeouts only)."""

    def __init__(self, base_url: str, min_interval: float = 0.1, retries: int = 4,
                 timeout: float = 30.0, transport: httpx.BaseTransport | None = None,
                 user_agent: str = "lockin-assistant/0.1 (read-only; personal use)"):
        self.base_url = base_url
        self.retries = retries
        self._limiter = RateLimiter(min_interval)
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport,
                                    headers={"User-Agent": user_agent})

    def get_json(self, path: str, params: dict | None = None) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries):
            self._limiter.wait()
            try:
                r = self._client.get(path, params=params)
            except httpx.TransportError as e:  # timeouts, connection resets
                last = e
            else:
                if r.status_code == 429 or r.status_code >= 500:
                    last = FetchError(f"HTTP {r.status_code}", r.status_code)
                elif r.status_code >= 400:
                    raise FetchError(f"GET {self.base_url}{path}: HTTP {r.status_code}", r.status_code)
                else:
                    return r.json()
            time.sleep(min(30.0, 0.5 * 2**attempt) * (0.5 + random.random()))
        raise FetchError(f"GET {self.base_url}{path} failed after {self.retries} attempts: {last}")


    def get_bytes(self, path: str, params: dict | None = None, headers: dict | None = None
                  ) -> tuple[int, bytes, str]:
        """(status, body, content type). 403/404 are answers, not errors: a report slot that
        was never published returns 403 or 404."""
        last: Exception | None = None
        for attempt in range(self.retries):
            self._limiter.wait()
            try:
                r = self._client.get(path, params=params, headers=headers)
            except httpx.TransportError as e:
                last = e
            else:
                if r.status_code == 429 or r.status_code >= 500:
                    last = FetchError(f"HTTP {r.status_code}", r.status_code)
                else:
                    return r.status_code, r.content, r.headers.get("content-type", "")
            time.sleep(min(30.0, 0.5 * 2**attempt) * (0.5 + random.random()))
        raise FetchError(f"GET {self.base_url}{path} failed after {self.retries} attempts: {last}")


def utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
