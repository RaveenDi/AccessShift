"""Record/replay of Meta Model API traffic at the HTTP transport layer.

meta mode  : RecordingTransport wraps the real transport and saves every successful request/response pair.
replay mode: ReplayTransport answers from those files and raises ReplayMissError (loudly) on a miss.

Key = sha256 of the normalized request (method, path, JSON body with long strings such as base64 images
replaced by their sha256). The Authorization header is never read into the key or written to disk.
Recordings are model output: treat them as data, keep them out of git (they can contain your photos' content).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

MAX_INLINE_STR = 4096


class ReplayMissError(Exception):
    """Deliberately NOT a ProviderError: graceful-degradation paths must not swallow a replay miss."""


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def normalize(obj):
    """JSON-compatible copy where very long strings (images) are replaced by a digest."""
    if isinstance(obj, str):
        return {"$sha256": _sha(obj), "len": len(obj)} if len(obj) > MAX_INLINE_STR else obj
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    return obj


def request_key(method: str, path: str, body) -> str:
    canon = json.dumps({"m": method.upper(), "p": path, "b": normalize(body)}, sort_keys=True,
                       separators=(",", ":"), ensure_ascii=True)
    return _sha(canon)


def _request_body(request: httpx.Request):
    raw = request.read()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return {"$raw_sha256": _sha(raw)}


def _key_of(request: httpx.Request) -> tuple[str, object]:
    body = _request_body(request)
    return request_key(request.method, request.url.path, body), body


class RecordingTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport, directory: Path | str):
        self.inner, self.dir = inner, Path(directory)
        self.saved = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key, body = _key_of(request)
        response = self.inner.handle_request(request)
        response.read()
        if response.status_code < 400:  # errors are never recorded; a replay of a 429 would only mislead
            self._save(key, request, body, response)
        return response

    def _save(self, key, request, body, response):
        ctype = response.headers.get("content-type", "")
        try:
            rbody, is_json = response.json(), True
        except ValueError:
            rbody, is_json = response.text, False
        rec = {"version": 1, "key": key, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "request": {"method": request.method, "path": request.url.path, "body": normalize(body),
                           "model": body.get("model") if isinstance(body, dict) else None},
               "response": {"status_code": response.status_code, "content_type": ctype,
                            "is_json": is_json, "body": rbody}}
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{key}.tmp"
        tmp.write_text(json.dumps(rec, ensure_ascii=True), encoding="utf-8")
        os.replace(tmp, self.dir / f"{key}.json")
        self.saved += 1


class ReplayTransport(httpx.BaseTransport):
    def __init__(self, directory: Path | str):
        self.dir = Path(directory)
        self.hits: set[str] = set()
        self.misses: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key, _ = _key_of(request)
        f = self.dir / f"{key}.json"
        if not f.exists():
            self.misses.append(key)
            raise ReplayMissError(
                f"replay miss: no recording for {request.method} {request.url.path} (key {key[:16]}...) in "
                f"{self.dir}. The request changed or was never recorded; re-record with MODEL_PROVIDER=meta "
                f"(live, billed) instead of guessing an answer.")
        self.hits.add(key)
        r = json.loads(f.read_text(encoding="utf-8"))["response"]
        if r.get("is_json", True):
            return httpx.Response(r["status_code"], json=r["body"], request=request)
        return httpx.Response(r["status_code"], text=r["body"], request=request,
                              headers={"content-type": r.get("content_type") or "text/plain"})


def live_transport(directory: Path | str, inner: httpx.BaseTransport | None = None) -> httpx.BaseTransport:
    """Real network transport that records every successful exchange (meta mode)."""
    return RecordingTransport(inner or httpx.HTTPTransport(), directory)
