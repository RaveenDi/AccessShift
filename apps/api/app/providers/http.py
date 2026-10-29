"""HTTP client for the Meta Model API: one place for telemetry (and, below, retry policy).

UNVERIFIED: the `usage` field names (prompt_tokens / completion_tokens) follow the OpenAI-compatible surface
assumed in the planning docs. If the API omits them, token counts stay None and cost is reported as unknown.
"""
from __future__ import annotations

import time

import httpx

from ..telemetry import CallRecord, estimate_cost, record_call
from .base import ProviderError

BASE = "https://api.meta.ai/v1"


def _usage(data) -> tuple[int | None, int | None]:
    u = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(u, dict):
        return None, None
    def num(v):
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    return num(u.get("prompt_tokens")), num(u.get("completion_tokens"))


class ModelClient:
    def __init__(self, api_key: str, transport: httpx.BaseTransport | None = None, timeout: float = 60):
        self.http = httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {api_key}"},
                                 timeout=timeout, transport=transport)

    def post(self, path: str, payload: dict, *, purpose: str = "", prompt_version: str | None = None) -> dict:
        model = str(payload.get("model", ""))
        t0 = time.perf_counter()
        status, http_status, attempts = "ok", None, 1
        data, itok, otok = None, None, None
        try:
            try:
                r = self.http.post(path, json=payload)
                http_status = r.status_code
                r.raise_for_status()
                data = r.json()
            except (httpx.HTTPError, ValueError) as e:
                status = f"error:{type(e).__name__}"
                raise ProviderError(f"meta api call failed: {type(e).__name__}") from e
            itok, otok = _usage(data)
            return data
        finally:
            ms = (time.perf_counter() - t0) * 1000
            record_call(CallRecord(path, model, purpose, prompt_version, attempts, round(ms, 1), status, http_status,
                                   itok, otok, estimate_cost(model, itok, otok) if status == "ok" else None))
