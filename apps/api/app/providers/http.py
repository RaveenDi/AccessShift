"""HTTP client for the Meta Model API: retry policy, call budget and telemetry in one place.

Policy (config.http_policy): timeouts, HTTP 429 and 5xx are retried at most `max_retries` (2) times with
exponential backoff; a Retry-After header (seconds or HTTP date) is honoured, and a Retry-After longer than
`retry_after_max_s` is not waited out (fail fast, clearly). Other 4xx and non-JSON bodies are not retried.

VERIFIED 2026-10-30: chat completions return usage.prompt_tokens / completion_tokens (+ nested *_details). The
Responses route used by SAM returns usage with input_tokens=0 and images_processed instead, so SAM calls have no
token counts and are costed per call (config.PRICE_TABLE_DEFAULT["sam-3.1"]).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from ..config import http_policy
from ..telemetry import CallRecord, estimate_cost, record_call
from .base import ProviderError
from .recording import RecordingTransport

BASE = "https://api.meta.ai/v1"


class CallBudgetExceeded(Exception):
    """Deliberately NOT a ProviderError: an exhausted --max-calls budget must stop the run, not degrade it."""


class CallBudget:
    """Counts live HTTP attempts (retries included) so an eval run can be capped (evals/run.py --max-calls)."""

    def __init__(self):
        self.limit, self.used = None, 0

    def set(self, limit: int | None) -> None:
        self.limit, self.used = limit, 0

    def charge(self) -> None:
        if self.limit is not None and self.used >= self.limit:
            raise CallBudgetExceeded(f"live call budget of {self.limit} HTTP calls is exhausted")
        self.used += 1


BUDGET = CallBudget()


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((dt - datetime.now(timezone.utc)).total_seconds(), 0.0)
    except (TypeError, ValueError):
        return None


def _usage(data) -> tuple[int | None, int | None]:
    u = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(u, dict):
        return None, None

    def num(v):
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    return num(u.get("prompt_tokens")), num(u.get("completion_tokens"))


class ModelClient:
    def __init__(self, api_key: str, transport: httpx.BaseTransport | None = None, timeout: float | None = None,
                 live: bool | None = None):
        self.policy = http_policy()
        self.http = httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {api_key}"},
                                 timeout=timeout or self.policy.timeout_s, transport=transport)
        self.live = (transport is None or isinstance(transport, RecordingTransport)) if live is None else live
        self.sleep = time.sleep  # tests replace this

    def _attempt(self, path: str, payload: dict):
        """-> (data|None, http_status|None, kind|None, retryable, retry_after)"""
        try:
            r = self.http.post(path, json=payload)
        except httpx.TimeoutException:
            return None, None, "timeout", True, None
        except httpx.TransportError:
            return None, None, "network", True, None
        code = r.status_code
        if code == 429:
            return None, code, "rate_limited", True, parse_retry_after(r.headers.get("retry-after"))
        if code >= 500:
            return None, code, "server_error", True, parse_retry_after(r.headers.get("retry-after"))
        if code >= 400:
            return None, code, "http_error", False, None
        try:
            data = r.json()
        except ValueError:
            return None, code, "malformed_response", False, None
        if not isinstance(data, dict):
            return None, code, "malformed_response", False, None
        return data, code, None, False, None

    def post(self, path: str, payload: dict, *, purpose: str = "", prompt_version: str | None = None) -> dict:
        model, pol = str(payload.get("model", "")), self.policy
        attempts, api_s, http_status, status = 0, 0.0, None, "ok"
        data, itok, otok = None, None, None
        try:
            while True:
                attempts += 1
                if self.live:
                    BUDGET.charge()
                t0 = time.perf_counter()
                data, http_status, kind, retryable, retry_after = self._attempt(path, payload)
                api_s += time.perf_counter() - t0
                if kind is None:
                    status, (itok, otok) = "ok", _usage(data)
                    return data
                status = f"error:{kind}"
                if not retryable or attempts > pol.max_retries:
                    raise ProviderError(f"meta api call failed: {kind}" + (f" (HTTP {http_status})" if http_status
                                                                           else "") + f" after {attempts} attempt(s)", kind)
                if retry_after is not None:
                    if retry_after > pol.retry_after_max_s:
                        raise ProviderError(f"meta api call failed: {kind}; Retry-After {retry_after:.0f}s exceeds "
                                            f"the {pol.retry_after_max_s:.0f}s cap, not waiting", kind)
                    delay = retry_after
                else:
                    delay = min(pol.backoff_base_s * 2 ** (attempts - 1), pol.backoff_max_s)
                self.sleep(delay)
        finally:
            record_call(CallRecord(path, model, purpose, prompt_version, attempts, round(api_s * 1000, 1), status,
                                   http_status, itok, otok,
                                   estimate_cost(model, itok, otok) if status == "ok" else None))
