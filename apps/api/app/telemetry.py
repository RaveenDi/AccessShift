"""Per-step telemetry: model, prompt version, tokens (when returned), latency, estimated cost, run totals.

HTTP-level calls made inside a `step(...)` block are collected through a context variable, so providers do not
need to know about the trace. Token counts are None when the API did not return usage. Prices come from
config.price_table() and are UNVERIFIED placeholders unless the owner overrides them.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from . import config

CALL_SINK: ContextVar[list | None] = ContextVar("call_sink", default=None)


@dataclass
class CallRecord:
    path: str
    model: str
    purpose: str
    prompt_version: str | None
    attempts: int
    latency_ms: float
    status: str  # ok | error:<kind>
    http_status: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd_est: float | None = None


def estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None, attempts: int = 1) -> float | None:
    p = config.price_table().get(model)
    if not p:
        return None
    if "per_call" in p:
        return float(p["per_call"]) * attempts
    if input_tokens is None and output_tokens is None:
        return None
    return ((input_tokens or 0) * p.get("input_per_mtok", 0) + (output_tokens or 0) * p.get("output_per_mtok", 0)) / 1e6


def record_call(rec: CallRecord) -> None:
    sink = CALL_SINK.get()
    if sink is not None:
        sink.append(rec)


class StepHandle:
    def __init__(self, rec: dict):
        self.rec, self.output, self.status = rec, "", None

    def out(self, s: str) -> None:
        self.output = s

    def fail(self, s: str, status: str = "tool_error") -> None:
        self.output, self.status = s, status


def _sum(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _apply_calls(rec: dict, calls: list[CallRecord]) -> None:
    rec["api_calls"] = len(calls)
    rec["retries"] = sum(max(c.attempts - 1, 0) for c in calls)
    rec["input_tokens"] = _sum(c.input_tokens for c in calls)
    rec["output_tokens"] = _sum(c.output_tokens for c in calls)
    rec["latency_ms"] = round(sum(c.latency_ms for c in calls), 1)
    cost = _sum(c.cost_usd_est for c in calls)
    rec["cost_usd_est"] = round(cost, 6) if cost is not None else None
    if rec.get("prompt_version") is None and calls:
        rec["prompt_version"] = calls[0].prompt_version
    if any(c.status != "ok" for c in calls) and rec["status"] == "ok":
        rec["api_errors"] = [c.status for c in calls if c.status != "ok"]


@contextmanager
def step(trace: list, agent: str, role: str, model: str, inp: str, kind: str = "deterministic",
         prompt_version: str | None = None):
    rec = {"n": len(trace) + 1, "agent": agent, "role": role, "model": model, "kind": kind,
           "prompt_version": prompt_version, "input_summary": inp, "output_summary": "", "duration_ms": 0,
           "status": "ok"}
    trace.append(rec)
    h, calls, t0 = StepHandle(rec), [], time.perf_counter()
    token = CALL_SINK.set(calls)
    try:
        yield h
    except Exception as e:
        rec["status"] = "error"
        h.output = f"{type(e).__name__}: {e}"
        raise
    finally:
        CALL_SINK.reset(token)
        rec["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["output_summary"] = h.output
        if h.status and rec["status"] == "ok":
            rec["status"] = h.status
        _apply_calls(rec, calls)


def totals(trace: list[dict]) -> dict:
    cost = _sum(r.get("cost_usd_est") for r in trace)
    return {"steps": len(trace),
            "agent_model_turns": sum(r.get("kind") == "model_turn" for r in trace),
            "agent_tool_calls": sum(r.get("kind") == "tool_call" for r in trace),
            "tool_errors": sum(r.get("status") == "tool_error" for r in trace),
            "api_calls": sum(r.get("api_calls", 0) for r in trace),
            "retries": sum(r.get("retries", 0) for r in trace),
            "input_tokens": _sum(r.get("input_tokens") for r in trace),
            "output_tokens": _sum(r.get("output_tokens") for r in trace),
            "api_latency_ms": round(sum(r.get("latency_ms", 0.0) for r in trace), 1),
            "wall_ms": round(sum(r.get("duration_ms", 0.0) for r in trace), 1),
            "cost_usd_est": round(cost, 6) if cost is not None else 0.0,
            "prices_verified": config.PRICES_VERIFIED,
            "cost_basis": ("UNVERIFIED placeholder prices (config.PRICE_TABLE_DEFAULT) unless PRICE_TABLE_JSON "
                           "is set; 0 when no billed call was made"),
            "tokens_note": "None = the API did not return usage (mock mode, or the field shape is UNVERIFIED)"}
