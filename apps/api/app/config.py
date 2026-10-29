"""Runtime configuration read from the environment (never from files that hold secrets).

Every number marked UNVERIFIED is a placeholder to be replaced from the real Meta pricing/limits pages.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Prompt versions are recorded on every model step so a trace says which prompt produced it.
PROMPT_VERSIONS = {
    "perceiver_labels": "pl-1",
    "sam_segment": "sam-1",
    "planner_agent": "pa-1",
    "planner_recovery": "pr-1",
    "constraint_interpreter": "ci-1",
}

# UNVERIFIED placeholder prices in USD. They exist so the cost plumbing can be exercised; they are NOT Meta's
# prices. Override with PRICE_TABLE_JSON='{"muse-spark-1.3": {"input_per_mtok": .., "output_per_mtok": ..}}'.
PRICE_TABLE_DEFAULT = {
    "muse-spark-1.3": {"input_per_mtok": 1.0, "output_per_mtok": 3.0},
    "sam-3.1": {"per_call": 0.005},
}
PRICES_VERIFIED = False


def _num(name: str, default, cast=int):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from None


def price_table() -> dict:
    raw = os.environ.get("PRICE_TABLE_JSON", "").strip()
    return json.loads(raw) if raw else PRICE_TABLE_DEFAULT


def recordings_dir() -> Path:
    """Where meta-mode request/response pairs are saved and replay reads them (gitignored)."""
    p = os.environ.get("RECORDINGS_DIR", "").strip()
    return Path(p) if p else ROOT / "apps" / "api" / "demo_data" / "recordings"


def record_enabled() -> bool:
    return os.environ.get("RECORD_MODEL_CALLS", "1").strip() not in ("0", "false", "no")


@dataclass(frozen=True)
class AgentCaps:
    max_steps: int = 8  # tool invocations per agent phase
    max_model_calls: int = 8  # model turns per agent phase (repairs count)
    recovery_max_steps: int = 6
    recovery_max_model_calls: int = 6
    max_ask_user: int = 2
    repair_attempts: int = 1
    max_moves_cap: int = 3


def agent_caps() -> AgentCaps:
    d = AgentCaps()
    return AgentCaps(
        max_steps=_num("AGENT_MAX_STEPS", d.max_steps),
        max_model_calls=_num("AGENT_MAX_MODEL_CALLS", d.max_model_calls),
        recovery_max_steps=_num("AGENT_RECOVERY_MAX_STEPS", d.recovery_max_steps),
        recovery_max_model_calls=_num("AGENT_RECOVERY_MAX_MODEL_CALLS", d.recovery_max_model_calls),
        max_ask_user=_num("AGENT_MAX_ASK_USER", d.max_ask_user),
        repair_attempts=_num("AGENT_REPAIR_ATTEMPTS", d.repair_attempts),
        max_moves_cap=_num("AGENT_MAX_MOVES_CAP", d.max_moves_cap),
    )


@dataclass(frozen=True)
class HttpPolicy:
    max_retries: int = 2
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    retry_after_max_s: float = 30.0
    timeout_s: float = 60.0


def http_policy() -> HttpPolicy:
    d = HttpPolicy()
    return HttpPolicy(
        max_retries=_num("HTTP_MAX_RETRIES", d.max_retries),
        backoff_base_s=_num("HTTP_BACKOFF_BASE_S", d.backoff_base_s, float),
        backoff_max_s=_num("HTTP_BACKOFF_MAX_S", d.backoff_max_s, float),
        retry_after_max_s=_num("HTTP_RETRY_AFTER_MAX_S", d.retry_after_max_s, float),
        timeout_s=_num("HTTP_TIMEOUT_S", d.timeout_s, float),
    )
