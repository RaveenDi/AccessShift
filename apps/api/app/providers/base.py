"""Provider interfaces. Models perceive and interpret; they never decide connectivity."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field


class ProviderError(RuntimeError):
    """A provider call failed. `kind` is one of: error, timeout, rate_limited, server_error, http_error,
    malformed_response (set by the HTTP layer once retries are exhausted)."""

    def __init__(self, msg: str, kind: str = "error"):
        super().__init__(msg)
        self.kind = kind


@dataclass
class DetectedObject:
    label: str
    mask: np.ndarray  # bool HxW in image pixels
    confidence: float = 1.0


@dataclass
class PerceptionResult:
    board_corners_px: list  # TL, TR, BR, BL
    check_points_px: list  # two points of the independent known-length reference
    objects: list[DetectedObject]
    notes: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)  # provider-detected problems, e.g. {"kind": "no_mask", ...}
    # where the calibration inputs came from, e.g. {"board_corners": "refined", "check_points": "seed"}; empty = not tracked
    sources: dict[str, str] = field(default_factory=dict)


class ConstraintSpec(BaseModel):
    """Validated structure a constraint interpreter must return (model output is data, not truth)."""
    start_name: str = Field(min_length=1, max_length=40)
    end_name: str = Field(min_length=1, max_length=40)
    clearance_cm: float = Field(gt=20, lt=300)
    locked_labels: list[str] = Field(default_factory=list, max_length=20)
    max_moves: int = Field(default=2, ge=0, le=3)
    ambiguities: list[str] = Field(default_factory=list)


class PerceptionProvider(Protocol):
    name: str
    model: str

    def perceive(self, image_bgr: np.ndarray) -> PerceptionResult: ...


class ConstraintProvider(Protocol):
    name: str
    model: str

    def interpret(self, text: str, known_labels: list[str]) -> ConstraintSpec: ...


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None  # None when the model's arguments were not a valid JSON object
    error: str | None = None


@dataclass
class ModelTurn:
    content: str | None
    tool_calls: list[ToolCall]
    model: str
    finish_reason: str | None = None


class PlannerProvider(Protocol):
    """One model turn of the tool-calling loop. Messages and tool specs use a neutral shape
    ({"role", "content", "tool_calls": [{"id","name","arguments": dict}], "tool_call_id"}); the wire format
    (OpenAI-style `tools` / `tool_calls`, verified live for Meta on 2026-10-30) lives only inside the provider."""
    name: str
    model: str

    def next_turn(self, messages: list[dict], tools: list[dict]) -> ModelTurn: ...
