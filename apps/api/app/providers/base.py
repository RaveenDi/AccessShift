"""Provider interfaces. Models perceive and interpret; they never decide connectivity."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field


class ProviderError(RuntimeError):
    pass


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
