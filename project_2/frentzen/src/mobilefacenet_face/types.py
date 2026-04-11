from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class Detection:
    box: np.ndarray
    landmarks: np.ndarray
    confidence: float


@dataclass(slots=True)
class MatchResult:
    identity: str | None
    score: float
    accepted: bool
    top_k: list[tuple[str, float]] = field(default_factory=list)


@dataclass(slots=True)
class FacePrediction:
    detection: Detection
    aligned_face: np.ndarray
    embedding: np.ndarray
    match: MatchResult
    timings_ms: dict[str, float]


@dataclass(slots=True)
class PipelineResult:
    predictions: list[FacePrediction]
    timings_ms: dict[str, float]


@dataclass(slots=True)
class SampleRecord:
    identity: str
    path: Path
    split: str
    group_id: str


@dataclass(slots=True)
class EvaluationSample:
    identity: str | None
    predicted_identity: str | None
    score: float
    accepted: bool
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
