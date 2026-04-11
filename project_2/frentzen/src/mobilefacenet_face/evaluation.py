from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .dataset import read_manifest
from .types import EvaluationSample


def calibrate_threshold(validation_rows: list[tuple[str, list[float]]]) -> float:
    candidates = np.linspace(-1.0, 1.0, 401)
    best_threshold = 0.35
    best_f1 = -1.0
    labels: list[int] = []
    scores: list[float] = []
    for label, pair_scores in validation_rows:
        labels.extend([1 if label == "genuine" else 0] * len(pair_scores))
        scores.extend(pair_scores)
    if not scores:
        return best_threshold
    y_true = np.asarray(labels)
    y_scores = np.asarray(scores)
    for threshold in candidates:
        y_pred = (y_scores >= threshold).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(samples: list[EvaluationSample]) -> dict:
    known_samples = [sample for sample in samples if sample.identity is not None]
    y_true = [sample.identity for sample in known_samples]
    y_pred = [sample.predicted_identity or "unknown" for sample in known_samples]
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist() if y_true else []
    top1_correct = sum(1 for sample in known_samples if sample.predicted_identity == sample.identity)
    top1_accuracy = top1_correct / max(1, len(known_samples))
    unknown_samples = [sample for sample in samples if sample.identity is None]
    unknown_rejection = (
        sum(1 for sample in unknown_samples if not sample.accepted) / max(1, len(unknown_samples))
        if unknown_samples
        else None
    )
    false_accept_rate = (
        sum(1 for sample in unknown_samples if sample.accepted) / max(1, len(unknown_samples))
        if unknown_samples
        else None
    )
    false_reject_rate = (
        sum(1 for sample in known_samples if not sample.accepted) / max(1, len(known_samples))
        if known_samples
        else None
    )
    avg_latency = sum(sample.latency_ms for sample in samples) / max(1, len(samples))
    return {
        "samples": len(samples),
        "known_samples": len(known_samples),
        "unknown_samples": len(unknown_samples),
        "top1_accuracy": top1_accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "unknown_rejection_rate": unknown_rejection,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "average_latency_ms": avg_latency,
        "labels": labels,
        "confusion_matrix": cm,
        "counts_by_identity": dict(Counter(y_true)),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def manifest_by_split(manifest_path: str | Path) -> dict[str, list]:
    records = read_manifest(manifest_path)
    by_split: dict[str, list] = {"enrollment": [], "validation": [], "test": []}
    for record in records:
        by_split.setdefault(record.split, []).append(record)
    return by_split
