from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..dataset import read_manifest
from ..evaluation import calibrate_threshold, compute_metrics, load_image, save_metrics
from ..gallery import EmbeddingGallery
from .enroll_gallery import build_gallery_from_manifest
from ..pipeline import build_pipeline
from ..types import EvaluationSample


def _to_sample(record, result) -> EvaluationSample:
    if not result.predictions:
        return EvaluationSample(
            identity=record.identity,
            predicted_identity=None,
            score=float("-inf"),
            accepted=False,
            latency_ms=result.timings_ms["total_ms"],
            metadata={"path": str(record.path)},
        )
    prediction = result.predictions[0]
    return EvaluationSample(
        identity=record.identity,
        predicted_identity=prediction.match.identity,
        score=prediction.match.score,
        accepted=prediction.match.accepted,
        latency_ms=result.timings_ms["total_ms"],
        metadata={"path": str(record.path)},
    )


def _run_evaluation(config_path, checkpoint_path=None) -> dict:
    config = load_config(config_path)
    manifest = read_manifest(config["outputs"]["split_manifest"])
    _, gallery = build_gallery_from_manifest(config_path, checkpoint_path=checkpoint_path)
    pipeline = build_pipeline(config_path, gallery_path=None, checkpoint_path=checkpoint_path, load_gallery=False)
    pipeline.gallery = gallery

    validation_scores: list[tuple[str, list[float]]] = []
    skipped_validation_missing = 0
    for record in manifest:
        if record.split != "validation":
            continue
        try:
            frame = load_image(record.path)
        except FileNotFoundError:
            skipped_validation_missing += 1
            continue
        result = pipeline.run(frame)
        sample = _to_sample(record, result)
        validation_scores.append(
            ("genuine" if sample.predicted_identity == sample.identity else "impostor", [sample.score])
        )
    pipeline.threshold = calibrate_threshold(validation_scores)

    test_samples = []
    skipped_test_missing = 0
    for record in manifest:
        if record.split != "test":
            continue
        try:
            frame = load_image(record.path)
        except FileNotFoundError:
            skipped_test_missing += 1
            continue
        result = pipeline.run(frame)
        test_samples.append(_to_sample(record, result))

    metrics = compute_metrics(test_samples)
    metrics["calibrated_threshold"] = pipeline.threshold
    metrics["checkpoint_path"] = checkpoint_path or config["recognizer"].get("checkpoint_path")
    metrics["skipped_validation_missing"] = skipped_validation_missing
    metrics["skipped_test_missing"] = skipped_test_missing
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--comparison-checkpoint-path", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = _run_evaluation(args.config, checkpoint_path=args.checkpoint_path)
    save_metrics(metrics, config["outputs"]["metrics_path"])
    if args.comparison_checkpoint_path:
        comparison = _run_evaluation(args.config, checkpoint_path=args.comparison_checkpoint_path)
        comparison_path = Path(config["outputs"]["comparison_metrics_path"])
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        with comparison_path.open("w", encoding="utf-8") as handle:
            json.dump({"primary": metrics, "comparison": comparison}, handle, indent=2)
    print(
        f"Saved metrics to {config['outputs']['metrics_path']}. "
        f"Skipped validation missing={metrics['skipped_validation_missing']}, "
        f"test missing={metrics['skipped_test_missing']}."
    )


if __name__ == "__main__":
    main()
