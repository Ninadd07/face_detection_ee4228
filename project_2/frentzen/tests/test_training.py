import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from mobilefacenet_face.preprocessing import prepare_aligned_faces
from mobilefacenet_face.preprocessing import prune_failed_source_images
from mobilefacenet_face.recognizers.mobilefacenet import MobileFaceNet, MobileFaceNetEmbedder, save_backbone_checkpoint
from mobilefacenet_face.training import train_mobilefacenet
from mobilefacenet_face.types import Detection


def _write_config(project_root: Path, overrides: dict) -> Path:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "name": "test-mobilefacenet",
        "device": "cpu",
        "detector": {
            "type": "mtcnn",
            "image_size": 160,
            "margin": 0,
            "min_face_size": 20,
            "thresholds": [0.6, 0.7, 0.7],
            "factor": 0.709,
            "post_process": False,
        },
        "recognizer": {
            "type": "mobilefacenet",
            "embedding_dim": 128,
            "input_size": [112, 96],
            "checkpoint_path": None,
        },
        "preprocess": {
            "cache_aligned_faces": True,
            "aligned_dir": "artifacts/aligned_faces",
            "aligned_manifest": "artifacts/aligned_faces/aligned_manifest.csv",
            "overwrite": False,
        },
        "matcher": {"threshold": 0.35, "top_k": 3, "aggregation": "mean"},
        "data": {
            "root_dir": "training_images_augmented",
            "split_seed": 42,
            "enrollment_ratio": 0.5,
            "validation_ratio": 0.25,
            "test_ratio": 0.25,
            "group_pattern": r"(vid\d+_frame\d+)",
        },
        "training": {
            "output_dir": "artifacts/checkpoints",
            "best_checkpoint_path": "artifacts/checkpoints/mobilefacenet_best.pth",
            "last_checkpoint_path": "artifacts/checkpoints/mobilefacenet_last.pth",
            "training_state_path": "artifacts/checkpoints/training_state.json",
            "training_metrics_path": "artifacts/reports/training_metrics.json",
            "random_seed": 42,
            "batch_size": 2,
            "epochs": 1,
            "warmup_epochs": 1,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "early_stopping_patience": 2,
            "num_workers": 0,
            "label_smoothing": 0.0,
            "max_samples_per_group": 2,
            "selection_metric": "val_accuracy",
            "augmentation": {
                "horizontal_flip": False,
                "color_jitter": False,
                "random_erasing": False,
            },
        },
        "outputs": {
            "split_manifest": "artifacts/splits/compact_splits.csv",
            "gallery_path": "artifacts/galleries/compact_gallery.npz",
            "metrics_path": "artifacts/reports/compact_metrics.json",
            "comparison_metrics_path": "artifacts/reports/compact_comparison_metrics.json",
            "benchmark_path": "artifacts/reports/compact_benchmark.json",
            "prediction_log_path": "artifacts/reports/live_predictions.jsonl",
            "failed_sources_path": "artifacts/reports/aligned_failures.csv",
            "failed_contact_sheet_path": "artifacts/reports/aligned_failures_contact_sheet.jpg",
        },
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(config.get(section), dict):
            config[section].update(values)
        else:
            config[section] = values
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_checkpoint_roundtrip(tmp_path):
    model = MobileFaceNet()
    checkpoint_path = tmp_path / "mobilefacenet_roundtrip.pth"
    save_backbone_checkpoint(model, checkpoint_path, epoch=1, metrics={"val_accuracy": 1.0})
    embedder = MobileFaceNetEmbedder(device=torch.device("cpu"), checkpoint_path=str(checkpoint_path))
    assert embedder.checkpoint_size_bytes() is not None


def test_prepare_aligned_faces_with_cached_output(tmp_path, monkeypatch):
    project_root = tmp_path
    image_path = project_root / "sample.jpg"
    cv2.imwrite(str(image_path), np.zeros((160, 160, 3), dtype=np.uint8))

    manifest_path = project_root / "artifacts" / "splits" / "compact_splits.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["identity", "path", "split", "group_id"])
        writer.writeheader()
        writer.writerow(
            {
                "identity": "alice",
                "path": str(image_path),
                "split": "enrollment",
                "group_id": "vid0_frame00001",
            }
        )

    config_path = _write_config(project_root, {})

    class DummyDetector:
        def detect(self, image_rgb):
            return [
                Detection(
                    box=np.array([0, 0, 100, 100], dtype=np.float32),
                    landmarks=np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32),
                    confidence=0.99,
                )
            ]

    class DummyAligner:
        def align(self, image_rgb, landmarks):
            return np.zeros((112, 96, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "mobilefacenet_face.preprocessing.build_detector_and_aligner",
        lambda config: (DummyDetector(), DummyAligner(), torch.device("cpu")),
    )

    summary = prepare_aligned_faces(config_path)
    assert summary.prepared == 1
    summary_second = prepare_aligned_faces(config_path)
    assert summary_second.prepared == 1
    aligned_manifest = project_root / "artifacts" / "aligned_faces" / "aligned_manifest.csv"
    assert aligned_manifest.exists()
    assert (project_root / "artifacts" / "reports" / "aligned_failures.csv").exists()


def test_prepare_aligned_faces_writes_failure_contact_sheet(tmp_path, monkeypatch):
    project_root = tmp_path
    image_path = project_root / "sample.jpg"
    cv2.imwrite(str(image_path), np.zeros((160, 160, 3), dtype=np.uint8))

    manifest_path = project_root / "artifacts" / "splits" / "compact_splits.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["identity", "path", "split", "group_id"])
        writer.writeheader()
        writer.writerow(
            {
                "identity": "alice",
                "path": str(image_path),
                "split": "enrollment",
                "group_id": "vid0_frame00001",
            }
        )

    config_path = _write_config(project_root, {})

    class DummyDetector:
        def detect(self, image_rgb):
            return []

    class DummyAligner:
        def align(self, image_rgb, landmarks):
            raise AssertionError("align should not be called when detection fails")

    monkeypatch.setattr(
        "mobilefacenet_face.preprocessing.build_detector_and_aligner",
        lambda config: (DummyDetector(), DummyAligner(), torch.device("cpu")),
    )

    summary = prepare_aligned_faces(config_path)
    assert summary.failed == 1
    failure_report = project_root / "artifacts" / "reports" / "aligned_failures.csv"
    failure_sheet = project_root / "artifacts" / "reports" / "aligned_failures_contact_sheet.jpg"
    assert failure_report.exists()
    assert failure_sheet.exists()


def test_prune_failed_source_images_dry_run_and_apply(tmp_path):
    report_path = tmp_path / "aligned_failures.csv"
    source_path = tmp_path / "failed.jpg"
    cv2.imwrite(str(source_path), np.zeros((8, 8, 3), dtype=np.uint8))
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["identity", "source_path", "aligned_path", "split", "group_id", "confidence", "success", "error"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "identity": "alice",
                "source_path": str(source_path),
                "aligned_path": "",
                "split": "enrollment",
                "group_id": "vid0_frame00001",
                "confidence": "",
                "success": "false",
                "error": "No face detected.",
            }
        )

    dry_run = prune_failed_source_images(report_path, apply=False)
    assert dry_run["existing_sources"] == 1
    assert dry_run["deleted_sources"] == 0
    assert source_path.exists()

    applied = prune_failed_source_images(report_path, apply=True)
    assert applied["deleted_sources"] == 1
    assert not source_path.exists()


def test_train_mobilefacenet_smoke(tmp_path):
    project_root = tmp_path
    aligned_dir = project_root / "artifacts" / "aligned_faces"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("enrollment", "validation"):
        for identity, value in (("alice", 32), ("bob", 96)):
            count = 2 if split == "enrollment" else 1
            for idx in range(count):
                image = np.full((112, 96, 3), value + idx, dtype=np.uint8)
                aligned_path = aligned_dir / f"{split}_{identity}_{idx}.jpg"
                cv2.imwrite(str(aligned_path), image)
                rows.append(
                    {
                        "identity": identity,
                        "source_path": str(aligned_path),
                        "aligned_path": str(aligned_path),
                        "split": split,
                        "group_id": f"{identity}_group_{idx}",
                        "confidence": "0.99",
                        "success": "true",
                        "error": "",
                    }
                )

    aligned_manifest = aligned_dir / "aligned_manifest.csv"
    with aligned_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["identity", "source_path", "aligned_path", "split", "group_id", "confidence", "success", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    pretrained_checkpoint = project_root / "pretrained_mobilefacenet.pth"
    save_backbone_checkpoint(MobileFaceNet(), pretrained_checkpoint, epoch=0, metrics={})
    config_path = _write_config(
        project_root,
        {
            "recognizer": {"checkpoint_path": str(pretrained_checkpoint)},
            "preprocess": {"aligned_manifest": str(aligned_manifest)},
        },
    )

    result = train_mobilefacenet(config_path)
    assert Path(result["best_checkpoint_path"]).exists()
    assert Path(result["last_checkpoint_path"]).exists()
    assert (project_root / "artifacts" / "reports" / "training_metrics.json").exists()


def test_train_mobilefacenet_accepts_auto_device(tmp_path):
    project_root = tmp_path
    aligned_dir = project_root / "artifacts" / "aligned_faces"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("enrollment", "validation"):
        for identity, value in (("alice", 32), ("bob", 96)):
            count = 2 if split == "enrollment" else 1
            for idx in range(count):
                image = np.full((112, 96, 3), value + idx, dtype=np.uint8)
                aligned_path = aligned_dir / f"{split}_{identity}_{idx}.jpg"
                cv2.imwrite(str(aligned_path), image)
                rows.append(
                    {
                        "identity": identity,
                        "source_path": str(aligned_path),
                        "aligned_path": str(aligned_path),
                        "split": split,
                        "group_id": f"{identity}_group_{idx}",
                        "confidence": "0.99",
                        "success": "true",
                        "error": "",
                    }
                )

    aligned_manifest = aligned_dir / "aligned_manifest.csv"
    with aligned_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["identity", "source_path", "aligned_path", "split", "group_id", "confidence", "success", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    pretrained_checkpoint = project_root / "pretrained_mobilefacenet.pth"
    save_backbone_checkpoint(MobileFaceNet(), pretrained_checkpoint, epoch=0, metrics={})
    config_path = _write_config(
        project_root,
        {
            "device": "auto",
            "recognizer": {"checkpoint_path": str(pretrained_checkpoint)},
            "preprocess": {"aligned_manifest": str(aligned_manifest)},
        },
    )

    result = train_mobilefacenet(config_path)
    assert Path(result["best_checkpoint_path"]).exists()
