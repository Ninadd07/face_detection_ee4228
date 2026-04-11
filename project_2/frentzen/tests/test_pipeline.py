import numpy as np
import yaml
from pathlib import Path

from mobilefacenet_face.alignment import FaceAligner
from mobilefacenet_face.cli.enroll_gallery import build_gallery_from_manifest
from mobilefacenet_face.cli.evaluate_compact import _run_evaluation
from mobilefacenet_face.gallery import EmbeddingGallery
from mobilefacenet_face.pipeline import RecognitionPipeline, build_pipeline
from mobilefacenet_face.types import Detection, MatchResult, PipelineResult, SampleRecord


class DummyDetector:
    def detect(self, image_rgb):
        return [
            Detection(
                box=np.array([10, 10, 80, 80], dtype=np.float32),
                landmarks=np.array([[25, 30], [55, 30], [40, 45], [28, 62], [52, 62]], dtype=np.float32),
                confidence=0.99,
            )
        ]


class DummyEmbedder:
    def embed(self, aligned_faces):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_pipeline_runs_with_swappable_components():
    gallery = EmbeddingGallery.from_templates({"alice": [np.array([1.0, 0.0], dtype=np.float32)]})
    pipeline = RecognitionPipeline(
        detector=DummyDetector(),
        aligner=FaceAligner(112),
        embedder=DummyEmbedder(),
        gallery=gallery,
        threshold=0.1,
        top_k=1,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipeline.run(frame)
    assert len(result.predictions) == 1
    assert result.predictions[0].match.identity == "alice"


def test_build_pipeline_can_skip_loading_existing_gallery(tmp_path, monkeypatch):
    project_root = tmp_path
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = project_root / "artifacts" / "galleries" / "compact_gallery.npz"
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    EmbeddingGallery.from_templates({"alice": [np.array([1.0, 0.0], dtype=np.float32)]}).save(gallery_path)

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
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    class StubEmbedder:
        def __init__(self):
            self.input_size = (112, 96)

        def embed(self, aligned_faces):
            return np.zeros((len(aligned_faces), 128), dtype=np.float32)

    monkeypatch.setattr("mobilefacenet_face.pipeline.build_detector_and_aligner", lambda config: (DummyDetector(), FaceAligner((112, 96)), None))
    monkeypatch.setattr("mobilefacenet_face.pipeline.build_embedder", lambda config, checkpoint_path=None: StubEmbedder())

    pipeline = build_pipeline(config_path, load_gallery=False)
    assert pipeline.gallery is None


def test_build_gallery_from_manifest_skips_missing_sources(monkeypatch):
    records = [
        SampleRecord(identity="alice", path=Path("missing.jpg"), split="enrollment", group_id="g1"),
        SampleRecord(identity="alice", path=Path("ok.jpg"), split="enrollment", group_id="g2"),
    ]

    class StubPipeline:
        def run(self, frame):
            return PipelineResult(
                predictions=[
                    type(
                        "Prediction",
                        (),
                        {
                            "embedding": np.array([1.0, 0.0], dtype=np.float32),
                        },
                    )()
                ],
                timings_ms={"total_ms": 1.0},
            )

    monkeypatch.setattr(
        "mobilefacenet_face.cli.enroll_gallery.load_config",
        lambda config_path=None: {
            "outputs": {"split_manifest": "ignored.csv"},
            "matcher": {"aggregation": "mean"},
        },
    )
    monkeypatch.setattr("mobilefacenet_face.cli.enroll_gallery.read_manifest", lambda path: records)
    monkeypatch.setattr("mobilefacenet_face.cli.enroll_gallery.build_pipeline", lambda *args, **kwargs: StubPipeline())

    def fake_load_image(path):
        if str(path) == "missing.jpg":
            raise FileNotFoundError("missing")
        return np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr("mobilefacenet_face.cli.enroll_gallery.load_image", fake_load_image)

    config, gallery = build_gallery_from_manifest(None)
    assert gallery.identities == ["alice"]
    assert gallery.embeddings.shape == (1, 2)
    assert config["_enrollment_summary"]["skipped_missing"] == 1


def test_run_evaluation_skips_missing_sources(monkeypatch):
    records = [
        SampleRecord(identity="alice", path=Path("missing_validation.jpg"), split="validation", group_id="g1"),
        SampleRecord(identity="alice", path=Path("ok_validation.jpg"), split="validation", group_id="g2"),
        SampleRecord(identity="alice", path=Path("missing_test.jpg"), split="test", group_id="g3"),
        SampleRecord(identity="alice", path=Path("ok_test.jpg"), split="test", group_id="g4"),
    ]

    class StubPipeline:
        def __init__(self):
            self.threshold = 0.35
            self.gallery = None

        def run(self, frame):
            return PipelineResult(
                predictions=[
                    type(
                        "Prediction",
                        (),
                        {
                            "match": MatchResult(identity="alice", score=0.9, accepted=True, top_k=[("alice", 0.9)]),
                        },
                    )()
                ],
                timings_ms={"total_ms": 1.0},
            )

    monkeypatch.setattr(
        "mobilefacenet_face.cli.evaluate_compact.load_config",
        lambda config_path=None: {
            "outputs": {"split_manifest": "ignored.csv"},
            "recognizer": {"checkpoint_path": "stub.ckpt"},
        },
    )
    monkeypatch.setattr("mobilefacenet_face.cli.evaluate_compact.read_manifest", lambda path: records)
    monkeypatch.setattr(
        "mobilefacenet_face.cli.evaluate_compact.build_gallery_from_manifest",
        lambda config_path, checkpoint_path=None: (
            {"outputs": {"split_manifest": "ignored.csv"}, "recognizer": {"checkpoint_path": "stub.ckpt"}},
            EmbeddingGallery.from_templates({"alice": [np.array([1.0, 0.0], dtype=np.float32)]}),
        ),
    )
    monkeypatch.setattr("mobilefacenet_face.cli.evaluate_compact.build_pipeline", lambda *args, **kwargs: StubPipeline())

    def fake_load_image(path):
        if str(path).startswith("missing"):
            raise FileNotFoundError("missing")
        return np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr("mobilefacenet_face.cli.evaluate_compact.load_image", fake_load_image)

    metrics = _run_evaluation(None)
    assert metrics["skipped_validation_missing"] == 1
    assert metrics["skipped_test_missing"] == 1
    assert metrics["samples"] == 1
