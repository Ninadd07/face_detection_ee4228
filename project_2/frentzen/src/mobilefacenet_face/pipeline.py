from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .alignment import FaceAligner
from .config import load_config
from .detectors import MTCNNDetector
from .gallery import EmbeddingGallery
from .recognizers import MobileFaceNetEmbedder
from .sizes import normalize_input_size
from .types import FacePrediction, MatchResult, PipelineResult


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_detector_and_aligner(config: dict) -> tuple[MTCNNDetector, FaceAligner, torch.device]:
    device = resolve_device(config.get("device", "auto"))
    detector_cfg = config["detector"]
    recognizer_cfg = config["recognizer"]
    detector = MTCNNDetector(
        device=device,
        image_size=detector_cfg.get("image_size", 160),
        margin=detector_cfg.get("margin", 0),
        min_face_size=detector_cfg.get("min_face_size", 20),
        thresholds=tuple(detector_cfg.get("thresholds", [0.6, 0.7, 0.7])),
        factor=detector_cfg.get("factor", 0.709),
        post_process=detector_cfg.get("post_process", False),
    )
    aligner = FaceAligner(image_size=normalize_input_size(recognizer_cfg.get("input_size", [112, 96])))
    return detector, aligner, device


def build_embedder(config: dict, checkpoint_path: str | Path | None = None) -> MobileFaceNetEmbedder:
    _, _, device = build_detector_and_aligner(config)
    recognizer_cfg = config["recognizer"]
    active_checkpoint = str(checkpoint_path) if checkpoint_path is not None else recognizer_cfg.get("checkpoint_path")
    return MobileFaceNetEmbedder(
        device=device,
        checkpoint_path=active_checkpoint,
        embedding_dim=recognizer_cfg.get("embedding_dim", 128),
        input_size=normalize_input_size(recognizer_cfg.get("input_size", [112, 96])),
    )


class RecognitionPipeline:
    def __init__(
        self,
        detector,
        aligner: FaceAligner,
        embedder,
        gallery: EmbeddingGallery | None,
        threshold: float,
        top_k: int,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.gallery = gallery
        self.threshold = threshold
        self.top_k = top_k

    def run(self, frame_bgr: np.ndarray) -> PipelineResult:
        total_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detect_start = time.perf_counter()
        detections = self.detector.detect(frame_rgb)
        detect_ms = (time.perf_counter() - detect_start) * 1000.0

        align_start = time.perf_counter()
        aligned_faces = [self.aligner.align(frame_rgb, detection.landmarks) for detection in detections]
        align_ms = (time.perf_counter() - align_start) * 1000.0

        embed_start = time.perf_counter()
        embeddings = self.embedder.embed(aligned_faces)
        embed_ms = (time.perf_counter() - embed_start) * 1000.0

        match_start = time.perf_counter()
        predictions: list[FacePrediction] = []
        for detection, aligned_face, embedding in zip(detections, aligned_faces, embeddings):
            match = (
                self.gallery.match(embedding, threshold=self.threshold, top_k=self.top_k)
                if self.gallery is not None
                else MatchResult(identity=None, score=float("-inf"), accepted=False, top_k=[])
            )
            predictions.append(
                FacePrediction(
                    detection=detection,
                    aligned_face=aligned_face,
                    embedding=embedding,
                    match=match,
                    timings_ms={},
                )
            )
        match_ms = (time.perf_counter() - match_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0
        per_face_ms = embed_ms / max(1, len(predictions))
        for prediction in predictions:
            prediction.timings_ms.update({"per_face_embed_ms": per_face_ms, "total_frame_ms": total_ms})
        return PipelineResult(
            predictions=predictions,
            timings_ms={
                "detect_ms": detect_ms,
                "align_ms": align_ms,
                "embed_ms": embed_ms,
                "match_ms": match_ms,
                "total_ms": total_ms,
            },
        )


def build_pipeline(
    config_path: str | Path | None,
    gallery_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    load_gallery: bool = True,
) -> RecognitionPipeline:
    config = load_config(config_path)
    matcher_cfg = config["matcher"]
    detector, aligner, _ = build_detector_and_aligner(config)
    embedder = build_embedder(config, checkpoint_path=checkpoint_path)
    gallery = None
    if load_gallery:
        active_gallery_path = Path(gallery_path or config["outputs"]["gallery_path"])
        gallery = EmbeddingGallery.load(active_gallery_path) if active_gallery_path.exists() else None
    return RecognitionPipeline(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        gallery=gallery,
        threshold=matcher_cfg.get("threshold", 0.35),
        top_k=matcher_cfg.get("top_k", 3),
    )
