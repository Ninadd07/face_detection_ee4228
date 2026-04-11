from __future__ import annotations

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from ..types import Detection
from .base import FaceDetector


class MTCNNDetector(FaceDetector):
    def __init__(
        self,
        device: torch.device,
        image_size: int = 160,
        margin: int = 0,
        min_face_size: int = 20,
        thresholds: tuple[float, float, float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
        post_process: bool = False,
    ) -> None:
        self.detector = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            post_process=post_process,
            keep_all=True,
            device=device,
        )

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        pil_image = Image.fromarray(image_rgb)
        boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)
        detections: list[Detection] = []
        if boxes is None or probs is None or landmarks is None:
            return detections
        for box, prob, points in zip(boxes, probs, landmarks):
            if box is None or prob is None or points is None:
                continue
            detections.append(
                Detection(
                    box=np.asarray(box, dtype=np.float32),
                    landmarks=np.asarray(points, dtype=np.float32),
                    confidence=float(prob),
                )
            )
        return detections
