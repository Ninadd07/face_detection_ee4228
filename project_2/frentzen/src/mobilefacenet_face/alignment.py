from __future__ import annotations

import cv2
import numpy as np

from .sizes import normalize_input_size

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceAligner:
    def __init__(self, image_size: int | tuple[int, int] | list[int] = 112) -> None:
        self.image_size = normalize_input_size(image_size)

    def align(self, image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        if landmarks.shape != (5, 2):
            raise ValueError(f"Expected 5-point landmarks, got {landmarks.shape}.")
        output_height, output_width = self.image_size
        dst = ARCFACE_DST.copy()
        if output_width == 112:
            dst[:, 0] += 8.0
        else:
            dst[:, 0] *= float(output_width) / 96.0
        dst[:, 1] *= float(output_height) / 112.0
        transform = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst)[0]
        if transform is None:
            raise ValueError("Failed to estimate alignment transform.")
        return cv2.warpAffine(
            image_rgb,
            transform,
            (output_width, output_height),
            borderValue=0.0,
        )
