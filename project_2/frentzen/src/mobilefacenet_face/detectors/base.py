from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Detection


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        raise NotImplementedError
