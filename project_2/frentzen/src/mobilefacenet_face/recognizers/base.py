from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FaceEmbedder(ABC):
    @abstractmethod
    def embed(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError
