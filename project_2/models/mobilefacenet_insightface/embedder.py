"""
MobileFaceNet embedder for the mobilefacenet_insightface evaluation pipeline.

Requires mobilefacenet_face to be installed:
    pip install -e <repo_root>/frentzen/frentzen
See README.md for full setup instructions.
"""
import cv2
import numpy as np
import torch

from .config import MFN_DEFAULT_CHECKPOINT

try:
    from mobilefacenet_face.recognizers.mobilefacenet import MobileFaceNetEmbedder as _MFNEmbedder
    from mobilefacenet_face.alignment import FaceAligner
except ImportError as exc:
    raise ImportError(
        "mobilefacenet_face is not installed.\n"
        "Run:  pip install -e <repo_root>/frentzen/frentzen\n"
        "See mobilefacenet_insightface/README.md for full setup instructions."
    ) from exc

_INPUT_SIZE = (112, 96)  # (height, width) expected by MobileFaceNet


class MobileFaceNetEmbedder:
    """Wraps frentzen's MobileFaceNetEmbedder with the InsightFace detection interface.

    Parameters
    ----------
    checkpoint_path : str | Path | None
        Path to a .pth checkpoint file. Defaults to MFN_DEFAULT_CHECKPOINT
        (the fine-tuned mobilefacenet_best.pth, or the MFN_CHECKPOINT env var).

    Interface
    ---------
    get_embedding(image_bgr, detection) -> np.ndarray | None
        image_bgr  : BGR uint8 ndarray — full frame as returned by cv2.imread
        detection  : dict from FaceDetector.detect(); must contain 'face_obj'
                     with a .kps attribute of shape (5, 2).
        Returns a 128-dim L2-normalised float32 ndarray, or None on any failure.
    """

    def __init__(self, checkpoint_path=None):
        checkpoint = str(checkpoint_path or MFN_DEFAULT_CHECKPOINT)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._embedder = _MFNEmbedder(
            device=device,
            checkpoint_path=checkpoint,
            embedding_dim=128,
            input_size=_INPUT_SIZE,
        )
        self._aligner = FaceAligner(image_size=_INPUT_SIZE)

    def get_embedding(self, image_bgr: np.ndarray, detection: dict) -> np.ndarray | None:
        face_obj = detection.get("face_obj")
        if face_obj is None:
            return None

        kps = getattr(face_obj, "kps", None)
        if kps is None:
            return None

        landmarks = np.array(kps, dtype=np.float32)
        if landmarks.shape != (5, 2):
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            aligned = self._aligner.align(image_rgb, landmarks)
        except ValueError:
            return None

        # embed() returns ndarray (N, 128), already L2-normalised
        return self._embedder.embed([aligned])[0]
