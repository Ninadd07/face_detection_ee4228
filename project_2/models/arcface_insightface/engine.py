"""
engine.py

ArcFaceEngine:
- wraps InsightFace FaceAnalysis (SCRFD + ArcFace)
- loads/saves embeddings and builds per-person prototypes
- exposes a simple API for GUI / scripts:
    - recognize_frame(frame) -> list of {bbox, name, score}
    - register_person_from_images(person_name, image_paths)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from .config import (
    AUG_IMG_ROOT,
    EMB_ROOT,
    DET_SIZE,
    PROVIDERS,
    ARC_FACE_INPUT_SIZE,
    PROTOTYPE_THRESHOLD,
    UNKNOWN_LABEL,
    DEBUG,
)


class ArcFaceEngine:
    """
    High-level face recognition engine for the ArcFace + InsightFace model.

    Responsibilities:
    - Initialize InsightFace FaceAnalysis
    - Load existing embeddings from disk and build prototypes
    - Provide recognize_frame() for live recognition
    - Provide methods to add / update persons from image sets
    """

    def __init__(self, threshold: float = PROTOTYPE_THRESHOLD) -> None:
        self.threshold = threshold

        # 1. Initialize InsightFace FaceAnalysis (detection + embeddings)
        self.app = FaceAnalysis(providers=PROVIDERS)
        self.app.prepare(ctx_id=-1, det_size=DET_SIZE)

        # 2. In-memory database
        #    person_to_embs: { "jessica": np.ndarray (N, 512), ... }
        #    prototypes:    { "jessica": np.ndarray (512,), ... }
        self.person_to_embs: Dict[str, np.ndarray] = {}
        self.prototypes: Dict[str, np.ndarray] = {}

        # 3. Ensure embedding root exists
        EMB_ROOT.mkdir(parents=True, exist_ok=True)

        # 4. Load embeddings from disk and build prototypes
        self._load_database()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Run face detection + recognition on a single BGR frame.

        Returns a list of dicts:
        [
          {
            "bbox": (x1, y1, x2, y2),
            "name": str,           # person name or UNKNOWN_LABEL
            "score": float,        # cosine similarity to prototype
          },
          ...
        ]
        """
        faces = self.app.get(frame)
        results: List[Dict] = []

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box

            emb = face.normed_embedding  # shape: (512,)
            name, score = self._predict_person_prototype(emb)

            results.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "name": name,
                    "score": float(score),
                }
            )

        return results

    def register_person_from_images(
        self,
        person_name: str,
        image_paths: List[Path],
    ) -> None:
        """
        Build / update embeddings for a person from a list of image paths.

        Typical usage:
        - Called by an offline script that passes augmented image paths, OR
        - Called by a GUI after capturing ~10 frames and saving them as images.

        This method:
        - Computes embeddings for those images
        - Appends them to that person's .npy file(s)
        - Updates in-memory person_to_embs and prototypes
        """
        if not image_paths:
            if DEBUG:
                print(f"[ArcFaceEngine] No images provided for {person_name}")
            return

        # Compute embeddings for all given images
        new_embs = []

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                if DEBUG:
                    print(f"[ArcFaceEngine] Warning: cannot read {img_path}")
                continue

            # Optional: resize to 112x112 if you feed directly to embedding model.
            # Here we rely on FaceAnalysis (which expects a full frame),
            # so we simply pass the cropped face as a small frame.
            img_resized = cv2.resize(
                img,
                ARC_FACE_INPUT_SIZE,
                interpolation=cv2.INTER_AREA,
            )

            faces = self.app.get(img_resized)
            if len(faces) != 1:
                if DEBUG:
                    print(
                        f"[ArcFaceEngine] Skipping {img_path}, "
                        f"detected faces: {len(faces)}"
                    )
                continue

            emb = faces[0].normed_embedding
            new_embs.append(emb)

        if not new_embs:
            if DEBUG:
                print(f"[ArcFaceEngine] No valid embeddings for {person_name}")
            return

        new_embs = np.vstack(new_embs)  # (K, 512)

        # Save / append to disk
        person_dir = EMB_ROOT / person_name
        person_dir.mkdir(parents=True, exist_ok=True)

        # Save a new file with incremental index
        existing_files = sorted(person_dir.glob("emb_*.npy"))
        next_idx = len(existing_files) + 1
        out_path = person_dir / f"emb_{next_idx:03d}.npy"
        np.save(out_path, new_embs)

        if DEBUG:
            print(
                f"[ArcFaceEngine] Saved {new_embs.shape} embeddings to {out_path}"
            )

        # Update in-memory DB
        self._load_person_embeddings(person_name)
        self._build_prototype_for_person(person_name)

    def reload_database(self) -> None:
        """
        Re-load all embeddings from disk and rebuild prototypes.
        Call this if embeddings on disk change externally.
        """
        self._load_database()

    # ------------------------------------------------------------------
    # Internal: database loading / prototypes
    # ------------------------------------------------------------------

    def _load_database(self) -> None:
        """Load all persons' embeddings from EMB_ROOT and build prototypes."""
        self.person_to_embs.clear()
        self.prototypes.clear()

        if not EMB_ROOT.exists():
            return

        for person_dir in sorted(EMB_ROOT.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            self._load_person_embeddings(person_name)
            self._build_prototype_for_person(person_name)

        if DEBUG:
            print(
                "[ArcFaceEngine] Loaded persons:",
                list(self.person_to_embs.keys()),
            )

    def _load_person_embeddings(self, person_name: str) -> None:
        """
        Load all .npy files for one person into self.person_to_embs[person_name].
        """
        person_dir = EMB_ROOT / person_name
        if not person_dir.exists():
            return

        all_embs = []

        for npy_path in sorted(person_dir.glob("*.npy")):
            embs = np.load(str(npy_path))
            if embs.ndim == 1:
                embs = embs[None, :]
            all_embs.append(embs)

        if not all_embs:
            return

        all_embs = np.vstack(all_embs)  # (N, 512)
        self.person_to_embs[person_name] = all_embs

        if DEBUG:
            print(
                f"[ArcFaceEngine] {person_name}: loaded {all_embs.shape[0]} "
                f"embeddings from {person_dir}"
            )

    def _build_prototype_for_person(self, person_name: str) -> None:
        """Compute mean embedding for one person and store in self.prototypes."""
        embs = self.person_to_embs.get(person_name)
        if embs is None or len(embs) == 0:
            return

        proto = np.mean(embs, axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        self.prototypes[person_name] = proto

    # ------------------------------------------------------------------
    # Internal: similarity + prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1D vectors."""
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    def _predict_person_prototype(
        self,
        emb: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Predict identity using prototype-per-person matching.

        Returns:
            (best_name, best_score)
            best_name is UNKNOWN_LABEL if best_score < self.threshold
        """
        if not self.prototypes:
            return UNKNOWN_LABEL, 0.0

        emb = emb / (np.linalg.norm(emb) + 1e-8)

        best_name = UNKNOWN_LABEL
        best_score = -1.0

        for name, proto in self.prototypes.items():
            score = self._cosine_similarity(emb, proto)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self.threshold:
            return UNKNOWN_LABEL, best_score

        return best_name, best_score
