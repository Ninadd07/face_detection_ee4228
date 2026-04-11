from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import MatchResult


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


@dataclass
class EmbeddingGallery:
    identities: list[str]
    embeddings: np.ndarray

    @classmethod
    def from_templates(
        cls,
        templates: dict[str, list[np.ndarray]],
        aggregation: str = "mean",
    ) -> "EmbeddingGallery":
        if aggregation != "mean":
            raise ValueError(f"Unsupported aggregation mode: {aggregation}")
        identities: list[str] = []
        rows: list[np.ndarray] = []
        for identity, vectors in sorted(templates.items()):
            if not vectors:
                continue
            matrix = np.vstack(vectors).astype(np.float32)
            template = matrix.mean(axis=0, keepdims=True)
            identities.append(identity)
            rows.append(l2_normalize(template)[0])
        embeddings = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
        return cls(identities=identities, embeddings=embeddings)

    def match(self, embedding: np.ndarray, threshold: float, top_k: int = 3) -> MatchResult:
        if not self.identities:
            return MatchResult(identity=None, score=float("-inf"), accepted=False, top_k=[])
        vector = l2_normalize(embedding.astype(np.float32).reshape(1, -1))
        if self.embeddings.shape[1] != vector.shape[1]:
            raise ValueError(
                f"Gallery embedding dimension mismatch: gallery has {self.embeddings.shape[1]}, "
                f"query has {vector.shape[1]}"
            )
        scores = (self.embeddings @ vector.T).reshape(-1)
        order = np.argsort(scores)[::-1]
        top_entries = [(self.identities[idx], float(scores[idx])) for idx in order[:top_k]]
        best_identity, best_score = top_entries[0]
        accepted = best_score >= threshold
        return MatchResult(
            identity=best_identity if accepted else None,
            score=float(best_score),
            accepted=accepted,
            top_k=top_entries,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, identities=np.array(self.identities), embeddings=self.embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingGallery":
        payload = np.load(Path(path), allow_pickle=False)
        identities = [str(item) for item in payload["identities"].tolist()]
        return cls(identities=identities, embeddings=payload["embeddings"].astype(np.float32))
