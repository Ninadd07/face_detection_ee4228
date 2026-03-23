import numpy as np

from .config import UNKNOWN_LABEL


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class FaceMatcher:
    def __init__(self, labels, prototypes, threshold, unknown_label=UNKNOWN_LABEL):
        self.labels = labels
        self.prototypes = prototypes
        self.threshold = threshold
        self.unknown_label = unknown_label

    def match(self, embedding):
        scores = [cosine_similarity(embedding, p) for p in self.prototypes]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.threshold:
            return self.labels[best_idx], best_score

        return self.unknown_label, best_score
