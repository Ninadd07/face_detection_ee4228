import numpy as np

from .utils import normalize_embedding


class FaceDatabase:
    def __init__(self):
        self.labels = []
        self.prototypes = []

    def add_identity(self, label, embeddings):
        if not embeddings:
            return

        proto = np.mean(np.stack(embeddings, axis=0), axis=0)
        proto = normalize_embedding(proto)

        self.labels.append(label)
        self.prototypes.append(proto)

    def save(self, path):
        np.savez(
            path,
            labels=np.array(self.labels, dtype=object),
            prototypes=np.array(self.prototypes, dtype=np.float32),
        )

    @classmethod
    def load(cls, path):
        obj = cls()
        data = np.load(path, allow_pickle=True)
        obj.labels = list(data["labels"])
        obj.prototypes = data["prototypes"]
        return obj
