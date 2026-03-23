from .utils import normalize_embedding


class ArcFaceEmbedder:
    def __init__(self, mode="pretrained", checkpoint=None):
        self.mode = mode
        self.checkpoint = checkpoint

    def get_embedding_from_detection(self, detection):
        emb = detection.get("embedding", None)
        if emb is None:
            return None
        return normalize_embedding(emb)
