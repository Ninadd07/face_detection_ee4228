from insightface.app import FaceAnalysis

from .config import MODEL_NAME, DET_SIZE, CTX_ID


class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(name=MODEL_NAME)
        self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)

    def detect(self, image):
        faces = self.app.get(image)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox.tolist()

            results.append({
                "bbox": (x1, y1, x2, y2),
                "face_obj": face,
                "embedding": getattr(face, "embedding", None),
                "det_score": float(getattr(face, "det_score", 0.0)),
            })

        return results
