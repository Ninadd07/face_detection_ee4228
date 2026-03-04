import cv2
import numpy as np
from insightface.app import FaceAnalysis
from recognition_utils import load_person_embeddings, cosine_similarity

def build_prototypes(person_to_embs: dict) -> dict:
    """
    Compute one prototype (mean embedding) per person.
    Returns dict: { name: 512-dim vector }.
    """
    prototypes = {}
    for name, embs in person_to_embs.items():
        # embs: (N, 512)
        proto = np.mean(embs, axis=0)
        # re-normalize to unit length for cosine similarity
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes[name] = proto
    return prototypes

def predict_person_prototype(embedding, prototypes, threshold=0.7):
    """
    Prototype-based matching (main method).
    embedding: (512,)
    prototypes: { name: (512,) }
    Returns: (best_name, best_score)
    """
    best_name = "unknown"
    best_score = -1.0

    # ensure embedding is normalized
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

    for name, proto in prototypes.items():
        score = cosine_similarity(embedding, proto)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return "unknown", best_score
    return best_name, best_score

# ------------------------------------------------------------------
# OPTIONAL: Original all-samples matching approach (kept for reference)
#
# from recognition_utils import predict_person
#
# def predict_person_all_samples(embedding, person_to_embs, threshold=0.6):
#     """
#     Wrapper around the original all-samples method.
#     Uses cosine similarity and average similarity per person.
#     """
#     return predict_person(embedding, person_to_embs,
#                           threshold=threshold, metric="cosine")
# ------------------------------------------------------------------


def main():
    # 1. Load all embeddings (per person)
    person_to_embs = load_person_embeddings(emb_root="embeddings")

    if not person_to_embs:
        print("[ERROR] No embeddings loaded. Check your embeddings/ folder.")
        return

    # 2. Build prototypes (mean embedding per person)
    prototypes = build_prototypes(person_to_embs)
    print("[INFO] Prototypes built for:", list(prototypes.keys()))

    # 3. Initialize InsightFace
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # 4. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        faces = app.get(frame)

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box

            emb = face.normed_embedding  # (512,)

            # --- MAIN METHOD: prototype-based ---
            name, score = predict_person_prototype(
                emb, prototypes, threshold=0.6
            )

            # --- OPTIONAL: all-samples method (commented) ---
            # name, score = predict_person_all_samples(
            #     emb, person_to_embs, threshold=0.6
            # )

            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Live Recognition (prototype) - press q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
