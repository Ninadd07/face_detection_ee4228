# finetune_livedemo.py
import argparse
import time
import cv2

from .config import DEFAULT_THRESHOLD, UNKNOWN_LABEL
from .finetune_config import FINETUNE_EMBEDDINGS_DIR
from .detector import FaceDetector
from .finetune_embedder import FinetunedEmbedder
from .database import FaceDatabase
from .matcher import FaceMatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str,
                        default=str(FINETUNE_EMBEDDINGS_DIR / "finetuned_enroll.npz"))
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--detect-every", type=int, default=2)
    parser.add_argument("--window-name", type=str, default="ArcFace Finetuned Live")
    args = parser.parse_args()

    db = FaceDatabase.load(args.db)
    detector = FaceDetector()
    embedder = FinetunedEmbedder()
    matcher = FaceMatcher(db.labels, db.prototypes, args.threshold)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[finetune_livedemo] Cannot open camera")
        return

    print("[finetune_livedemo] Press q to quit.")

    frame_idx = 0
    frame_count = 0
    last_results = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[finetune_livedemo] Failed to read frame")
            break

        frame_idx += 1
        frame_count += 1

        if frame_idx % args.detect_every == 0:
            small = cv2.resize(frame, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_AREA) if args.scale != 1.0 else frame

            detections = detector.detect(small)
            current_results = []

            for det in detections:
                emb = embedder.get_embedding(small, det)
                if emb is None:
                    continue

                pred_label, score = matcher.match(emb)

                x1, y1, x2, y2 = det["bbox"]
                if args.scale != 1.0:
                    x1 = int(x1 / args.scale)
                    y1 = int(y1 / args.scale)
                    x2 = int(x2 / args.scale)
                    y2 = int(y2 / args.scale)

                current_results.append({
                    "bbox": (x1, y1, x2, y2),
                    "name": pred_label,
                    "score": float(score),
                })

            last_results = current_results

        for result in last_results:
            x1, y1, x2, y2 = result["bbox"]
            name = result["name"]
            score = result["score"]
            is_unknown = name == UNKNOWN_LABEL
            color = (0, 0, 255) if is_unknown else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {score:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            print(f"[finetune_livedemo] Approx FPS: {fps:.2f}")

        cv2.imshow(args.window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
