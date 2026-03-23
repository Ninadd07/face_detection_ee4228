"""
webcam_live.py

Standalone demo for ArcFaceEngine:
- Opens the webcam
- Runs face detection + recognition on each frame
- Draws bounding boxes and labels (name + score)
- Quits when 'q' is pressed

Usage (from project root, with venv active):
    python -m models.arcface_insightface.webcam_live
"""

import cv2

from .engine import ArcFaceEngine
from .config import PROTOTYPE_THRESHOLD, UNKNOWN_LABEL


def main() -> None:
    # Initialize engine (loads embeddings_arcface and builds prototypes)
    engine = ArcFaceEngine(threshold=PROTOTYPE_THRESHOLD)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[webcam_live] Cannot open camera")
        return

    print("[webcam_live] Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[webcam_live] Failed to grab frame")
            break

        # Run recognition on this frame
        results = engine.recognize_frame(frame)

        # Draw results
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            name = r["name"]
            score = r["score"]

            is_unknown = (name == UNKNOWN_LABEL)
            color = (0, 0, 255) if is_unknown else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} ({score:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("ArcFace Live Recognition - press q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
