import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from keras_vggface_compat import patch_keras_for_vggface

from vggface_config import (
    CLASS_INDEX_PATH,
    CONFIDENCE_THRESHOLD,
    DETECTION_SCALE,
    INPUT_SIZE,
    MODEL_PATH,
    PROCESS_EVERY_N_FRAMES,
)


patch_keras_for_vggface()
from keras_vggface.utils import preprocess_input


def load_label_map(class_index_path: Path) -> Dict[int, str]:
    with class_index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(k): v for k, v in payload["index_to_class"].items()}


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    face_resized = cv2.resize(face_bgr, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)
    return preprocess_input(batch, version=1)


def detect_faces(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    detection_scale: float,
) -> List[Tuple[int, int, int, int]]:
    small = cv2.resize(frame, (0, 0), fx=detection_scale, fy=detection_scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
    )

    boxes = []
    inv = 1.0 / detection_scale
    for (x, y, w, h) in faces:
        x1 = int(x * inv)
        y1 = int(y * inv)
        x2 = int((x + w) * inv)
        y2 = int((y + h) * inv)
        boxes.append((x1, y1, x2, y2))
    return boxes


def predict_name(
    model: tf.keras.Model,
    label_map: Dict[int, str],
    face_bgr: np.ndarray,
    conf_threshold: float,
) -> Tuple[str, float]:
    batch = preprocess_face(face_bgr)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    if conf < conf_threshold:
        return "Unknown", conf
    return label_map.get(idx, "Unknown"), conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live VGGFace friend recognition.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--class-map", type=Path, default=CLASS_INDEX_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--detection-scale", type=float, default=DETECTION_SCALE)
    parser.add_argument("--process-every", type=int, default=PROCESS_EVERY_N_FRAMES)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Trained model not found: {args.model}")
    if not args.class_map.exists():
        raise FileNotFoundError(f"Class map not found: {args.class_map}")

    model = tf.keras.models.load_model(str(args.model), compile=False)
    label_map = load_label_map(args.class_map)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    frame_counter = 0
    draw_results: List[Tuple[Tuple[int, int, int, int], str, float]] = []
    last_t = time.time()
    fps = 0.0

    print("[live_webcam_vggface] Press 'q' to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_counter += 1
        if frame_counter % max(1, args.process_every) == 0:
            boxes = detect_faces(frame, face_cascade, args.detection_scale)
            draw_results = []
            for (x1, y1, x2, y2) in boxes:
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                face = frame[y1:y2, x1:x2]
                name, conf = predict_name(model, label_map, face, args.confidence)
                draw_results.append(((x1, y1, x2, y2), name, conf))

        current_t = time.time()
        dt = max(1e-6, current_t - last_t)
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = current_t

        for (x1, y1, x2, y2), name, conf in draw_results:
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({conf:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("VGGFace Live Recognition", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
