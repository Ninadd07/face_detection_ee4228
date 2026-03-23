from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis


VIDEO_ROOT = Path("/Users/jessica/Documents/GitHub/face_detection_ee4228/data_raw/videos")   # CHANGE THIS
IMAGE_ROOT = Path("/Users/jessica/Documents/GitHub/face_detection_ee4228/data_raw/images")   # CHANGE THIS

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
CTX_ID = 0  # use -1 if CPU only

SAMPLE_EVERY_N_FRAMES = 10
MAX_SAVED_PER_VIDEO = 100

SAVE_FACE_CROP = True      # True = save padded face crop, False = save full frame
FACE_PADDING = 0.70        # 70% padding around detected face
OUTPUT_FACE_SIZE = 200     # saved crop size: 200x200

BLUR_THRESHOLD = 0.0      # lower = more tolerant, higher = stricter


def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def make_detector():
    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)
    return app


def clamp(val, lo, hi):
    return max(lo, min(val, hi))


def padded_crop(image, bbox, padding_ratio=0.70):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding_ratio / 2.0)
    pad_y = int(bh * padding_ratio / 2.0)

    nx1 = clamp(x1 - pad_x, 0, w)
    ny1 = clamp(y1 - pad_y, 0, h)
    nx2 = clamp(x2 + pad_x, 0, w)
    ny2 = clamp(y2 + pad_y, 0, h)

    crop = image[ny1:ny2, nx1:nx2]
    return crop


def process_video(video_path: Path, out_dir: Path, detector):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[extract_clean_frames] Cannot open {video_path}")
        return {
            "saved": 0,
            "sampled": 0,
            "multi_face": 0,
            "no_face": 0,
            "blurry": 0,
        }

    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    sampled = 0
    multi_face = 0
    no_face = 0
    blurry = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue

        sampled += 1

        faces = detector.get(frame)

        if len(faces) == 0:
            no_face += 1
            frame_idx += 1
            continue

        if len(faces) != 1:
            multi_face += 1
            frame_idx += 1
            continue

        face = faces[0]
        crop = padded_crop(frame, face.bbox, padding_ratio=FACE_PADDING)

        if crop.size == 0:
            frame_idx += 1
            continue

        blur_score = variance_of_laplacian(crop)
        if blur_score < BLUR_THRESHOLD:
            blurry += 1
            frame_idx += 1
            continue

        if SAVE_FACE_CROP:
            save_img = cv2.resize(crop, (OUTPUT_FACE_SIZE, OUTPUT_FACE_SIZE), interpolation=cv2.INTER_AREA)
        else:
            save_img = frame

        out_name = f"{video_path.stem}_frame{frame_idx:06d}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), save_img)

        saved += 1
        if MAX_SAVED_PER_VIDEO is not None and saved >= MAX_SAVED_PER_VIDEO:
            break

        frame_idx += 1

    cap.release()

    return {
        "saved": saved,
        "sampled": sampled,
        "multi_face": multi_face,
        "no_face": no_face,
        "blurry": blurry,
    }


def main():
    if not VIDEO_ROOT.exists():
        raise FileNotFoundError(f"VIDEO_ROOT not found: {VIDEO_ROOT}")

    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    detector = make_detector()

    for person_dir in sorted(VIDEO_ROOT.iterdir()):
        if not person_dir.is_dir():
            continue

        out_dir = IMAGE_ROOT / person_dir.name

        person_saved = 0
        person_sampled = 0
        person_multi = 0
        person_no_face = 0
        person_blurry = 0

        for video_path in sorted(person_dir.iterdir()):
            if video_path.suffix.lower() not in VIDEO_EXTS:
                continue

            stats = process_video(video_path, out_dir, detector)

            person_saved += stats["saved"]
            person_sampled += stats["sampled"]
            person_multi += stats["multi_face"]
            person_no_face += stats["no_face"]
            person_blurry += stats["blurry"]

            print(
                f"[extract_clean_frames] {person_dir.name} | {video_path.name}: "
                f"sampled={stats['sampled']}, saved={stats['saved']}, "
                f"no_face={stats['no_face']}, multi_face={stats['multi_face']}, blurry={stats['blurry']}"
            )

        print(
            f"[extract_clean_frames] {person_dir.name} TOTAL: "
            f"sampled={person_sampled}, saved={person_saved}, "
            f"no_face={person_no_face}, multi_face={person_multi}, blurry={person_blurry}"
        )

    print("[extract_clean_frames] Done.")


if __name__ == "__main__":
    main()
