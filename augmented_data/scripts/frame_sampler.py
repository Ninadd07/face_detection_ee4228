"""
Stage 1: Frame Sampling & Face Cropping

Reads videos from Training_Videos/<Person>/*.mp4, detects faces using InsightFace,
applies blur filtering, crops & aligns faces, and saves them as JPEG images.

Output: augmented_data/images/<Person>/raw_vid{i}_frame{n}.jpg
"""

import os
import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import (
    VIDEO_ROOT,
    IMAGE_OUTPUT_ROOT,
    GPU_ID,
    INSIGHTFACE_MODEL,
    DET_SIZE,
    RAW_IMAGES_PER_PERSON,
    BLUR_THRESHOLD,
    FACE_CONFIDENCE_THRESHOLD,
    FACE_CROP_SIZE,
    FACE_PADDING,
    JPEG_QUALITY,
    VIDEO_EXTENSIONS,
)


def is_blurry(image: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    """Check if an image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def crop_face_with_padding(
    frame: np.ndarray, bbox: np.ndarray, padding: float = FACE_PADDING
) -> np.ndarray:
    """
    Crop a face from the frame with extra padding around the bounding box.
    Returns the cropped and resized face image.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)

    # Calculate padding
    face_w = x2 - x1
    face_h = y2 - y1
    pad_w = int(face_w * (padding - 1) / 2)
    pad_h = int(face_h * (padding - 1) / 2)

    # Apply padding with boundary checks
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    # Resize to target size
    face_crop = cv2.resize(face_crop, FACE_CROP_SIZE, interpolation=cv2.INTER_AREA)
    return face_crop


def get_largest_face(faces):
    """From a list of detected faces, return the one with the largest bounding box area."""
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest


def init_face_detector() -> FaceAnalysis:
    """Initialize InsightFace model. Uses GPU if cuDNN is available, else CPU."""
    print(f"[INFO] Initializing InsightFace ({INSIGHTFACE_MODEL})...")

    # Try CUDA first, fall back to CPU gracefully
    try:
        app = FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": str(GPU_ID)}, {}],
        )
        app.prepare(ctx_id=GPU_ID, det_size=DET_SIZE)
        print(f"[INFO] Face detector ready (GPU {GPU_ID}).")
    except Exception:
        print("[WARN] CUDA unavailable, using CPU for face detection.")
        app = FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=DET_SIZE)
        print("[INFO] Face detector ready (CPU).")
    return app


def get_video_frame_count(video_path: str) -> int:
    """Get total frame count for a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def sample_frames_from_video(
    app: FaceAnalysis,
    video_path: str,
    output_dir: str,
    video_index: int,
    target_frames: int,
) -> int:
    """
    Sample frames from a single video, detect faces, crop, and save.

    Args:
        app: InsightFace detector
        video_path: Path to the video file
        output_dir: Directory to save cropped faces
        video_index: Index of this video (for naming)
        target_frames: How many frames to try to extract from this video

    Returns:
        Number of face crops successfully saved
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"  [WARN] Video has 0 frames: {video_path}")
        cap.release()
        return 0

    # Calculate sampling interval to spread frames evenly across the video
    # Sample more frames than needed to account for blur/detection failures
    oversample_factor = 2.0
    frame_step = max(1, int(total_frames / (target_frames * oversample_factor)))

    saved_count = 0
    frame_idx = 0
    skipped_blur = 0
    skipped_noface = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Detect faces — single-person filter: skip if not exactly 1 face
            faces = app.get(frame)

            # Filter by confidence
            faces = [f for f in faces if f.det_score >= FACE_CONFIDENCE_THRESHOLD]

            if len(faces) != 1:
                skipped_noface += 1
                frame_idx += 1
                continue

            # Exactly one face — use it directly
            face = faces[0]
            face_crop = crop_face_with_padding(frame, face.bbox)

            if face_crop is None:
                frame_idx += 1
                continue

            # Check blur only on the cropped face region (much more meaningful)
            if is_blurry(face_crop, threshold=BLUR_THRESHOLD):
                skipped_blur += 1
                frame_idx += 1
                continue

            # Save
            filename = f"raw_vid{video_index}_frame{frame_idx:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved_count += 1

            if saved_count >= target_frames:
                break

        frame_idx += 1

    cap.release()
    print(
        f"  [OK] vid{video_index}: saved {saved_count}/{target_frames} "
        f"(blur_skip={skipped_blur}, noface_skip={skipped_noface}, "
        f"step={frame_step}, total_frames={total_frames})"
    )
    return saved_count


def process_all_persons(app: FaceAnalysis) -> dict:
    """
    Process all person folders under VIDEO_ROOT.

    Returns:
        Dictionary mapping person name to number of raw images saved.
    """
    if not os.path.isdir(VIDEO_ROOT):
        print(f"[ERROR] Video root not found: {VIDEO_ROOT}")
        sys.exit(1)

    results = {}

    person_dirs = sorted(
        [
            d
            for d in os.listdir(VIDEO_ROOT)
            if os.path.isdir(os.path.join(VIDEO_ROOT, d))
        ]
    )

    print(f"\n[INFO] Found {len(person_dirs)} persons: {person_dirs}")
    print(f"[INFO] Target: {RAW_IMAGES_PER_PERSON} raw images per person\n")

    for person_name in person_dirs:
        person_video_dir = os.path.join(VIDEO_ROOT, person_name)
        output_dir = os.path.join(IMAGE_OUTPUT_ROOT, person_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"{'='*60}")
        print(f"Processing: {person_name}")
        print(f"{'='*60}")

        # Collect video files
        video_files = sorted(
            [
                f
                for f in os.listdir(person_video_dir)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ]
        )

        if not video_files:
            print(f"  [WARN] No videos found for {person_name}")
            results[person_name] = 0
            continue

        # Distribute target frames evenly across videos
        frames_per_video = RAW_IMAGES_PER_PERSON // len(video_files)
        remainder = RAW_IMAGES_PER_PERSON % len(video_files)

        total_saved = 0
        for i, vfile in enumerate(video_files):
            video_path = os.path.join(person_video_dir, vfile)
            # Give the first video(s) any remainder frames
            target = frames_per_video + (1 if i < remainder else 0)

            print(f"  Video {i}: {vfile} (target: {target} frames)")
            saved = sample_frames_from_video(app, video_path, output_dir, i, target)
            total_saved += saved

        results[person_name] = total_saved
        print(f"  TOTAL for {person_name}: {total_saved} raw images\n")

    return results


def main():
    """Run the frame sampling pipeline."""
    print("=" * 60)
    print("STAGE 1: Frame Sampling & Face Cropping")
    print("=" * 60)

    app = init_face_detector()
    results = process_all_persons(app)

    # Summary
    print("\n" + "=" * 60)
    print("FRAME SAMPLING SUMMARY")
    print("=" * 60)
    for person, count in sorted(results.items()):
        status = "✓" if count >= RAW_IMAGES_PER_PERSON * 0.8 else "⚠"
        print(f"  {status} {person}: {count} raw images")
    total = sum(results.values())
    print(f"\n  Total: {total} raw images across {len(results)} persons")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
