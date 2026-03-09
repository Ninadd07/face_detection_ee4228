import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

VIDEO_ROOT = "/Users/jessica/Documents/GitHub/face_detection_ee4228/data_raw/videos"
EMB_ROOT = "embeddings"

def extract_embeddings_from_video(app, video_path, out_dir, frame_step=10, max_embeddings=50):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    embeddings = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            faces = app.get(frame)
            if len(faces) == 1:
                emb = faces[0].normed_embedding
                embeddings.append(emb)
                print(f"  collected emb #{len(embeddings)} at frame {frame_idx}")

                if len(embeddings) >= max_embeddings:
                    break

        frame_idx += 1

    cap.release()

    if not embeddings:
        print(f"[WARN] No embeddings collected from {video_path}")
        return 0

    embeddings = np.array(embeddings)
    # name output file based on video filename
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}_embeddings.npy")
    np.save(out_path, embeddings)
    print(f"[OK] Saved {embeddings.shape} to {out_path}")
    return embeddings.shape[0]

def main():
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    if not os.path.isdir(VIDEO_ROOT):
        print(f"[ERROR] {VIDEO_ROOT} folder not found")
        return

    # each subfolder under video_samples is a person_name
    for person_name in sorted(os.listdir(VIDEO_ROOT)):
        person_dir = os.path.join(VIDEO_ROOT, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n=== Processing person: {person_name} ===")
        out_person_dir = os.path.join(EMB_ROOT, person_name)

        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            video_path = os.path.join(person_dir, fname)
            print(f"[INFO] Video: {video_path}")
            extract_embeddings_from_video(app, video_path, out_person_dir)

if __name__ == "__main__":
    main()
