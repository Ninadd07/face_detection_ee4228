"""
tests/test_arcface_engine.py

Minimal sanity checks for the ArcFace + InsightFace pipeline:
- config paths
- ArcFaceEngine construction
- database loading
- embedding registration from a few images
- recognize_frame on a dummy frame

Run (from project root, venv active):
    python tests/test_arcface_engine.py
"""

from pathlib import Path
import sys

import numpy as np
import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.arcface_insightface.config import (
    DATA_ROOT,
    AUG_IMG_ROOT,
    EMB_ROOT,
    DET_SIZE,
    ARC_FACE_INPUT_SIZE,
    PROTOTYPE_THRESHOLD,
)
from models.arcface_insightface.engine import ArcFaceEngine


def check_paths():
    print("=== PATH CHECK ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_ROOT:    {DATA_ROOT}")
    print(f"AUG_IMG_ROOT: {AUG_IMG_ROOT}")
    print(f"EMB_ROOT:     {EMB_ROOT}")

    if not DATA_ROOT.exists():
        print("[WARN] DATA_ROOT does not exist yet")
    if not AUG_IMG_ROOT.exists():
        print("[WARN] AUG_IMG_ROOT does not exist yet (augmentation not run?)")


def test_engine_init():
    print("\n=== ENGINE INIT ===")
    try:
        engine = ArcFaceEngine(threshold=PROTOTYPE_THRESHOLD)
    except Exception as e:
        print("[ERROR] ArcFaceEngine init failed:")
        raise

    print("[OK] ArcFaceEngine created.")
    print(f"  DET_SIZE: {DET_SIZE}")
    print(f"  ARC_FACE_INPUT_SIZE: {ARC_FACE_INPUT_SIZE}")
    print(f"  Known persons in memory: {list(engine.person_to_embs.keys())}")
    return engine


def test_register_person_from_images(engine: ArcFaceEngine, person_name: str):
    """
    Try registering a person from the first few augmented images.
    This checks:
    - reading images from disk
    - InsightFace detection on cropped images
    - saving embeddings to EMB_ROOT
    - updating in-memory DB and prototypes
    """
    print(f"\n=== REGISTER PERSON FROM IMAGES: {person_name} ===")

    person_dir = AUG_IMG_ROOT / person_name
    if not person_dir.exists():
        print(f"[WARN] No augmented folder for {person_name}: {person_dir}")
        return

    img_paths = sorted(person_dir.glob("*.jpg"))
    if not img_paths:
        print(f"[WARN] No .jpg images found for {person_name}")
        return

    # Use a small subset to keep it fast
    subset = img_paths[:20]
    print(f"  Using {len(subset)} images from {person_dir}")

    engine.register_person_from_images(person_name, subset)

    # Check in-memory
    embs = engine.person_to_embs.get(person_name)
    proto = engine.prototypes.get(person_name)

    if embs is None:
        print("[ERROR] person_to_embs not updated for", person_name)
    else:
        print(f"[OK] person_to_embs[{person_name}] shape: {embs.shape}")

    if proto is None:
        print("[ERROR] prototype not built for", person_name)
    else:
        print(f"[OK] prototype[{person_name}] shape: {proto.shape}")


def test_recognize_frame(engine: ArcFaceEngine):
    """
    Run recognize_frame on a dummy frame to ensure the call path works.
    This does NOT test accuracy, just that nothing crashes.
    """
    print("\n=== RECOGNIZE_FRAME (dummy frame) ===")

    # Create a dummy gray image (no faces expected)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    results = engine.recognize_frame(dummy)

    print(f"[OK] recognize_frame ran, results length: {len(results)}")
    if results:
        print("  (There were detections on dummy frame, which is unusual but not fatal.)")


def main():
    check_paths()

    # 1) Test engine init + loading existing embeddings (if any)
    engine = test_engine_init()

    # 2) If you know at least one person folder exists under AUG_IMG_ROOT,
    #    put a name here to test registration end-to-end.
    #    Example: "jessica"
    example_person = "Jessica"
    test_register_person_from_images(engine, example_person)

    # 3) Sanity check recognize_frame() on a dummy frame
    test_recognize_frame(engine)

    print("\nAll basic tests completed (check messages above for errors/warnings).")


if __name__ == "__main__":
    main()
