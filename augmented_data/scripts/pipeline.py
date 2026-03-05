"""
Master Pipeline: Frame Sampling → Augmentation

Runs both stages sequentially:
  Stage 1: Extract & crop faces from training videos
  Stage 2: Augment face crops to reach target count

Usage:
    cd face_detection_ee4228/augmented_data/scripts
    /path/to/.venv/bin/python pipeline.py

    Or with the venv activated:
    python pipeline.py

Options:
    --stage1-only    Run only frame sampling
    --stage2-only    Run only augmentation
    --dry-run        Show what would be done without processing
"""

import os
import sys
import time
import argparse
import glob

from config import (
    VIDEO_ROOT,
    IMAGE_OUTPUT_ROOT,
    RAW_IMAGES_PER_PERSON,
    TOTAL_IMAGES_PER_PERSON,
    FACE_CROP_SIZE,
    GPU_ID,
)


def print_banner():
    """Print pipeline banner with configuration."""
    print("\n" + "=" * 60)
    print("  FACE DATA PREPROCESSING & AUGMENTATION PIPELINE")
    print("=" * 60)
    print(f"  Video source:     {os.path.abspath(VIDEO_ROOT)}")
    print(f"  Output dir:       {os.path.abspath(IMAGE_OUTPUT_ROOT)}")
    print(f"  Face crop size:   {FACE_CROP_SIZE[0]}×{FACE_CROP_SIZE[1]}")
    print(f"  GPU:              cuda:{GPU_ID}")
    print(f"  Raw per person:   {RAW_IMAGES_PER_PERSON}")
    print(f"  Total per person: {TOTAL_IMAGES_PER_PERSON}")
    print("=" * 60 + "\n")


def show_dataset_overview():
    """Show an overview of what's in the video directory."""
    if not os.path.isdir(VIDEO_ROOT):
        print(f"[ERROR] Video root not found: {VIDEO_ROOT}")
        return False

    print("[INFO] Dataset overview:")
    person_dirs = sorted(
        [d for d in os.listdir(VIDEO_ROOT) if os.path.isdir(os.path.join(VIDEO_ROOT, d))]
    )

    for person in person_dirs:
        person_path = os.path.join(VIDEO_ROOT, person)
        videos = [
            f
            for f in os.listdir(person_path)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".wmv"))
        ]
        print(f"  {person}: {len(videos)} videos")

    print(f"\n  Total: {len(person_dirs)} persons")
    return True


def show_output_summary():
    """Show summary of generated data."""
    if not os.path.isdir(IMAGE_OUTPUT_ROOT):
        print("[INFO] No output data yet.")
        return

    print("\n[INFO] Output data summary:")
    person_dirs = sorted(
        [
            d
            for d in os.listdir(IMAGE_OUTPUT_ROOT)
            if os.path.isdir(os.path.join(IMAGE_OUTPUT_ROOT, d))
        ]
    )

    grand_total = 0
    for person in person_dirs:
        person_path = os.path.join(IMAGE_OUTPUT_ROOT, person)
        raw = len(glob.glob(os.path.join(person_path, "raw_*.jpg")))
        aug = len(glob.glob(os.path.join(person_path, "aug_*.jpg")))
        total = raw + aug
        grand_total += total
        status = "✓" if total >= TOTAL_IMAGES_PER_PERSON * 0.95 else "⚠"
        print(f"  {status} {person}: {total} images (raw={raw}, aug={aug})")

    print(f"\n  Grand total: {grand_total} images")


def run_stage1():
    """Run frame sampling stage."""
    from frame_sampler import main as stage1_main

    start = time.time()
    results = stage1_main()
    elapsed = time.time() - start
    print(f"\n[INFO] Stage 1 completed in {elapsed:.1f}s")
    return results


def run_stage2():
    """Run augmentation stage."""
    from augmentor import main as stage2_main

    start = time.time()
    results = stage2_main()
    elapsed = time.time() - start
    print(f"\n[INFO] Stage 2 completed in {elapsed:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Face Data Preprocessing & Augmentation Pipeline"
    )
    parser.add_argument(
        "--stage1-only", action="store_true", help="Run only frame sampling"
    )
    parser.add_argument(
        "--stage2-only", action="store_true", help="Run only augmentation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and dataset overview without processing",
    )
    args = parser.parse_args()

    print_banner()

    if not show_dataset_overview():
        sys.exit(1)

    if args.dry_run:
        show_output_summary()
        print("\n[INFO] Dry run complete. No processing done.")
        return

    total_start = time.time()

    if args.stage2_only:
        print("\n[INFO] Skipping Stage 1 (--stage2-only)\n")
    else:
        print("\n" + "─" * 60)
        print("  STAGE 1: Frame Sampling & Face Cropping")
        print("─" * 60)
        run_stage1()

    if args.stage1_only:
        print("\n[INFO] Skipping Stage 2 (--stage1-only)\n")
    else:
        print("\n" + "─" * 60)
        print("  STAGE 2: Data Augmentation")
        print("─" * 60)
        run_stage2()

    total_elapsed = time.time() - total_start

    # Final summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    show_output_summary()
    print(f"\n  Total time: {total_elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
