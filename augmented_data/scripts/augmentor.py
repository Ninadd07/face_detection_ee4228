"""
Stage 2: Data Augmentation

Reads raw face crops from augmented_data/images/<Person>/raw_*.jpg
and applies augmentation transforms to reach the target count per person.

Augmentation categories:
  - Geometric: flip, rotation, affine
  - Photometric: brightness, contrast, color jitter, noise
  - Occlusion: random erasing (CoarseDropout)
  - Weather: rain, snow, fog overlays

Output: augmented_data/images/<Person>/aug_*.jpg
"""

import os
import sys
import glob
import random

import cv2
import numpy as np
import albumentations as A

from config import (
    IMAGE_OUTPUT_ROOT,
    TOTAL_IMAGES_PER_PERSON,
    FACE_CROP_SIZE,
    JPEG_QUALITY,
    RANDOM_SEED,
)


# ─── Weather Augmentations (custom, not in albumentations) ───────────────────


def add_rain(image: np.ndarray, intensity: float = 0.6) -> np.ndarray:
    """Add realistic rain streaks to an image."""
    result = image.copy()
    h, w = result.shape[:2]

    # Number of rain drops proportional to intensity
    num_drops = int(h * w * intensity * 0.001)
    for _ in range(num_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(5, 15)
        thickness = random.randint(1, 2)
        angle_offset = random.randint(-5, 5)

        x_end = x + angle_offset
        y_end = min(h - 1, y + length)

        # Semi-transparent white streaks
        cv2.line(result, (x, y), (x_end, y_end), (200, 200, 200), thickness)

    # Slight blur to blend
    result = cv2.GaussianBlur(result, (3, 3), 0)

    # Darken slightly to simulate overcast
    result = cv2.convertScaleAbs(result, alpha=0.85, beta=-10)
    return result


def add_snow(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Add snow particles to an image."""
    result = image.copy()
    h, w = result.shape[:2]

    # Brighten the image slightly (snow reflection)
    result = cv2.convertScaleAbs(result, alpha=1.05, beta=15)

    # Add snowflakes
    num_flakes = int(h * w * intensity * 0.002)
    for _ in range(num_flakes):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(1, 3)
        brightness = random.randint(200, 255)
        cv2.circle(result, (x, y), radius, (brightness, brightness, brightness), -1)

    # Soft blur for realism
    result = cv2.GaussianBlur(result, (3, 3), 0)
    return result


def add_fog(image: np.ndarray, intensity: float = 0.4) -> np.ndarray:
    """Add fog/haze overlay to an image."""
    result = image.copy().astype(np.float32)

    # Create a white overlay
    fog = np.ones_like(result, dtype=np.float32) * 255.0

    # Blend: result = (1 - intensity) * image + intensity * fog
    result = cv2.addWeighted(result, 1.0 - intensity, fog, intensity, 0)

    # Reduce contrast slightly
    mean_val = np.mean(result)
    result = result * 0.8 + mean_val * 0.2

    return np.clip(result, 0, 255).astype(np.uint8)


# ─── Albumentations Pipelines ───────────────────────────────────────────────

def get_geometric_transform():
    """Geometric augmentations: flip, rotate, affine."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=0,  # rotation already handled above
            p=0.5,
        ),
    ])


def get_photometric_transform():
    """Photometric augmentations: brightness, contrast, color, noise."""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.5,
        ),
        A.GaussNoise(std_range=(5.0 / 255, 25.0 / 255), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def get_occlusion_transform():
    """Occlusion augmentation: random rectangular erasing."""
    return A.Compose([
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill="random",
            p=0.5,
        ),
    ])


def get_combined_transform():
    """Full augmentation pipeline combining all categories (no weather)."""
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=0,
            p=0.3,
        ),
        # Photometric
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4
        ),
        A.GaussNoise(std_range=(5.0 / 255, 20.0 / 255), p=0.3),
        # Occlusion
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(0.05, 0.1),
            hole_width_range=(0.05, 0.1),
            fill="random",
            p=0.3,
        ),
    ])


# ─── Augmentation Categories ────────────────────────────────────────────────
# Each category with its weight (probability of being chosen)
AUGMENTATION_CATEGORIES = [
    ("geometric", get_geometric_transform, 0.25),
    ("photometric", get_photometric_transform, 0.25),
    ("occlusion", get_occlusion_transform, 0.10),
    ("combined", get_combined_transform, 0.20),
    ("rain", None, 0.07),
    ("snow", None, 0.06),
    ("fog", None, 0.07),
]


def apply_augmentation(image: np.ndarray, category: str, transform=None) -> np.ndarray:
    """Apply a single augmentation based on category."""
    if category == "rain":
        intensity = random.uniform(0.3, 0.8)
        return add_rain(image, intensity)
    elif category == "snow":
        intensity = random.uniform(0.3, 0.7)
        return add_snow(image, intensity)
    elif category == "fog":
        intensity = random.uniform(0.2, 0.5)
        return add_fog(image, intensity)
    else:
        # albumentations-based transform
        result = transform(image=image)
        return result["image"]


def augment_person(person_name: str, person_dir: str) -> int:
    """
    Augment raw images for a single person to reach the target count.

    Returns:
        Number of augmented images created.
    """
    # Find all raw images
    raw_pattern = os.path.join(person_dir, "raw_*.jpg")
    raw_files = sorted(glob.glob(raw_pattern))

    if not raw_files:
        print(f"  [WARN] No raw images found for {person_name}")
        return 0

    num_raw = len(raw_files)
    num_augmented_needed = max(0, TOTAL_IMAGES_PER_PERSON - num_raw)

    print(f"  Raw images: {num_raw}, augmented needed: {num_augmented_needed}")

    if num_augmented_needed == 0:
        print(f"  [INFO] Already at target count, skipping augmentation")
        return 0

    # Pre-build transforms
    transforms = {}
    for cat_name, transform_fn, _ in AUGMENTATION_CATEGORIES:
        if transform_fn is not None:
            transforms[cat_name] = transform_fn()

    # Build weighted category list
    categories = [cat for cat, _, _ in AUGMENTATION_CATEGORIES]
    weights = [w for _, _, w in AUGMENTATION_CATEGORIES]

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    saved_count = 0
    aug_counts = {cat: 0 for cat in categories}

    # Round-robin through raw images, applying random augmentations
    attempts = 0
    max_attempts = num_augmented_needed * 3  # safety limit

    while saved_count < num_augmented_needed and attempts < max_attempts:
        # Pick a random raw image
        raw_file = random.choice(raw_files)
        raw_basename = os.path.splitext(os.path.basename(raw_file))[0]
        # e.g., "raw_vid0_frame00030" -> extract video/frame info
        base_id = raw_basename.replace("raw_", "")

        # Pick a random augmentation category
        category = random.choices(categories, weights=weights, k=1)[0]

        # Load image
        image = cv2.imread(raw_file)
        if image is None:
            attempts += 1
            continue

        # Ensure correct size
        if image.shape[:2] != (FACE_CROP_SIZE[1], FACE_CROP_SIZE[0]):
            image = cv2.resize(image, FACE_CROP_SIZE, interpolation=cv2.INTER_AREA)

        # Apply augmentation
        try:
            aug_image = apply_augmentation(
                image, category, transforms.get(category)
            )
        except Exception as e:
            print(f"  [WARN] Augmentation failed: {e}")
            attempts += 1
            continue

        # Save
        aug_idx = saved_count
        filename = f"aug_{base_id}_{category}_{aug_idx:04d}.jpg"
        filepath = os.path.join(person_dir, filename)
        cv2.imwrite(filepath, aug_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        saved_count += 1
        aug_counts[category] += 1
        attempts += 1

    # Print category distribution
    print(f"  Augmentation distribution:")
    for cat, count in sorted(aug_counts.items()):
        if count > 0:
            print(f"    {cat}: {count}")

    return saved_count


def process_all_persons() -> dict:
    """
    Augment images for all persons.

    Returns:
        Dictionary mapping person name to augmentation stats.
    """
    if not os.path.isdir(IMAGE_OUTPUT_ROOT):
        print(f"[ERROR] Image output root not found: {IMAGE_OUTPUT_ROOT}")
        print("[INFO] Run frame_sampler.py first!")
        sys.exit(1)

    results = {}

    person_dirs = sorted(
        [
            d
            for d in os.listdir(IMAGE_OUTPUT_ROOT)
            if os.path.isdir(os.path.join(IMAGE_OUTPUT_ROOT, d))
        ]
    )

    if not person_dirs:
        print("[ERROR] No person directories found. Run frame_sampler.py first!")
        sys.exit(1)

    print(f"\n[INFO] Found {len(person_dirs)} persons: {person_dirs}")
    print(f"[INFO] Target: {TOTAL_IMAGES_PER_PERSON} total images per person\n")

    for person_name in person_dirs:
        person_dir = os.path.join(IMAGE_OUTPUT_ROOT, person_name)

        print(f"{'='*60}")
        print(f"Augmenting: {person_name}")
        print(f"{'='*60}")

        aug_count = augment_person(person_name, person_dir)

        # Count total images
        total = len(glob.glob(os.path.join(person_dir, "*.jpg")))
        results[person_name] = {"augmented": aug_count, "total": total}

        print(f"  TOTAL for {person_name}: {total} images\n")

    return results


def main():
    """Run the augmentation pipeline."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("STAGE 2: Data Augmentation")
    print("=" * 60)

    results = process_all_persons()

    # Summary
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    for person, stats in sorted(results.items()):
        status = "✓" if stats["total"] >= TOTAL_IMAGES_PER_PERSON * 0.95 else "⚠"
        print(
            f"  {status} {person}: {stats['total']} total "
            f"({stats['augmented']} augmented)"
        )
    grand_total = sum(s["total"] for s in results.values())
    print(f"\n  Grand total: {grand_total} images across {len(results)} persons")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
