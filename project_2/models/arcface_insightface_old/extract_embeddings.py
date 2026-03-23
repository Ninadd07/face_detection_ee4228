"""
extract_embeddings.py

Offline script to build the ArcFace embeddings database from
the augmented image dataset.

Input layout:
    augmented_data/training_images_augmented/<Person>/*.jpg

Output layout (for THIS model):
    embeddings_arcface/<Person>/emb_XXX.npy 

Usage (from project root, with venv active):
    python -m models.arcface_insightface.extract_embeddings
"""

from pathlib import Path

from .config import AUG_IMG_ROOT, EMB_ROOT, DEBUG
from .engine import ArcFaceEngine


def collect_person_image_paths() -> dict[str, list[Path]]:
    """
    Scan AUG_IMG_ROOT and return:
        { "person_name": [Path(...jpg), Path(...jpg), ...], ... }
    """
    person_to_paths: dict[str, list[Path]] = {}

    if not AUG_IMG_ROOT.exists():
        raise FileNotFoundError(
            f"AUG_IMG_ROOT not found: {AUG_IMG_ROOT}. "
            "Expected augmented images at data/training_images_augmented/<Person>/..."
        )

    for person_dir in sorted(AUG_IMG_ROOT.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        img_paths = sorted(
            p for p in person_dir.glob("*.jpg")
            if p.is_file()
        )

        if not img_paths:
            if DEBUG:
                print(f"[extract_embeddings] No .jpg images for {person_name}")
            continue

        person_to_paths[person_name] = img_paths

    return person_to_paths


def main() -> None:
    # Create engine; it will load any existing embeddings from EMB_ROOT.
    engine = ArcFaceEngine()

    if DEBUG:
        print(f"[extract_embeddings] AUG_IMG_ROOT: {AUG_IMG_ROOT}")
        print(f"[extract_embeddings] EMB_ROOT: {EMB_ROOT}")

    person_to_paths = collect_person_image_paths()
    if not person_to_paths:
        print("[extract_embeddings] No persons found in augmented dataset.")
        return

    for person_name, img_paths in person_to_paths.items():
        print(f"\n=== Processing person: {person_name} ===")
        print(f"  Found {len(img_paths)} images.")

        # You can limit to a subset if needed, e.g. img_paths[:600]
        engine.register_person_from_images(person_name, img_paths)

    print("\n[extract_embeddings] Done building embeddings for ArcFace.")


if __name__ == "__main__":
    main()
