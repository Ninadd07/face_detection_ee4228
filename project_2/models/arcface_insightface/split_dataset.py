import random
import shutil
from pathlib import Path

from .config import (
    RAW_DATA_DIR,
    ENROLL_DIR,
    VAL_DIR,
    TEST_DIR,
    ENROLL_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)
from .utils import ensure_dirs, list_images, seed_everything


MAX_IMAGES_PER_IDENTITY = 120


def clear_split_dirs():
    for split_dir in [ENROLL_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def split_identity(person_dir: Path):
    images = list_images(person_dir)
    if len(images) < 3:
        print(f"[split_dataset] Skipping {person_dir.name}: not enough images")
        return

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(images)

    original_count = len(images)

    if MAX_IMAGES_PER_IDENTITY is not None:
        images = images[:MAX_IMAGES_PER_IDENTITY]

    used_count = len(images)

    n = len(images)
    n_enroll = int(n * ENROLL_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_enroll - n_val

    enroll_imgs = images[:n_enroll]
    val_imgs = images[n_enroll:n_enroll + n_val]
    test_imgs = images[n_enroll + n_val:]

    split_map = {
        ENROLL_DIR / person_dir.name: enroll_imgs,
        VAL_DIR / person_dir.name: val_imgs,
        TEST_DIR / person_dir.name: test_imgs,
    }

    for out_dir, split_imgs in split_map.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in split_imgs:
            shutil.copy2(src, out_dir / src.name)

    print(
        f"[split_dataset] {person_dir.name}: "
        f"original={original_count}, used={used_count}, "
        f"enroll={len(enroll_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}"
    )


def main():
    seed_everything()

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"RAW_DATA_DIR does not exist: {RAW_DATA_DIR}")

    ensure_dirs([ENROLL_DIR, VAL_DIR, TEST_DIR])
    clear_split_dirs()

    person_dirs = sorted([p for p in RAW_DATA_DIR.iterdir() if p.is_dir()])
    for person_dir in person_dirs:
        split_identity(person_dir)

    print("[split_dataset] Done.")


if __name__ == "__main__":
    main()
