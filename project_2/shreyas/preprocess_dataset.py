import argparse
import json
from pathlib import Path
from typing import Dict, List

from vggface_config import ARTIFACT_DIR, DATASET_DIR, DATASET_SUMMARY_PATH


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _iter_images(folder: Path) -> List[Path]:
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]


def collect_summary(dataset_dir: Path) -> Dict:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")

    class_folders = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    if not class_folders:
        raise ValueError(f"No class folders found under: {dataset_dir}")

    class_counts: Dict[str, int] = {}
    total_images = 0

    for class_dir in class_folders:
        images = _iter_images(class_dir)
        class_counts[class_dir.name] = len(images)
        total_images += len(images)

    non_empty_classes = [name for name, count in class_counts.items() if count > 0]

    return {
        "dataset_dir": str(dataset_dir),
        "num_classes": len(class_folders),
        "num_non_empty_classes": len(non_empty_classes),
        "classes": [p.name for p in class_folders],
        "class_image_counts": class_counts,
        "total_images": total_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the personal face dataset before training."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to training_images_augmented directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_SUMMARY_PATH,
        help="Path to save dataset summary JSON.",
    )
    args = parser.parse_args()

    summary = collect_summary(args.dataset_dir)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[preprocess_dataset] Dataset summary saved:", args.output)
    print("[preprocess_dataset] Classes:", summary["num_classes"])
    print("[preprocess_dataset] Total images:", summary["total_images"])


if __name__ == "__main__":
    main()
