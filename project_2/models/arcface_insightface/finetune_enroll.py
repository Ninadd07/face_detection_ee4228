from pathlib import Path

from .config import ENROLL_DIR
from .detector import FaceDetector
from .database import FaceDatabase
from .finetune_config import FINETUNE_EMBEDDINGS_DIR
from .finetune_embedder import FinetunedEmbedder
from .utils import list_images, load_image, save_json, ensure_dirs


def build_database(enroll_dir: Path, db_out: Path, summary_out: Path):
    detector = FaceDetector()
    embedder = FinetunedEmbedder()
    db = FaceDatabase()

    summary = {}

    for person_dir in sorted(enroll_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        embeddings = []
        total = 0
        used = 0
        skipped = 0

        for img_path in list_images(person_dir):
            total += 1
            try:
                image = load_image(img_path)
                detections = detector.detect(image)

                if len(detections) == 0:
                    skipped += 1
                    continue

                det = max(
                    detections,
                    key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                )

                emb = embedder.get_embedding(image, det)
                if emb is None:
                    skipped += 1
                    continue

                embeddings.append(emb)
                used += 1
            except Exception:
                skipped += 1

        db.add_identity(person_dir.name, embeddings)
        summary[person_dir.name] = {
            "total_images": total,
            "used_images": used,
            "skipped_images": skipped,
        }

        print(f"[enroll_finetuned] {person_dir.name}: used={used}, skipped={skipped}, total={total}")

    db.save(db_out)
    save_json(summary, summary_out)


def main():
    ensure_dirs([FINETUNE_EMBEDDINGS_DIR])

    db_out = FINETUNE_EMBEDDINGS_DIR / "finetuned_enroll.npz"
    summary_out = FINETUNE_EMBEDDINGS_DIR / "finetuned_enroll_summary.json"

    build_database(ENROLL_DIR, db_out, summary_out)

    print(f"[enroll_finetuned] Saved database to {db_out}")
    print(f"[enroll_finetuned] Saved summary to {summary_out}")


if __name__ == "__main__":
    main()
