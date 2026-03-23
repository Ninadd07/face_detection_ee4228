from pathlib import Path

from .config import ENROLL_DIR, EMBEDDINGS_DIR
from .detector import FaceDetector
from .embedder import ArcFaceEmbedder
from .database import FaceDatabase
from .utils import list_images, load_image, save_json, ensure_dirs


def build_database(enroll_dir: Path, db_out: Path, summary_out: Path):
    detector = FaceDetector()
    embedder = ArcFaceEmbedder()
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

                if len(detections) != 1:
                    skipped += 1
                    continue

                emb = embedder.get_embedding_from_detection(detections[0])
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

        print(
            f"[enroll] {person_dir.name}: "
            f"used={used}, skipped={skipped}, total={total}"
        )

    db.save(db_out)
    save_json(summary, summary_out)


def main():
    ensure_dirs([EMBEDDINGS_DIR])

    db_out = EMBEDDINGS_DIR / "baseline_enroll.npz"
    summary_out = EMBEDDINGS_DIR / "baseline_enroll_summary.json"

    build_database(ENROLL_DIR, db_out, summary_out)

    print(f"[enroll] Saved database to {db_out}")
    print(f"[enroll] Saved summary to {summary_out}")


if __name__ == "__main__":
    main()
