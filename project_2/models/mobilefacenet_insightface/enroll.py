"""
Build a MobileFaceNet embedding gallery from enrollment images.

Usage
-----
# From the ISD repo root:
python -m face_detection_ee4228.project_2.models.mobilefacenet_insightface.enroll [OPTIONS]

Options
-------
--enroll-dir   Directory of enrollment images, one sub-folder per identity.
               Default: arcface_insightface ENROLL_DIR
--db-out       Output .npz path.
               Default: artifacts_mobilefacenet/embeddings/mobilefacenet_enroll.npz
--checkpoint   MobileFaceNet .pth checkpoint. Overrides MFN_CHECKPOINT env var.
"""
import argparse
from pathlib import Path

from ..arcface_insightface.config import ENROLL_DIR
from ..arcface_insightface.detector import FaceDetector
from ..arcface_insightface.database import FaceDatabase
from ..arcface_insightface.utils import list_images, load_image, save_json, ensure_dirs
from .config import MFN_EMBEDDINGS_DIR, MFN_DEFAULT_CHECKPOINT
from .embedder import MobileFaceNetEmbedder


def build_database(enroll_dir: Path, db_out: Path, summary_out: Path, checkpoint=None):
    detector = FaceDetector()
    embedder = MobileFaceNetEmbedder(checkpoint_path=checkpoint)
    db = FaceDatabase()
    summary = {}

    for person_dir in sorted(enroll_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person = person_dir.name
        embeddings = []
        total = 0
        skipped = 0

        for img_path in list_images(person_dir):
            total += 1
            try:
                image = load_image(img_path)
                detections = detector.detect(image)
                # Strict single-face policy: keeps gallery clean
                if len(detections) != 1:
                    skipped += 1
                    continue
                emb = embedder.get_embedding(image, detections[0])
                if emb is None:
                    skipped += 1
                    continue
                embeddings.append(emb)
            except Exception:
                skipped += 1

        if embeddings:
            db.add_identity(person, embeddings)

        summary[person] = {
            "total_images": total,
            "used_images": total - skipped,
            "skipped_images": skipped,
        }
        print(f"[enroll] {person}: {total - skipped}/{total} used")

    db.save(str(db_out))
    save_json(summary, summary_out)
    print(f"[enroll] Saved database  → {db_out}")
    print(f"[enroll] Saved summary   → {summary_out}")


def main():
    parser = argparse.ArgumentParser(description="Build MobileFaceNet enrollment gallery.")
    parser.add_argument("--enroll-dir", type=str, default=str(ENROLL_DIR))
    parser.add_argument(
        "--db-out",
        type=str,
        default=str(MFN_EMBEDDINGS_DIR / "mobilefacenet_enroll.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to MobileFaceNet .pth checkpoint. "
             "Overrides MFN_CHECKPOINT env var and the default fine-tuned model.",
    )
    args = parser.parse_args()

    ensure_dirs([MFN_EMBEDDINGS_DIR])
    enroll_dir = Path(args.enroll_dir)
    db_out = Path(args.db_out)
    summary_out = db_out.with_name(db_out.stem + "_summary.json")

    build_database(
        enroll_dir,
        db_out,
        summary_out,
        checkpoint=args.checkpoint or str(MFN_DEFAULT_CHECKPOINT),
    )


if __name__ == "__main__":
    main()
