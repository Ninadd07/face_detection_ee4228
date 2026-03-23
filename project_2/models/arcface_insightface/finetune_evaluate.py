import argparse
import json
from pathlib import Path

from .config import (
    VAL_DIR,
    TEST_DIR,
    UNKNOWN_DIR,
    DEFAULT_THRESHOLD,
    UNKNOWN_LABEL,
)
from .finetune_config import FINETUNE_EMBEDDINGS_DIR, FINETUNE_METRICS_DIR
from .database import FaceDatabase
from .detector import FaceDetector
from .finetune_embedder import FinetunedEmbedder
from .matcher import cosine_similarity
from .utils import ensure_dirs, list_images, load_image, save_json


# ── same as evaluate.py ──────────────────────────────────────────────────────

def iter_split_images(split_dir: Path):
    if not split_dir.exists():
        return
    for person_dir in sorted(split_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in list_images(person_dir):
            yield person_dir.name, img_path


def get_best_match(db, emb):
    scores = [cosine_similarity(emb, p) for p in db.prototypes]
    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    return db.labels[best_idx], float(scores[best_idx])


def build_records(split_dir: Path, db: FaceDatabase, is_unknown_case: bool,
                  detector: FaceDetector, embedder: FinetunedEmbedder):
    records = []

    for true_label, img_path in iter_split_images(split_dir):
        actual_true_label = UNKNOWN_LABEL if is_unknown_case else true_label

        try:
            image = load_image(img_path)
            detections = detector.detect(image)

            if len(detections) == 0:
                records.append({
                    "image": str(img_path),
                    "true_label": actual_true_label,
                    "best_label": None,
                    "best_score": None,
                    "status": "DETECTION_FAIL",
                    "is_unknown_case": is_unknown_case,
                })
                continue

            # keep largest face instead of requiring exactly one face
            det = max(
                detections,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
            )

            emb = embedder.get_embedding(image, det)
            if emb is None:
                records.append({
                    "image": str(img_path),
                    "true_label": actual_true_label,
                    "best_label": None,
                    "best_score": None,
                    "status": "EMBED_FAIL",
                    "is_unknown_case": is_unknown_case,
                })
                continue

            best_label, best_score = get_best_match(db, emb)

            records.append({
                "image": str(img_path),
                "true_label": actual_true_label,
                "best_label": best_label,
                "best_score": best_score,
                "status": "OK",
                "is_unknown_case": is_unknown_case,
            })

        except Exception:
            records.append({
                "image": str(img_path),
                "true_label": actual_true_label,
                "best_label": None,
                "best_score": None,
                "status": "ERROR",
                "is_unknown_case": is_unknown_case,
            })

    return records


def apply_threshold(records, threshold):
    out = []
    for r in records:
        rr = dict(r)
        if rr["status"] == "OK":
            rr["pred_label"] = rr["best_label"] if rr["best_score"] >= threshold else UNKNOWN_LABEL
        else:
            rr["pred_label"] = rr["status"]
        out.append(rr)
    return out


def compute_metrics(records):
    valid = [r for r in records if r["status"] == "OK"]

    overall_total = len(valid)
    overall_correct = sum(1 for r in valid if r["pred_label"] == r["true_label"])

    known_valid = [r for r in valid if not r["is_unknown_case"]]
    unknown_valid = [r for r in valid if r["is_unknown_case"]]

    known_correct = sum(1 for r in known_valid if r["pred_label"] == r["true_label"])
    unknown_correct = sum(1 for r in unknown_valid if r["pred_label"] == UNKNOWN_LABEL)

    return {
        "num_records": len(records),
        "num_valid": overall_total,
        "num_detection_fail": sum(1 for r in records if r["status"] == "DETECTION_FAIL"),
        "num_embed_fail": sum(1 for r in records if r["status"] == "EMBED_FAIL"),
        "num_error": sum(1 for r in records if r["status"] == "ERROR"),
        "overall_accuracy": overall_correct / overall_total if overall_total else 0.0,
        "known_accuracy": known_correct / len(known_valid) if known_valid else 0.0,
        "unknown_rejection_rate": unknown_correct / len(unknown_valid) if unknown_valid else None,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--db", type=str,
                        default=str(FINETUNE_EMBEDDINGS_DIR / "finetuned_enroll.npz"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--unknown-dir", type=str, default=str(UNKNOWN_DIR))
    parser.add_argument("--tag", type=str, default="finetuned")
    args = parser.parse_args()

    ensure_dirs([FINETUNE_METRICS_DIR])

    split_dir = VAL_DIR if args.split == "val" else TEST_DIR
    unknown_dir = Path(args.unknown_dir)
    db = FaceDatabase.load(args.db)

    print("[finetune_evaluate] Loading detector once...")
    detector = FaceDetector()

    print("[finetune_evaluate] Loading finetuned embedder once...")
    embedder = FinetunedEmbedder()

    print("[finetune_evaluate] Building records...")
    known_base_records = build_records(split_dir, db, is_unknown_case=False,
                                       detector=detector, embedder=embedder)
    unknown_base_records = (
        build_records(unknown_dir, db, is_unknown_case=True,
                      detector=detector, embedder=embedder)
        if unknown_dir.exists() else []
    )

    if args.sweep:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
        all_results = []
        best_threshold = None
        best_score = -1.0

        for t in thresholds:
            records = apply_threshold(known_base_records + unknown_base_records, t)
            metrics = compute_metrics(records)
            metrics["threshold"] = float(t)
            all_results.append(metrics)

            selection_score = metrics["known_accuracy"]
            if metrics["unknown_rejection_rate"] is not None:
                selection_score = (0.5 * metrics["known_accuracy"]
                                   + 0.5 * metrics["unknown_rejection_rate"])

            if selection_score > best_score:
                best_score = selection_score
                best_threshold = t

        out_path = FINETUNE_METRICS_DIR / f"{args.tag}_{args.split}_threshold_sweep.json"
        save_json({
            "split": args.split,
            "db": args.db,
            "best_threshold": best_threshold,
            "results": all_results,
        }, out_path)

        print(f"[finetune_evaluate] Saved threshold sweep to {out_path}")
        print(f"[finetune_evaluate] Best threshold: {best_threshold}")
        return

    final_records = apply_threshold(known_base_records + unknown_base_records, args.threshold)
    metrics = compute_metrics(final_records)

    metrics_out = FINETUNE_METRICS_DIR / f"{args.tag}_{args.split}_metrics.json"
    records_out = FINETUNE_METRICS_DIR / f"{args.tag}_{args.split}_records.json"

    save_json({
        "split": args.split,
        "db": args.db,
        "threshold": args.threshold,
        "metrics": metrics,
    }, metrics_out)
    save_json(final_records, records_out)

    print(f"[finetune_evaluate] Saved metrics to {metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
