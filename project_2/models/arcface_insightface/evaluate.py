from pathlib import Path
import argparse
import json

from .config import (
    VAL_DIR,
    TEST_DIR,
    UNKNOWN_DIR,
    EMBEDDINGS_DIR,
    METRICS_DIR,
    DEFAULT_THRESHOLD,
    UNKNOWN_LABEL,
)
from .detector import FaceDetector
from .embedder import ArcFaceEmbedder
from .database import FaceDatabase
from .matcher import cosine_similarity
from .utils import ensure_dirs, list_images, load_image, save_json


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


def build_records(split_dir: Path, db: FaceDatabase, is_unknown_case: bool):
    detector = FaceDetector()
    embedder = ArcFaceEmbedder()

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

            emb = embedder.get_embedding_from_detection(det)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--db", type=str, default=str(EMBEDDINGS_DIR / "baseline_enroll.npz"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--unknown-dir", type=str, default=str(UNKNOWN_DIR))
    parser.add_argument("--tag", type=str, default="baseline")
    args = parser.parse_args()

    ensure_dirs([METRICS_DIR])

    split_dir = VAL_DIR if args.split == "val" else TEST_DIR
    unknown_dir = Path(args.unknown_dir)
    db = FaceDatabase.load(args.db)

    known_base_records = build_records(split_dir, db, is_unknown_case=False)
    unknown_base_records = build_records(unknown_dir, db, is_unknown_case=True) if unknown_dir.exists() else []

    if args.sweep:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
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
                selection_score = 0.5 * metrics["known_accuracy"] + 0.5 * metrics["unknown_rejection_rate"]

            if selection_score > best_score:
                best_score = selection_score
                best_threshold = t

        out_path = METRICS_DIR / f"{args.tag}_{args.split}_threshold_sweep.json"
        save_json({
            "split": args.split,
            "db": args.db,
            "best_threshold": best_threshold,
            "results": all_results,
        }, out_path)

        print(f"[evaluate] Saved threshold sweep to {out_path}")
        print(f"[evaluate] Best threshold: {best_threshold}")
        return

    final_records = apply_threshold(known_base_records + unknown_base_records, args.threshold)
    metrics = compute_metrics(final_records)

    metrics_out = METRICS_DIR / f"{args.tag}_{args.split}_metrics_v2.json"
    records_out = METRICS_DIR / f"{args.tag}_{args.split}_records_v2.json"

    save_json({
        "split": args.split,
        "db": args.db,
        "threshold": args.threshold,
        "metrics": metrics,
    }, metrics_out)
    save_json(final_records, records_out)

    print(f"[evaluate] Saved metrics to {metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
