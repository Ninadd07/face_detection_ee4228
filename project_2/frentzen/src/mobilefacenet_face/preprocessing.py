from __future__ import annotations

import csv
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .config import load_config
from .dataset import read_manifest
from .pipeline import build_detector_and_aligner


@dataclass(slots=True)
class PreparationSummary:
    total: int
    prepared: int
    failed: int


FAILURE_REPORT_FIELDS = [
    "identity",
    "source_path",
    "aligned_path",
    "split",
    "group_id",
    "confidence",
    "success",
    "error",
]


def write_failure_report(rows: list[dict[str, str]], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_contact_sheet(
    rows: list[dict[str, str]],
    output_path: str | Path,
    *,
    thumb_size: tuple[int, int] = (128, 128),
    columns: int = 5,
) -> None:
    if not rows:
        return

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    padding = 12
    caption_height = 34
    cell_width = thumb_size[0] + padding * 2
    cell_height = thumb_size[1] + caption_height + padding * 2
    row_count = ceil(len(rows) / columns)
    canvas = Image.new("RGB", (columns * cell_width, row_count * cell_height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    for index, row in enumerate(rows):
        col = index % columns
        grid_row = index // columns
        x0 = col * cell_width
        y0 = grid_row * cell_height
        frame = (x0 + padding, y0 + padding, x0 + padding + thumb_size[0], y0 + padding + thumb_size[1])
        draw.rectangle(frame, fill=(230, 230, 230), outline=(180, 180, 180))

        source_path = Path(row["source_path"])
        image_bgr = cv2.imread(str(source_path))
        if image_bgr is None:
            preview = Image.fromarray(np.full((thumb_size[1], thumb_size[0], 3), 180, dtype=np.uint8))
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            preview = Image.fromarray(image_rgb)
        preview = ImageOps.contain(preview, thumb_size)
        offset_x = x0 + padding + (thumb_size[0] - preview.width) // 2
        offset_y = y0 + padding + (thumb_size[1] - preview.height) // 2
        canvas.paste(preview, (offset_x, offset_y))

        caption = f"{row['identity']} | {source_path.stem[:18]}"
        draw.text((x0 + padding, y0 + padding + thumb_size[1] + 6), caption, fill=(20, 20, 20))

    canvas.save(target, format="JPEG", quality=90)


def prune_failed_source_images(report_path: str | Path, *, apply: bool = False) -> dict[str, int]:
    target = Path(report_path)
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        candidate_paths = {
            Path(row["source_path"])
            for row in reader
            if row.get("source_path") and row.get("success", "false").lower() == "false"
        }

    existing_paths = [path for path in candidate_paths if path.exists()]
    deleted = 0
    if apply:
        for path in existing_paths:
            path.unlink()
            deleted += 1

    return {
        "total_failed_sources": len(candidate_paths),
        "existing_sources": len(existing_paths),
        "missing_sources": len(candidate_paths) - len(existing_paths),
        "deleted_sources": deleted,
    }


def prepare_aligned_faces(config_path: str | Path | None = None) -> PreparationSummary:
    config = load_config(config_path)
    records = read_manifest(config["outputs"]["split_manifest"])
    detector, aligner, _ = build_detector_and_aligner(config)
    aligned_dir = Path(config["preprocess"]["aligned_dir"])
    aligned_manifest = Path(config["preprocess"]["aligned_manifest"])
    failure_report_path = Path(config["outputs"].get("failed_sources_path", "artifacts/reports/aligned_failures.csv"))
    failure_contact_sheet_path = Path(
        config["outputs"].get("failed_contact_sheet_path", "artifacts/reports/aligned_failures_contact_sheet.jpg")
    )
    overwrite = bool(config["preprocess"].get("overwrite", False))

    aligned_dir.mkdir(parents=True, exist_ok=True)
    aligned_manifest.parent.mkdir(parents=True, exist_ok=True)

    prepared = 0
    failed = 0
    failed_rows: list[dict[str, str]] = []
    with aligned_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FAILURE_REPORT_FIELDS,
        )
        writer.writeheader()
        for record in records:
            source_path = Path(record.path)
            relative_name = f"{record.group_id}_{source_path.stem}.jpg"
            target_path = aligned_dir / record.split / record.identity / relative_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            success = False
            error = ""
            confidence = ""
            try:
                if target_path.exists() and not overwrite:
                    success = True
                else:
                    image_bgr = cv2.imread(str(source_path))
                    if image_bgr is None:
                        raise FileNotFoundError(f"Unable to read image: {source_path}")
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    detections = detector.detect(image_rgb)
                    if not detections:
                        raise RuntimeError("No face detected.")
                    best = max(detections, key=lambda detection: detection.confidence)
                    aligned_rgb = aligner.align(image_rgb, best.landmarks)
                    confidence = f"{best.confidence:.6f}"
                    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(target_path), aligned_bgr):
                        raise RuntimeError(f"Failed to write aligned image: {target_path}")
                    success = True
                prepared += 1 if success else 0
            except Exception as exc:  # noqa: BLE001
                failed += 1
                error = str(exc)
            row = {
                "identity": record.identity,
                "source_path": str(source_path),
                "aligned_path": str(target_path),
                "split": record.split,
                "group_id": record.group_id,
                "confidence": confidence,
                "success": str(success).lower(),
                "error": error,
            }
            if not success:
                failed_rows.append(row)
            writer.writerow(row)
    write_failure_report(failed_rows, failure_report_path)
    write_failure_contact_sheet(failed_rows, failure_contact_sheet_path)
    return PreparationSummary(total=len(records), prepared=prepared, failed=failed)
