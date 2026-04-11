from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .types import SampleRecord


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_identity_images(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def infer_group_id(path: Path, group_pattern: str | None = None) -> str:
    stem = path.stem
    if group_pattern:
        match = re.search(group_pattern, stem)
        if match:
            return match.group(1)
    match = re.search(r"(vid\d+_frame\d+)", stem)
    if match:
        return match.group(1)
    tokens = stem.split("_")
    return "_".join(tokens[:-1]) if len(tokens) > 1 else stem


def hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_split_manifest(
    root_dir: str | Path,
    enrollment_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    group_pattern: str | None,
    seed: int,
) -> list[SampleRecord]:
    if round(enrollment_ratio + validation_ratio + test_ratio, 6) != 1.0:
        raise ValueError("Split ratios must sum to 1.0.")

    import random

    rng = random.Random(seed)
    root = Path(root_dir)
    by_identity: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in iter_identity_images(root):
        identity = path.parent.name
        by_identity[identity][infer_group_id(path, group_pattern)].append(path)

    records: list[SampleRecord] = []
    for identity, groups in sorted(by_identity.items()):
        group_ids = list(groups.keys())
        rng.shuffle(group_ids)
        total = len(group_ids)
        enrollment_cut = max(1, int(total * enrollment_ratio))
        validation_cut = min(total, enrollment_cut + max(1, int(total * validation_ratio)))
        assignments = {
            "enrollment": group_ids[:enrollment_cut],
            "validation": group_ids[enrollment_cut:validation_cut],
            "test": group_ids[validation_cut:],
        }
        if not assignments["test"] and assignments["validation"]:
            assignments["test"].append(assignments["validation"].pop())
        for split, selected_groups in assignments.items():
            for group_id in selected_groups:
                for path in groups[group_id]:
                    records.append(SampleRecord(identity=identity, path=path, split=split, group_id=group_id))
    return records


def write_manifest(records: Iterable[SampleRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["identity", "path", "split", "group_id"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "identity": record.identity,
                    "path": str(record.path),
                    "split": record.split,
                    "group_id": record.group_id,
                }
            )


def read_manifest(path: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SampleRecord(
                    identity=row["identity"],
                    path=Path(row["path"]),
                    split=row["split"],
                    group_id=row["group_id"],
                )
            )
    return records


@dataclass(slots=True)
class DatasetAudit:
    identities: int
    images: int
    counts_per_identity: dict[str, int]
    duplicate_hashes: dict[str, list[str]]


def audit_dataset(root_dir: str | Path, max_hashes: int | None = None) -> DatasetAudit:
    counts: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    paths = iter_identity_images(root_dir)
    for index, path in enumerate(paths):
        counts[path.parent.name] += 1
        if max_hashes is None or index < max_hashes:
            hashes[hash_file(path)].append(str(path))
    duplicates = {key: value for key, value in hashes.items() if len(value) > 1}
    return DatasetAudit(
        identities=len(counts),
        images=sum(counts.values()),
        counts_per_identity=dict(sorted(counts.items())),
        duplicate_hashes=duplicates,
    )
