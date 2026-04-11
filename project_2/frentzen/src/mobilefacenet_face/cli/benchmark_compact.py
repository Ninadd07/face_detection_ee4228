from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..benchmark import benchmark_pipeline
from ..config import load_config
from ..dataset import read_manifest
from ..pipeline import build_pipeline
from .enroll_gallery import build_gallery_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--checkpoint-path", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = read_manifest(config["outputs"]["split_manifest"])
    test_paths = [record.path for record in manifest if record.split == "test"][: args.limit]
    pipeline = build_pipeline(args.config, gallery_path=config["outputs"]["gallery_path"], checkpoint_path=args.checkpoint_path)
    if args.checkpoint_path:
        _, gallery = build_gallery_from_manifest(args.config, checkpoint_path=args.checkpoint_path)
        pipeline.gallery = gallery
    metrics = benchmark_pipeline(pipeline, test_paths)
    output_path = Path(config["outputs"]["benchmark_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
