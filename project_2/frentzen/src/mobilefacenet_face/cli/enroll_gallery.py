from __future__ import annotations

import argparse
from collections import defaultdict

from ..config import load_config
from ..dataset import read_manifest
from ..evaluation import load_image
from ..gallery import EmbeddingGallery
from ..pipeline import build_pipeline


def build_gallery_from_manifest(config_path, checkpoint_path=None) -> tuple[dict, EmbeddingGallery]:
    config = load_config(config_path)
    manifest = read_manifest(config["outputs"]["split_manifest"])
    pipeline = build_pipeline(config_path, gallery_path=None, checkpoint_path=checkpoint_path, load_gallery=False)
    templates = defaultdict(list)
    skipped_missing = 0
    skipped_no_face = 0
    for record in manifest:
        if record.split != "enrollment":
            continue
        try:
            frame = load_image(record.path)
        except FileNotFoundError:
            skipped_missing += 1
            continue
        result = pipeline.run(frame)
        if result.predictions:
            templates[record.identity].append(result.predictions[0].embedding)
        else:
            skipped_no_face += 1
    gallery = EmbeddingGallery.from_templates(templates, aggregation=config["matcher"]["aggregation"])
    config["_enrollment_summary"] = {
        "skipped_missing": skipped_missing,
        "skipped_no_face": skipped_no_face,
        "templates": sum(len(vectors) for vectors in templates.values()),
    }
    return config, gallery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()
    config, gallery = build_gallery_from_manifest(args.config, checkpoint_path=args.checkpoint_path)
    output_path = args.output_path or config["outputs"]["gallery_path"]
    gallery.save(output_path)
    summary = config.get("_enrollment_summary", {})
    print(
        f"Saved gallery with {len(gallery.identities)} identities to {output_path}. "
        f"Skipped missing={summary.get('skipped_missing', 0)}, no-face={summary.get('skipped_no_face', 0)}."
    )


if __name__ == "__main__":
    main()
