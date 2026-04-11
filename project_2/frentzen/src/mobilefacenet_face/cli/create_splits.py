from __future__ import annotations

import argparse

from ..config import load_config
from ..dataset import create_split_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    records = create_split_manifest(
        root_dir=config["data"]["root_dir"],
        enrollment_ratio=config["data"]["enrollment_ratio"],
        validation_ratio=config["data"]["validation_ratio"],
        test_ratio=config["data"]["test_ratio"],
        group_pattern=config["data"].get("group_pattern"),
        seed=config["data"]["split_seed"],
    )
    write_manifest(records, config["outputs"]["split_manifest"])
    print(f"Wrote {len(records)} manifest rows to {config['outputs']['split_manifest']}")


if __name__ == "__main__":
    main()
