from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..preprocessing import prune_failed_source_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    report_path = Path(args.report or config["outputs"].get("failed_sources_path", "artifacts/reports/aligned_failures.csv"))
    result = prune_failed_source_images(report_path, apply=args.apply)
    print(json.dumps({"report_path": str(report_path), "apply": args.apply, **result}, indent=2))


if __name__ == "__main__":
    main()
