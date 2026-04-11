from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from ..config import load_config
from ..dataset import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    audit = audit_dataset(config["data"]["root_dir"], max_hashes=1000)
    print(json.dumps(asdict(audit), indent=2))


if __name__ == "__main__":
    main()
