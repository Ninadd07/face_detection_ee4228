from __future__ import annotations

import argparse

from ..preprocessing import prepare_aligned_faces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    summary = prepare_aligned_faces(args.config)
    print(
        f"Prepared aligned faces: total={summary.total}, prepared={summary.prepared}, failed={summary.failed}"
    )


if __name__ == "__main__":
    main()
