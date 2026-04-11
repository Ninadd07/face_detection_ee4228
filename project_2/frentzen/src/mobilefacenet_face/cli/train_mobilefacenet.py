from __future__ import annotations

import argparse

from ..training import train_mobilefacenet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    result = train_mobilefacenet(args.config)
    print(
        f"Training complete. Best checkpoint: {result['best_checkpoint_path']}. "
        f"Last checkpoint: {result['last_checkpoint_path']}."
    )


if __name__ == "__main__":
    main()
