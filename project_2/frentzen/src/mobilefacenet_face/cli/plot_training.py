from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from ..config import load_config

matplotlib.use("Agg")


def plot_training_curves(metrics_path: str | Path, output_path: str | Path) -> Path:
    metrics_file = Path(metrics_path)
    output_file = Path(output_path)

    with metrics_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    history = payload.get("history", [])
    if not history:
        raise ValueError(f"No training history found in {metrics_file}")

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    val_accuracy = [row["val_accuracy"] for row in history]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, marker="o", label="Train Accuracy")
    axes[1].plot(epochs, val_accuracy, marker="o", label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    metrics_path = args.metrics_path or config["training"]["training_metrics_path"]
    output_path = args.output_path or Path(config["training"]["training_metrics_path"]).with_name("training_curves.png")
    saved_path = plot_training_curves(metrics_path, output_path)
    print(f"Saved training curves to {saved_path}")


if __name__ == "__main__":
    main()
