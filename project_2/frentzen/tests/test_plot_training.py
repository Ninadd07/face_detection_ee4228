import json

from mobilefacenet_face.cli.plot_training import plot_training_curves


def test_plot_training_curves_writes_png(tmp_path):
    metrics_path = tmp_path / "training_metrics.json"
    output_path = tmp_path / "training_curves.png"
    metrics_path.write_text(
        json.dumps(
            {
                "history": [
                    {
                        "epoch": 1,
                        "learning_rate": 0.0003,
                        "train_loss": 1.2,
                        "train_accuracy": 0.55,
                        "val_loss": 1.0,
                        "val_accuracy": 0.60,
                    },
                    {
                        "epoch": 2,
                        "learning_rate": 0.0002,
                        "train_loss": 0.8,
                        "train_accuracy": 0.72,
                        "val_loss": 0.9,
                        "val_accuracy": 0.68,
                    },
                ],
                "best_metric": 0.68,
                "selection_metric": "val_accuracy",
            }
        ),
        encoding="utf-8",
    )

    saved = plot_training_curves(metrics_path, output_path)
    assert saved == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
