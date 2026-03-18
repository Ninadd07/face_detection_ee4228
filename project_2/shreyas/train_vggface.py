import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf

from keras_vggface_compat import patch_keras_for_vggface

from vggface_config import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    CLASS_INDEX_PATH,
    DATASET_DIR,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    HISTORY_CSV_PATH,
    INPUT_SIZE,
    LEARNING_RATE,
    MODEL_PATH,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)


patch_keras_for_vggface()
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace


def vggface_preprocess(batch):
    return preprocess_input(batch, version=1)


def build_generators(dataset_dir: Path, batch_size: int):
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=vggface_preprocess,
        validation_split=VALIDATION_SPLIT,
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.10,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=vggface_preprocess,
        validation_split=VALIDATION_SPLIT,
    )

    train_loader = train_gen.flow_from_directory(
        str(dataset_dir),
        target_size=INPUT_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
        subset="training",
    )

    valid_loader = valid_gen.flow_from_directory(
        str(dataset_dir),
        target_size=INPUT_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=RANDOM_SEED,
        subset="validation",
    )

    return train_loader, valid_loader


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = VGGFace(
        model="vgg16",
        include_top=False,
        input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3),
        pooling="avg",
    )

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers:
        if layer.name.startswith("conv5"):
            layer.trainable = True

    x = tf.keras.layers.Dense(256, activation="relu")(base_model.output)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_class_indices(class_indices: Dict[str, int], output_path: Path) -> None:
    payload = {
        "class_to_index": class_indices,
        "index_to_class": {str(v): k for k, v in class_indices.items()},
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_history_csv(history: tf.keras.callbacks.History, output_path: Path) -> None:
    if not history.history:
        return

    keys = list(history.history.keys())
    num_rows = len(history.history[keys[0]])

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for idx in range(num_rows):
            row = [idx + 1] + [history.history[k][idx] for k in keys]
            writer.writerow(row)


def train(dataset_dir: Path, epochs: int, batch_size: int) -> Tuple[tf.keras.Model, Dict]:
    tf.random.set_seed(RANDOM_SEED)

    train_loader, valid_loader = build_generators(dataset_dir, batch_size)
    if train_loader.num_classes < 2:
        raise ValueError("Need at least 2 classes to train classifier.")

    model = build_model(num_classes=train_loader.num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_loader,
        validation_data=valid_loader,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = model.evaluate(valid_loader, verbose=0)
    eval_summary = dict(zip(model.metrics_names, [float(v) for v in metrics]))

    save_class_indices(train_loader.class_indices, CLASS_INDEX_PATH)
    save_history_csv(history, HISTORY_CSV_PATH)
    model.save(str(MODEL_PATH))

    return model, eval_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune VGGFace for friend classification.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to dataset folder with one subfolder per friend.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _, eval_summary = train(args.dataset_dir, args.epochs, args.batch_size)

    print("[train_vggface] Model saved:", MODEL_PATH)
    print("[train_vggface] Class map saved:", CLASS_INDEX_PATH)
    print("[train_vggface] History saved:", HISTORY_CSV_PATH)
    print("[train_vggface] Validation metrics:", eval_summary)


if __name__ == "__main__":
    main()
