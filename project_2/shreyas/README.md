# VGGFace Fine-Tuning Pipeline (Shreyas)

This folder contains an end-to-end pipeline for closed-set face recognition of 7 known people:

- dataset validation and summary export
- transfer learning with pretrained VGGFace (VGG16 backbone)
- artifact export (model, class map, training history)
- live webcam inference with face boxes, labels, confidence, and FPS

The implementation is designed to be practical on standard laptop hardware while still giving good performance for a small custom dataset.

## What This Pipeline Does

The system adapts pretrained face features to your project-specific identities.

1. Most VGGFace layers stay frozen.
2. Only the final convolution block (`conv5*`) is unfrozen.
3. A custom classifier head is trained on your class folders.
4. The trained model is used for real-time webcam recognition.

This approach balances accuracy, training stability, and compute cost.

## Folder Contents

- `vggface_config.py`: central paths and default hyperparameters.
- `keras_vggface_compat.py`: compatibility patch for `keras_vggface` with newer Keras/TensorFlow.
- `preprocess_dataset.py`: validates dataset and writes a summary JSON.
- `train_vggface.py`: builds generators, trains/fine-tunes model, writes artifacts.
- `live_webcam_vggface.py`: webcam face detection and recognition overlay.
- `requirements_vggface.txt`: Python dependencies.

## Expected Dataset Structure

Dataset root should contain one folder per identity:

```text
training_images_augmented/
|-- Abhiram/
|-- Frentzen/
|-- Jessica/
|-- Ninad/
|-- Ryan/
|-- Sasi/
`-- Shreyas/
```

Each subfolder name is treated as the class label.

Supported image extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

## Environment Setup

Run from repository root.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r project_2/shreyas/requirements_vggface.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r project_2/shreyas/requirements_vggface.txt
```

## Pipeline Stages

### 1) Validate and Summarize Dataset

```bash
python project_2/shreyas/preprocess_dataset.py --dataset-dir training_images_augmented
```

What it checks:

- dataset path exists
- class folders exist
- per-class image counts
- total image count

Output:

- `project_2/shreyas/artifacts/dataset_summary.json`

Optional output path:

```bash
python project_2/shreyas/preprocess_dataset.py --dataset-dir training_images_augmented --output custom_summary.json
```

### 2) Train / Fine-Tune VGGFace

```bash
python project_2/shreyas/train_vggface.py --dataset-dir training_images_augmented --epochs 25 --batch-size 16
```

Training details:

- pretrained VGGFace (`vgg16`, no top, global average pooling)
- layers frozen except `conv5*`
- head: `Dense(256, relu) -> Dropout(0.40) -> Dense(num_classes, softmax)`
- loss: categorical cross-entropy
- optimizer: Adam (`1e-4` default learning rate)

Data pipeline:

- resize to `224 x 224`
- RGB conversion
- VGGFace preprocessing (`version=1`)
- augmentation on training split: rotation, shifts, zoom, brightness, horizontal flip

Callbacks:

- best-checkpoint by validation accuracy
- early stopping (patience 6, restore best)
- reduce LR on validation loss plateau

Outputs:

- `project_2/shreyas/artifacts/vggface_friends.keras`
- `project_2/shreyas/artifacts/class_indices.json`
- `project_2/shreyas/artifacts/training_history.csv`

### 3) Live Webcam Recognition

```bash
python project_2/shreyas/live_webcam_vggface.py --camera-index 0 --confidence 0.60 --process-every 2
```

What it does:

- captures webcam frames
- detects faces using OpenCV Haar cascade
- classifies each face crop with trained VGGFace model
- overlays label + confidence + FPS
- assigns `Unknown` if confidence is below threshold
- press `q` to quit

If `--class-map` is missing but `--dataset-dir` exists, the script can generate the class map automatically.

## Default Configuration

Defined in `vggface_config.py`:

- input size: `224 x 224`
- batch size: `16`
- validation split: `0.20`
- random seed: `42`
- learning rate: `1e-4`
- epochs: `25`
- early stopping patience: `6`
- confidence threshold: `0.60`
- detection scale: `0.50`
- process every N frames: `2`

## Command Reference

Dataset summary:

```bash
python project_2/shreyas/preprocess_dataset.py --dataset-dir training_images_augmented
```

Training:

```bash
python project_2/shreyas/train_vggface.py --dataset-dir training_images_augmented --epochs 25 --batch-size 16
```

Webcam inference:

```bash
python project_2/shreyas/live_webcam_vggface.py --camera-index 0 --confidence 0.60 --process-every 2
```

Common optional webcam flags:

- `--model` to load a different model path
- `--class-map` to load a different class map path
- `--detection-scale` for speed vs detection quality tradeoff
- `--dataset-dir` fallback source for class-map generation

## Practical Tuning Notes

If webcam is laggy:

- increase `--process-every` to `3` or `4`
- reduce camera resolution from your webcam settings if available

If wrong labels are frequent:

- raise threshold (example: `--confidence 0.70`)

If too many faces become `Unknown`:

- lower threshold slightly (example: `--confidence 0.50`)

If training overfits:

- reduce epochs
- increase regularization/dropout
- improve class balance and data diversity
- review augmentation strength

## Notes and Limitations

- This is a closed-set classifier for known identities, not open-world face verification.
- Accuracy depends strongly on image quality, class balance, and lighting diversity.
- Haar cascade detection is lightweight but less robust than modern deep detectors.
- Real-time speed depends on CPU/GPU capability, webcam settings, and inference frequency.

## Typical Workflow

```bash
python project_2/shreyas/preprocess_dataset.py --dataset-dir training_images_augmented
python project_2/shreyas/train_vggface.py --dataset-dir training_images_augmented --epochs 25 --batch-size 16
python project_2/shreyas/live_webcam_vggface.py --camera-index 0 --confidence 0.60 --process-every 2
```
