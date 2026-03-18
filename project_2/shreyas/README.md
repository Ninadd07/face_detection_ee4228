# VGGFace Fine-Tuning (Shreyas)

This folder contains a complete VGGFace pipeline for 7-friend face recognition using:

- Dataset: `training_images_augmented/<friend_name>/*.jpg`
- Model: pre-trained VGGFace (VGG16 backbone)
- Output: trained classifier + live webcam prediction with face boxes and names

## Files

- `vggface_config.py` : central paths + hyperparameters
- `preprocess_dataset.py` : dataset validation + summary export
- `train_vggface.py` : fine-tunes VGGFace and saves artifacts
- `live_webcam_vggface.py` : webcam detection + recognition overlay
- `requirements_vggface.txt` : Python dependencies

## Setup

From repo root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r project_2/shreyas/requirements_vggface.txt
```

## 1) Preprocess / Validate Dataset

```bash
python project_2/shreyas/preprocess_dataset.py --dataset-dir training_images_augmented
```

Outputs:

- `project_2/shreyas/artifacts/dataset_summary.json`

## 2) Train and Save Model

```bash
python project_2/shreyas/train_vggface.py --dataset-dir training_images_augmented --epochs 25 --batch-size 16
```

Outputs:

- `project_2/shreyas/artifacts/vggface_friends.keras`
- `project_2/shreyas/artifacts/class_indices.json`
- `project_2/shreyas/artifacts/training_history.csv`

Notes:

- Early convolution blocks are frozen; only top layers + `conv5*` are fine-tuned.
- This is tuned for stability on Intel Iris Xe class systems (CPU-safe path, moderate batch size).

## 3) Live Webcam Prediction

```bash
python project_2/shreyas/live_webcam_vggface.py --camera-index 0 --confidence 0.60 --process-every 2
```

Behavior:

- Opens webcam and detects faces.
- Predicts class for each face.
- Draws bounding box + friend name (`Unknown` if confidence below threshold).
- Press `q` to quit.

## Tuning Tips

- If webcam is laggy, increase `--process-every` to `3` or `4`.
- If too many wrong labels, increase `--confidence` (e.g. `0.70`).
- If too many `Unknown` labels, lower `--confidence` slightly.
