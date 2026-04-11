# EE4228 Intelligent System Design

This repository contains the team codebase for EE4228 face detection and recognition work across two project phases:

- Project 1: traditional machine learning baseline work
- Project 2: deep-learning-based face recognition pipelines, evaluation, and live demo scripts

The repo is organized as a practical engineering workspace rather than a single monolithic package. Different subfolders contain independent pipelines that can be run separately.

## Repository Overview

```text
face_detection_ee4228/
|-- augmented_data/              # video-to-image sampling and augmentation pipeline
|-- data_raw/                    # raw source data notes (videos/images managed externally)
|-- project_1/                   # Project 1 materials (traditional ML track)
|-- project_2/
|   |-- evaluation/              # evaluation-related notes/assets
|   |-- frentzen/                # FaceNet/MobileFaceNet experiments and scripts
|   |-- models/
|   |   |-- arcface_insightface/         # ArcFace + InsightFace modular backend
|   |   |-- arcface_insightface_old/     # older ArcFace implementation
|   |   `-- mobilefacenet_insightface/   # MobileFaceNet variant with shared tooling
|   |-- shreyas/                 # VGGFace transfer-learning pipeline + webcam demo
|   `-- ui/                      # UI/UX assets and notes for Project 2
|-- training_images_augmented/   # per-identity image folders used for training
`-- README.md                    # you are here
```

## What Is Implemented

### 1) Dataset Preparation and Augmentation

`augmented_data/scripts/` includes a two-stage pipeline:

- Stage 1: sample frames from videos and crop faces
- Stage 2: augment face crops to expand dataset diversity

Primary purpose: produce balanced per-person image folders for downstream training.

### 2) ArcFace + InsightFace Pipeline (Project 2)

`project_2/models/arcface_insightface/` provides a modular recognition backend:

- configurable dataset split (enroll/val/test)
- database enrollment from known identities
- threshold sweep and evaluation metrics export
- live webcam recognition with unknown rejection

This path is designed for reproducible evaluation and deployment-style live testing.

### 3) MobileFaceNet Variant (Project 2)

`project_2/models/mobilefacenet_insightface/` reuses the same detection/evaluation flow while swapping the embedding backbone to MobileFaceNet for comparison.

### 4) VGGFace Fine-Tuning Pipeline (Project 2)

`project_2/shreyas/` contains an end-to-end transfer-learning workflow:

- dataset validation summary
- VGGFace (VGG16 backbone) fine-tuning
- artifact export (`.keras`, class map, history)
- live webcam inference with confidence thresholding

## Data Layout Expectations

Most training scripts expect one folder per identity under `training_images_augmented/`, for example:

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

Each identity folder should contain image files (commonly jpg/jpeg/png/bmp depending on the script).

Raw videos are documented under `data_raw/` and are typically managed outside git due to size.

## Environment and Dependency Strategy

This repository contains multiple independent pipelines with different dependencies. Use separate virtual environments per workflow when possible.

Typical pattern:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r <workflow-specific-requirements.txt>
```

Examples:

- `augmented_data/scripts/requirements.txt`
- `project_2/shreyas/requirements_vggface.txt`
- `project_2/models/mobilefacenet_insightface/requirements.txt`

Some ArcFace-related code paths currently rely on local environment setup and may require manual dependency alignment.

## Quick Start Paths

Choose one of the following tracks depending on your task.

### Track A: Build/refresh augmented dataset

1. Open `augmented_data/README.md` for full pipeline details.
2. Run `pipeline.py` (or stage-specific commands) from `augmented_data/scripts/`.
3. Verify output identity folders under `training_images_augmented/`.

### Track B: Run ArcFace evaluation flow

1. Configure paths and thresholds in `project_2/models/arcface_insightface/config.py`.
2. Run split -> enroll -> validation threshold sweep -> test evaluation.
3. Launch webcam demo with selected threshold.

### Track C: Train and run VGGFace webcam demo

1. Follow `project_2/shreyas/README.md` setup.
2. Run dataset preprocessing summary.
3. Train VGGFace model.
4. Run live webcam recognition.

## Suggested End-to-End Experiment Flow

If starting from scratch:

1. Prepare raw videos/images and validate folder structure.
2. Generate/refresh `training_images_augmented/` using augmentation pipeline.
3. Select model family (ArcFace, MobileFaceNet, or VGGFace).
4. Train/enroll and run validation threshold tuning.
5. Evaluate on held-out test data.
6. Run webcam demo for qualitative checks.
7. Store metrics/artifacts under model-specific artifact folders for comparison.

## Where Outputs Are Saved

Outputs are model-specific and remain inside each workflow folder, for example:

- `project_2/models/arcface_insightface/artifacts/`
- `project_2/models/arcface_insightface/artifacts_finetuned/`
- `project_2/models/mobilefacenet_insightface/artifacts_mobilefacenet/`
- `project_2/shreyas/artifacts/`

This layout keeps experiments isolated and makes side-by-side comparison easier.

## Known Limitations and Practical Notes

- Some subprojects are actively evolving and may have placeholder README files.
- Dependency versions differ across pipelines; isolated environments are recommended.
- Large datasets/artifacts are not fully tracked in git; team-shared storage may be required.
- Real-time inference performance depends on hardware, camera resolution, and model choice.

## Documentation Map

For detailed usage, refer to:

- `augmented_data/README.md`
- `project_2/models/arcface_insightface/README.md`
- `project_2/models/mobilefacenet_insightface/README.md`
- `project_2/shreyas/README.md`

## Maintainers

This repository is maintained by the EE4228 project team. Keep workflow-specific changes documented in the corresponding subfolder README so the root README can stay concise and navigational.
