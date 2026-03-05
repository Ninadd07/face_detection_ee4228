# Augmented Data — Face Preprocessing & Augmentation Pipeline

This module handles **dataset preprocessing** for the EE4228 face detection project. It takes raw training videos of each team member and produces a clean, augmented dataset of cropped face images ready for downstream model training.

## Overview

| Stage | Description | Output |
|-------|-------------|--------|
| **Stage 1** — Frame Sampling | Extracts frames from videos, detects & crops faces using InsightFace, filters blurry frames | `images/<Person>/raw_*.jpg` |
| **Stage 2** — Augmentation | Applies geometric, photometric, occlusion, and weather augmentations to reach target count | `images/<Person>/aug_*.jpg` |

## Pipeline Summary

```
Training_Videos/<Person>/*.mp4          (Input: 3–6 videos per person)
        │
        ▼  Stage 1: frame_sampler.py
        │  - InsightFace face detection (buffalo_l)
        │  - Single-face filter (exactly 1 face per frame)
        │  - Laplacian blur filtering on cropped face
        │  - Face cropping with 70% padding → 200×200 JPEG
        │
images/<Person>/raw_vid{i}_frame{n}.jpg  (150 raw images per person)
        │
        ▼  Stage 2: augmentor.py
        │  - Geometric: flip, rotation ±15°, affine scale/translate
        │  - Photometric: brightness/contrast, color jitter, noise
        │  - Occlusion: random rectangular erasing (CoarseDropout)
        │  - Weather: rain streaks, snow particles, fog/haze overlay
        │
images/<Person>/aug_*.jpg                (450 augmented per person)
        │
        ▼  Final output
600 face images per person, 4200 total across 7 members
```

## Directory Structure

```
augmented_data/
├── README.md               ← You are here
├── .venv/                   ← Python virtual environment
├── scripts/
│   ├── config.py            ← All configurable parameters
│   ├── frame_sampler.py     ← Stage 1: video → face crops
│   ├── augmentor.py         ← Stage 2: face crops → augmented dataset
│   ├── pipeline.py          ← Master script (runs both stages)
│   └── requirements.txt     ← Python dependencies
└── images/                  ← Generated output (gitignored)
    ├── Abhiram/
    │   ├── raw_vid0_frame00030.jpg
    │   ├── aug_vid0_frame00030_geometric_0001.jpg
    │   └── ...
    ├── Frentzen/
    ├── Jessica/
    ├── Ninad/
    ├── Ryan/
    ├── Sasi/
    └── Shreyas/
```

## Quick Start

### 1. Set up the environment

```bash
cd face_detection_ee4228/augmented_data

# Create virtual environment (already done if .venv exists)
python3 -m venv .venv

# Install dependencies
.venv/bin/pip install -r scripts/requirements.txt
```

### 2. Run the full pipeline

```bash
cd scripts
../.venv/bin/python pipeline.py
```

### 3. Run individual stages

```bash
# Stage 1 only (frame sampling)
../.venv/bin/python pipeline.py --stage1-only

# Stage 2 only (augmentation, requires Stage 1 output)
../.venv/bin/python pipeline.py --stage2-only

# Dry run (show config & dataset overview, no processing)
../.venv/bin/python pipeline.py --dry-run
```

## Configuration

All parameters are in [`scripts/config.py`](scripts/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GPU_ID` | `1` | GPU for InsightFace (0-indexed) |
| `RAW_IMAGES_PER_PERSON` | `150` | Target raw face crops per person |
| `TOTAL_IMAGES_PER_PERSON` | `600` | Target total (raw + augmented) per person |
| `FACE_CROP_SIZE` | `(200, 200)` | Output face crop resolution |
| `BLUR_THRESHOLD` | `30.0` | Laplacian variance threshold for blur detection (applied to cropped face only) |
| `FACE_PADDING` | `1.7` | Padding around detected face bbox (1.7 = 70% extra) |
| `FACE_CONFIDENCE_THRESHOLD` | `0.5` | Minimum InsightFace detection score |

## Augmentation Details

### Augmentation Category Weights

| Category | Weight | Techniques |
|----------|--------|------------|
| **Geometric** | 25% | Horizontal flip, rotation ±15°, affine scale/translate |
| **Photometric** | 25% | Brightness/contrast ±25%, hue/saturation shift, Gaussian noise |
| **Combined** | 20% | Mix of geometric + photometric + occlusion |
| **Occlusion** | 10% | CoarseDropout (1–3 random rectangular patches) |
| **Rain** | 7% | Semi-transparent diagonal streaks + darkening |
| **Snow** | 6% | White particle overlay + slight brightening |
| **Fog** | 7% | White haze overlay + contrast reduction |

### Why 200×200?

- ArcFace expects 112×112 — downscaling from 200 is trivial and lossless
- FaceNet expects 160×160 — downscaling from 200 preserves quality
- Larger crop size captures more facial context (hair, ears, jawline) for better recognition
- The 70% padding (`FACE_PADDING=1.7`) ensures the full head region is included

## Output Format

- **Format:** JPEG (quality 95)
- **Resolution:** 200×200 pixels
- **Naming convention:**
  - Raw: `raw_vid{video_index}_frame{frame_number:05d}.jpg`
  - Augmented: `aug_{base_id}_{category}_{index:04d}.jpg`
- **Organization:** One folder per person under `images/`

## Dependencies

- `insightface` — Face detection & alignment (buffalo_l model)
- `onnxruntime-gpu` — GPU inference for InsightFace
- `opencv-python-headless` — Video I/O, image processing
- `albumentations` — Image augmentation transforms
- `numpy` — Numerical operations

## Hardware

This pipeline was developed for the MLDA GPU server (gpu8):
- **GPU 1:** NVIDIA RTX 2080 Ti (11GB) — used for InsightFace inference
- Augmentation (Stage 2) is CPU-only and does not require a GPU

## Notes for Downstream Users

- All images are **cropped face regions** (not full frames)
- Images are **not normalized** — pixel values are in [0, 255] uint8
- To use with ArcFace: resize to 112×112 and normalize as per model requirements
- To use with FaceNet: resize to 160×160
- The dataset is balanced: 600 images per person

## Dataset Statistics (Last Run)

| Person | Videos | Raw Images | Augmented | Total | Blur Skips | No-Face Skips |
|--------|--------|------------|-----------|-------|------------|---------------|
| Abhiram | 3 | 150 | 450 | 600 | 0 | 0 |
| Frentzen | 4 | 150 | 450 | 600 | 0 | 0 |
| Jessica | 4 | 150 | 450 | 600 | 0 | 0 |
| Ninad | 4 | 150 | 450 | 600 | 0 | 0 |
| Ryan | 4 | 150 | 450 | 600 | 6 | 0 |
| Sasi | 4 | 150 | 450 | 600 | 3 | 0 |
| Shreyas | 6 | 150 | 450 | 600 | 28 | 17 |
| **Total** | **29** | **1050** | **3150** | **4200** | **37** | **17** |

- **Total pipeline time:** ~167 seconds (Stage 1: 161s, Stage 2: 4s)
- **Single-face filter:** Frames with ≠1 detected face are skipped (ensures clean, single-person crops)
