"""
config.py

Central configuration for the ArcFace + InsightFace model.

This file defines:
- Folder paths (data, embeddings)
- Model / detection parameters
- Embedding extraction parameters
- Recognition thresholds
"""

from pathlib import Path

# -----------------------------
# Root paths
# -----------------------------

# Project root (this file lives in project_2/models/arcface_insightface/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Shared raw data folder (used by all models)
DATA_ROOT = PROJECT_ROOT / "data_raw"

# Augmented training images produced by GAN:
# augmented_data/training_images_augmented/<Person>/*.jpg
AUG_IMG_ROOT = PROJECT_ROOT / "augmented_data/training_images_augmented"

# Optional: original training videos, if we still need them
VIDEO_ROOT = DATA_ROOT / "videos"

# Embeddings folder for THIS model only.
# Each model should have its own embeddings root so they don't clash.
# Structure:
#   embeddings_arcface/<Person>/*.npy
EMB_ROOT = PROJECT_ROOT / "embeddings/embeddings_arcface"


# -----------------------------
# InsightFace / detection config
# -----------------------------

# Detection input size for FaceAnalysis (SCRFD)
# This is passed into app.prepare(det_size=DET_SIZE)
DET_SIZE = (640, 640)

# Providers for ONNXRuntime. On Mac (no GPU), CPU is simplest.
PROVIDERS = ["CPUExecutionProvider"]


# -----------------------------
# Embedding extraction config
# -----------------------------

# ArcFace typically expects 112x112 aligned face inputs.
# GAN crops are ~200x200 faces, so we will resize them to 112x112 before feeding to the embedding model.
ARC_FACE_INPUT_SIZE = (112, 112)  # (width, height)

# When extracting embeddings from videos (if used):
FRAME_STEP = 10           # sample every Nth frame
MAX_EMBEDDINGS_PER_VIDEO = 50  # safety cap per video

# When extracting from augmented images, we process all images.
# We can still define a max cap per person if needed:
MAX_EMBEDDINGS_PER_PERSON = 600  # Our dataset size per person


# -----------------------------
# Recognition config
# -----------------------------

# Label used when no person matches above threshold
UNKNOWN_LABEL = "Unknown"

# Cosine similarity threshold for prototype matching.
# Typical ArcFace ranges are around 0.6–0.7; tune on your dataset.
PROTOTYPE_THRESHOLD = 0.6

# If you also experiment with the all-samples method, you can have a
# slightly different threshold for that (often a bit lower).
ALL_SAMPLES_THRESHOLD = 0.6


# -----------------------------
# Utility flags / options
# -----------------------------

# If True, engine prints extra debug info (e.g., loaded persons, shapes).
DEBUG = True
