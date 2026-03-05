"""
Configuration for the face data preprocessing & augmentation pipeline.
All tuneable parameters in one place.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
# Root of the repository
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Input videos: Training_Videos/<Person>/*.mp4
VIDEO_ROOT = os.path.join(REPO_ROOT, "..", "Training_Videos")

# Output directory for cropped face images
IMAGE_OUTPUT_ROOT = os.path.join(REPO_ROOT, "augmented_data", "images")

# ─── GPU / Model ─────────────────────────────────────────────────────────────
# Which GPU to use for InsightFace face detection (0-indexed)
GPU_ID = 1

# InsightFace model name (buffalo_l is the most accurate)
INSIGHTFACE_MODEL = "buffalo_l"

# Detection size for InsightFace
DET_SIZE = (640, 640)

# ─── Frame Sampling ─────────────────────────────────────────────────────────
# Target number of RAW face crops per person (across all their videos)
RAW_IMAGES_PER_PERSON = 150

# Minimum Laplacian variance to keep a frame (blur filter)
# Lower = more lenient, higher = stricter
BLUR_THRESHOLD = 30.0

# Minimum face detection confidence score
FACE_CONFIDENCE_THRESHOLD = 0.5

# ─── Face Crop ───────────────────────────────────────────────────────────────
# Output resolution for face crops
FACE_CROP_SIZE = (200, 200)

# Padding factor around detected face bbox (1.0 = no padding, 1.7 = 70% extra)
FACE_PADDING = 1.7

# ─── Augmentation ────────────────────────────────────────────────────────────
# Target TOTAL images per person (raw + augmented)
TOTAL_IMAGES_PER_PERSON = 600

# JPEG quality for saved images
JPEG_QUALITY = 95

# Random seed for reproducibility
RANDOM_SEED = 42

# ─── Video Extensions ───────────────────────────────────────────────────────
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".wmv")
