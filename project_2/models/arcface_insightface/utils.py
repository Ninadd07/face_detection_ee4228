import json
import random
from pathlib import Path

import cv2
import numpy as np

from .config import IMAGE_EXTS, RANDOM_SEED


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths):
    for p in paths:
        ensure_dir(p)


def list_images(folder):
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def normalize_embedding(emb):
    emb = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm
