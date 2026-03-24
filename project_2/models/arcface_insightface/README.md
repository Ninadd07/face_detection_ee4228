# ArcFace InsightFace Backend (EE4228 Project 2)

This folder contains a modular **ArcFace + InsightFace** backend for face recognition with:

- Proper **dataset split** (enroll / val / test)
- **Database enrollment** from images
- **Evaluation** with threshold sweep and unknown handling
- **Live webcam demo** with unknown label

The code is designed so that evaluation and live demo share the same backend.

---

## 1. Folder Layout (expected)

From the project root:

```text
project_2/
  models/
    arcface_insightface/
      config.py
      utils.py
      split_dataset.py
      detector.py
      embedder.py
      database.py
      matcher.py
      enroll.py
      evaluate.py
      live_demo.py
      extract_clean_frames.py
      artifacts/
        embeddings/
        metrics/
        logs/
      data/
        split/
          enroll/
          val/
          test/
        unknown/
```

Raw data is **outside** this folder (see `config.py`).

---

## 2. Setup

### 2.1. Dependencies

I still need to update my requirements.txt!

### 2.2. Configure Paths

Open `config.py` and set at least:

- `RAW_DATA_DIR` – absolute path to your raw dataset with per-person folders, e.g.:

  ```python
  RAW_DATA_DIR = Path("/Users/jessica/datasets/face/raw")
  ```

- (Optional but recommended) check:

  ```python
  LOCAL_DATA_DIR   # local data/ under this model
  SPLIT_DIR        # LOCAL_DATA_DIR / "split"
  ENROLL_DIR, VAL_DIR, TEST_DIR
  UNKNOWN_DIR      # LOCAL_DATA_DIR / "unknown"
  ARTIFACTS_DIR
  EMBEDDINGS_DIR
  METRICS_DIR
  LOGS_DIR
  ```

`config.py` also defines:

- `RANDOM_SEED` – to keep splits reproducible
- `ENROLL_RATIO`, `VAL_RATIO`, `TEST_RATIO` – split proportions
- `UNKNOWN_LABEL` – string used for unknown faces
- `DEFAULT_THRESHOLD` – default cosine similarity threshold
- InsightFace model settings (e.g. `MODEL_NAME`, `DETSIZE`)

All scripts import these constants instead of hardcoding paths.

---

## 3. Pipeline Overview

The typical workflow is:

1. **Split dataset** into enroll / val / test
2. **Enroll** database (build prototypes)
3. **Evaluate** on val (threshold sweep)
4. **Evaluate** on test with chosen threshold
5. **Run live demo** using the same database and threshold

Note: If Augmented Dataset is ready, please point config towards this dataset and directly run split_dataset, enroll and so on. No need to extract data.

### 3.1. split_dataset.py

Purpose:  
Create per-identity **enroll / val / test** splits from the raw data so that evaluation uses unseen images.

How it works:

- Reads folders from `RAW_DATA_DIR / person_name`
- Shuffles images with `RANDOM_SEED`
- Splits according to `ENROLL_RATIO`, `VAL_RATIO`, `TEST_RATIO`
- Copies files into:

  ```text
  ENROLL_DIR / person_name
  VAL_DIR / person_name
  TEST_DIR / person_name
  ```

Run:

```bash
python3 -m project_2.models.arcface_insightface.split_dataset
```

---

### 3.2. utils.py

Purpose:  
Common helper functions used across scripts.

Key functions:

- `seed_everything(seed)` – seeds Python / NumPy RNGs
- `ensure_dir(path)` / `ensure_dirs([...])` – create directories if missing
- `list_images(folder)` – list image files by extension
- `load_image(path)` – load BGR image with OpenCV
- `save_json(obj, path)` – save JSON with parent directory creation
- `normalize_embedding(emb)` – L2-normalize embedding vectors

You normally do not run this directly; other scripts import it.

---

### 3.3. detector.py

Purpose:  
Wrap InsightFace **FaceAnalysis** for face detection.

Key points:

- Initializes InsightFace with `MODEL_NAME`, `DETSIZE`, `CTX_ID` from `config.py`
- `FaceDetector.detect(image)`:
  - Input: BGR image (NumPy array)
  - Output: list of detections, each a dict like:

    ```python
    {
      "bbox": [x1, y1, x2, y2],
      "face_obj": <InsightFace face object>,
      "embedding": <512-dim vector or None>,
      "det_score": float
    }
    ```

Used by `enroll.py`, `evaluate.py`, `live_demo.py`.

---

### 3.4. embedder.py

Purpose:  
Convert a detection into a **normalized ArcFace embedding**.

Baseline version:

- `ArcFaceEmbedder.get_embedding_from_detection(detection)`:
  - Takes a detection from `FaceDetector.detect`
  - Extracts the embedding from the InsightFace face object
  - L2-normalizes it (via `normalize_embedding`)
  - Returns a 1D float32 vector, or `None` on failure

Abstraction point: later you can replace this with a fine-tuned embedder but keep the same interface.

---

### 3.5. database.py

Purpose:  
Store and load **identity prototypes**.

Core class: `FaceDatabase`

- `add_identity(label, embeddings)`:
  - `embeddings` is a list/array of 512D vectors for one person
  - Computes **mean vector**, normalizes it
  - Stores label and prototype

- `save(path)` / `load(path)`:
  - `save`: saves labels and prototypes to `.npz`
  - `load`: loads them back into a `FaceDatabase` instance

Artifacts:

- Baseline database typically saved as:

  ```text
  artifacts/embeddings/baseline_enroll.npz
  ```

---

### 3.6. matcher.py

Purpose:  
Matching logic – take one embedding and decide which identity (or unknown).

Key components:

- `cosine_similarity(a, b)` – standard cosine similarity
- `FaceMatcher(labels, prototypes, threshold, unknown_label)`:
  - `match(embedding)`:
    - Computes cosine similarity to each prototype
    - Picks max score
    - If `score >= threshold`: returns `(label, score)`
    - Else returns `(UNKNOWN_LABEL, score)`

Used in evaluation and live demo.

---

### 3.7. enroll.py

Purpose:  
Build the **face database** from the enroll split.

How it works:

- Loads `FaceDetector` and `ArcFaceEmbedder`
- Iterates over `ENROLL_DIR / person_name`
- For each image:
  - Detect faces
  - If detection fails or multiple faces, skip (or keep largest, depending on version)
  - Get embedding
  - Collect embeddings per identity
- Calls `FaceDatabase.add_identity(label, embeddings)` per person
- Saves:
  - `artifacts/embeddings/baseline_enroll.npz`
  - `baseline_enroll_summary.json` (per-identity stats: total / used / skipped)

Run:

```bash
python3 -m project_2.models.arcface_insightface.enroll
```

---

### 3.8. evaluate.py

Purpose:  
Evaluate recognition performance and tune the **threshold**.

Two main modes:

1. **Sweep mode** (`--sweep`):
   - Runs on `VAL_DIR` (and `UNKNOWN_DIR` if exists)
   - For a list of thresholds (e.g. 0.30–0.70, which can edit for greater sweep as per model):
     - Applies threshold to cached records
     - Computes:
       - `overall_accuracy`
       - `known_accuracy`
       - `unknown_rejection_rate`
   - Selects best threshold using:
     - known only, or a combination of known + unknown
   - Saves to:

     ```text
     artifacts/metrics/baseline_val_threshold_sweep.json
     ```

2. **Single-threshold mode** (no `--sweep`):
   - Applies a fixed threshold on `val` or `test`
   - Saves metrics and records:

     ```text
     artifacts/metrics/baseline_val_metrics.json
     artifacts/metrics/baseline_val_records.json
     artifacts/metrics/baseline_test_metrics.json
     artifacts/metrics/baseline_test_records.json
     ```

Run examples:

```bash
# Sweep on validation to find best threshold
python3 -m project_2.models.arcface_insightface.evaluate --split val --sweep

# Evaluate final performance on test at chosen threshold
python3 -m project_2.models.arcface_insightface.evaluate --split test --threshold 0.50
```

`evaluate.py` uses:

- `FaceDetector`, `ArcFaceEmbedder`, `FaceDatabase`, `FaceMatcher`
- `UNKNOWN_DIR` for unknown faces if available

---

### 3.9. live_demo.py

Purpose:  
Run **real-time webcam recognition** using the enrolled database and chosen threshold.

How it works:

- Loads:
  - `FaceDatabase` (e.g. `baseline_enroll.npz`)
  - `FaceDetector`
  - `ArcFaceEmbedder`
  - `FaceMatcher(threshold=...)`
- Opens webcam (default camera index 0)
- Main loop:
  - Optionally downscale frame (`--scale`, e.g. 0.5)
  - Run detection every N frames (`--detect-every`, e.g. 2)
  - For each detection:
    - Get embedding
    - Match via `FaceMatcher.match`
  - Draw:
    - Green box + name + score if known
    - Red box + “Unknown” if below threshold
  - Show FPS approximately every 30 frames

Run:

```bash
python3 -m project_2.models.arcface_insightface.live_demo \
  --db artifacts/embeddings/baseline_enroll.npz \
  --threshold 0.45 \
  --scale 0.5 \
  --detect-every 2
```

Press `q` to quit.

---

### 3.10. extract_clean_frames.py

Purpose:  
(Optional utility) Clean/extract **good quality frames** from training videos to build an image dataset.

Typical behavior (depending on your implementation):

- Opens person-specific video(s)
- Uses `FaceDetector` to find faces
- Applies filters:
  - Exactly one face in frame
  - Sufficient size / non-blurry (e.g. Laplacian variance)
- Crops face regions with some padding
- Saves them as clean face images into some folder (e.g. `data/frames_clean/person_name/`)

This script is useful if your raw data is mainly videos and you want consistent face crops before running augmentation or splitting.

---

### 3.11. config.py (recap)

Central configuration file. Controls:

- Paths
- Split ratios
- Threshold defaults
- InsightFace model config

All other scripts rely on `config.py`, so **update it first** when moving between machines.

---

## 4. End-to-End Command Summary

From **project root**, after configuring `config.py`:

```bash
# 1. Split raw dataset into enroll / val / test
python3 -m project_2.models.arcface_insightface.split_dataset

# 2. Enroll database from enroll split
python3 -m project_2.models.arcface_insightface.enroll

# 3. Threshold sweep on validation
python3 -m project_2.models.arcface_insightface.evaluate --split val --sweep

# 4. Once best threshold is chosen, evaluate on test
python3 -m project_2.models.arcface_insightface.evaluate --split test --threshold <BEST_THRESHOLD>

# 5. Run live demo with same database and threshold
python3 -m project_2.models.arcface_insightface.live_demo \
  --threshold <BEST_THRESHOLD>
```

Replace `<BEST_THRESHOLD>` with the one from the validation sweep JSON.
