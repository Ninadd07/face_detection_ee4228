# mobilefacenet_insightface — Technical Report

## Overview

This package evaluates **MobileFaceNet** as a face recognition backbone using the same
detection infrastructure and evaluation methodology as the `arcface_insightface` baseline.
It is designed as a sibling package: detection, gallery storage, matching, and dataset
splits are all imported from `arcface_insightface` without modification; the only component
that differs is the embedding model.

The evaluation follows a three-phase workflow — enrollment, threshold selection, and
final testing — which is described in full below.

---

## Package Structure

```
mobilefacenet_insightface/
├── __init__.py       # empty; marks this as a Python package
├── config.py         # output paths and checkpoint resolution
├── embedder.py       # MobileFaceNet embedding adapter
├── enroll.py         # Phase 1: build embedding gallery
├── evaluate.py       # Phase 2 & 3: threshold sweep and final evaluation
├── requirements.txt  # pip dependencies
└── README.md         # this document
```

Dependencies imported from `arcface_insightface` (read-only, unchanged):

| Symbol | Source file | Purpose |
|--------|-------------|---------|
| `FaceDetector` | `detector.py` | InsightFace buffalo_l detection |
| `FaceDatabase` | `database.py` | Gallery storage and loading |
| `cosine_similarity` | `matcher.py` | Similarity function |
| `ENROLL_DIR`, `VAL_DIR`, `TEST_DIR`, `UNKNOWN_DIR` | `config.py` | Dataset split paths |
| `DEFAULT_THRESHOLD`, `UNKNOWN_LABEL` | `config.py` | Evaluation constants |
| `list_images`, `load_image`, `save_json`, `ensure_dirs` | `utils.py` | I/O utilities |

---

## System Architecture

The recognition pipeline consists of four sequential stages:

```
Input image (BGR)
      │
      ▼
┌─────────────────────────────────┐
│  1. Face Detection               │
│  InsightFace buffalo_l (MTCNN-  │
│  style cascade, 640×640 input)  │
│  → bounding boxes + 5 landmarks │
└─────────────────────────────────┘
      │  face_obj.kps  (shape 5×2)
      ▼
┌─────────────────────────────────┐
│  2. Face Alignment               │
│  ArcFace-style affine warp      │
│  → normalised 112×96 RGB crop   │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  3. Embedding                    │
│  MobileFaceNet backbone         │
│  → 128-dim L2-normalised vector │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  4. Gallery Matching             │
│  Cosine similarity vs. enrolled │
│  identity prototypes            │
│  → predicted label + score      │
└─────────────────────────────────┘
```

The key architectural difference from the ArcFace baseline is **stage 3**: ArcFace
produces a 512-dimensional embedding directly from the InsightFace detection object,
whereas MobileFaceNet requires an explicit alignment step (stage 2) and produces a
128-dimensional embedding via a separate forward pass.

---

## Module Descriptions

### `config.py` — Path and Checkpoint Resolution

`config.py` is imported at module load time and sets up all output paths and the
checkpoint location. It creates the artifact directories immediately on import so that
other modules can write to them without additional setup.

**Artifact directories** are anchored to the package's own directory (`_PACKAGE_DIR`),
making them portable regardless of where the repo is cloned:

```
mobilefacenet_insightface/
└── artifacts_mobilefacenet/
    ├── embeddings/   ← gallery .npz files
    └── metrics/      ← JSON evaluation outputs
```

**Checkpoint resolution** uses a priority order to avoid hardcoded paths:

1. The `MFN_CHECKPOINT` environment variable — allows any user to override without
   touching code.
2. A fallback that constructs the path relative to the package directory by traversing
   four levels up to the ISD repo root and appending the standard frentzen checkpoint
   location (`frentzen/frentzen/artifacts/checkpoints/mobilefacenet_best.pth`).

The directory traversal used in the fallback is:
```
_PACKAGE_DIR.parents[0] = models/
_PACKAGE_DIR.parents[1] = project_2/
_PACKAGE_DIR.parents[2] = face_detection_ee4228/
_PACKAGE_DIR.parents[3] = ISD/  (repo root)
```

---

### `embedder.py` — MobileFaceNet Embedding Adapter

`embedder.py` bridges two different codebases: the InsightFace detection pipeline
(used in `arcface_insightface`) and the MobileFaceNet implementation (in the `frentzen`
package).

**Class: `MobileFaceNetEmbedder`**

On construction, two objects are initialised:

- `_MFNEmbedder` — the MobileFaceNet backbone loaded from `frentzen`, with
  `embedding_dim=128` and `input_size=(112, 96)`. The checkpoint is loaded at this
  point; a `FileNotFoundError` is raised if the checkpoint path does not exist.
  Device selection is automatic: CUDA if available, otherwise CPU.

- `FaceAligner` — the ArcFace-style affine aligner from `frentzen`, configured for
  a 112×96 output crop.

The `mobilefacenet_face` package must be installed for these imports to succeed. If it
is not installed, a `ImportError` is raised with an explicit install command rather than
a generic module-not-found message.

**Method: `get_embedding(image_bgr, detection) → np.ndarray | None`**

This method takes a full BGR frame and one detection dict (as produced by
`FaceDetector.detect()`) and returns a 128-dim float32 embedding, or `None` on failure.

The processing steps are:

1. **Landmark extraction.** The 5-point facial landmarks are read from
   `detection["face_obj"].kps`. These are the right eye, left eye, nose tip, right
   mouth corner, and left mouth corner, provided as a float32 array of shape `(5, 2)`.
   If `kps` is absent or has an unexpected shape, `None` is returned immediately.

2. **Colour conversion.** The input frame is in BGR order (OpenCV convention). It is
   converted to RGB before passing to the aligner, which follows the `frentzen`
   convention.

3. **Affine alignment.** `FaceAligner.align()` estimates a partial affine transform
   (scale + rotation, no shear) that maps the 5 detected landmarks to a canonical
   ArcFace template. The output is a 112×96 RGB crop. If the transform cannot be
   estimated (degenerate landmark configuration), a `ValueError` is caught and `None`
   is returned.

4. **Forward pass.** The aligned crop is passed to `_MFNEmbedder.embed()`, which
   normalises the pixel values to `[-1, 1]` via `(x - 127.5) / 128.0`, runs the
   MobileFaceNet forward pass, and L2-normalises the output. The returned array has
   shape `(1, 128)`; the single row is extracted and returned.

Because all embeddings are L2-normalised to unit vectors, the dot product between any
two embeddings equals their cosine similarity — a property assumed throughout the
gallery matching stage.

---

### `enroll.py` — Phase 1: Gallery Construction

`enroll.py` builds the embedding gallery that is used as the reference database during
evaluation. The gallery maps each known identity to a single **prototype vector** —
the mean of all its successfully embedded enrollment images, L2-renormalised.

**Function: `build_database(enroll_dir, db_out, summary_out, checkpoint)`**

The function iterates over the enrollment directory, which is expected to contain one
sub-folder per identity. For each identity:

1. All images in the sub-folder are discovered with `list_images()` (jpg, jpeg, png,
   bmp, webp).
2. Each image is loaded as a BGR array and passed through the detector.
3. **Strict single-face policy**: images where the detector finds zero or more than one
   face are skipped. This is intentionally stricter than the evaluation stage, which
   accepts images with multiple faces by taking the largest. The rationale is that
   enrollment images should be unambiguous — a gallery prototype built from crowded or
   ambiguous images will degrade recognition accuracy.
4. If detection succeeds, `get_embedding()` is called. If it returns `None` (landmark
   extraction or alignment failure), the image is also skipped.
5. All valid embeddings for an identity are collected and passed to
   `FaceDatabase.add_identity()`, which computes their mean and L2-renormalises it to
   produce the prototype. This single prototype vector represents the entire identity in
   the gallery.

Two files are written on completion:

- **`mobilefacenet_enroll.npz`** — a NumPy compressed archive containing a `labels`
  object array (identity names) and a `prototypes` float32 array of shape
  `(num_identities, 128)`.
- **`mobilefacenet_enroll_summary.json`** — per-identity counts of total, used, and
  skipped images, useful for diagnosing low enrollment rates.

---

### `evaluate.py` — Phase 2 & 3: Threshold Selection and Final Evaluation

`evaluate.py` runs evaluation in one of two modes controlled by the `--sweep` flag.
In both modes, the detector and embedder are instantiated **once** and reused across
all images — this avoids the overhead of repeated model loading.

#### Record Building — `build_records()`

For every image in the target split, `build_records()` produces a record dict that
captures the full outcome of the pipeline. The function processes known-identity images
and unknown-identity images with the same code path; the `is_unknown_case` flag
controls how `true_label` is assigned.

For each image:

1. The image is loaded and passed to the detector.
2. If no faces are found, a `DETECTION_FAIL` record is appended and processing moves
   to the next image.
3. If multiple faces are found, the **largest by bounding box area** is selected. This
   differs from the enrollment stage's strict single-face policy: during evaluation the
   subject of interest is assumed to be the most prominent face in frame.
4. `get_embedding()` is called on the selected detection. If it returns `None`, an
   `EMBED_FAIL` record is appended.
5. On success, `get_best_match()` computes the cosine similarity between the embedding
   and every prototype in the gallery and returns the identity with the highest score
   along with that score. An `OK` record is appended with `best_label` and
   `best_score`; no threshold decision is made at this stage.
6. Any unexpected exception produces an `ERROR` record, ensuring one image can never
   crash the entire evaluation run.

The separation of record-building from threshold application is deliberate: it allows
the same set of raw similarity scores to be re-evaluated at many thresholds without
re-running the expensive detection and embedding steps.

#### Threshold Application — `apply_threshold()`

`apply_threshold()` converts raw records into predictions. For `OK` records:

- If `best_score >= threshold`, `pred_label = best_label` (identity accepted).
- If `best_score < threshold`, `pred_label = UNKNOWN_LABEL` (identity rejected).

For non-`OK` records, `pred_label` is set to the status string (`DETECTION_FAIL`,
`EMBED_FAIL`, `ERROR`) so they appear distinctly in downstream analysis.

#### Metrics — `compute_metrics()`

Metrics are computed only over records with `status == "OK"` — detection and embedding
failures are reported as counts but excluded from accuracy calculations, since they
represent pipeline failures rather than recognition errors.

Three accuracy metrics are reported:

- **`overall_accuracy`**: fraction of all valid records (known + unknown) where
  `pred_label == true_label`.
- **`known_accuracy`**: fraction of known-identity valid records correctly identified.
  An incorrect prediction here is either a wrong identity (mis-identification) or
  `UNKNOWN_LABEL` (false rejection).
- **`unknown_rejection_rate`**: fraction of unknown-identity valid records that were
  correctly rejected (predicted as `UNKNOWN_LABEL`). Only present when unknown samples
  are provided; `None` otherwise.

The failure counts (`num_detection_fail`, `num_embed_fail`, `num_error`) are included
in the output to allow a complete accounting of all images in the split.

#### Mode 1 — Threshold Sweep (`--sweep`)

Threshold sweep mode is intended for use on the **validation split** to select the
best operating threshold before touching the test set.

Records are built once from both the validation split and the unknown-identity
directory. The same records are then evaluated at 11 thresholds:
`[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]`.

This range is lower than the ArcFace baseline sweep (`[0.30, 0.70]`) because
MobileFaceNet produces 128-dimensional embeddings rather than 512-dimensional ones.
In lower-dimensional spaces the dot product between random unit vectors is closer to
zero on average, so genuine match scores are typically lower than in ArcFace.

At each threshold, a **selection score** determines the best threshold:

- If unknown samples are present: `0.5 × known_accuracy + 0.5 × unknown_rejection_rate`
  — equal weight on identifying known faces and rejecting unknown ones.
- If no unknown samples are present: `known_accuracy` alone.

The threshold with the highest selection score is recorded as `best_threshold`.

Output: `artifacts_mobilefacenet/metrics/<tag>_val_threshold_sweep.json`

```json
{
  "split": "val",
  "db": "...",
  "best_threshold": 0.35,
  "results": [
    {
      "threshold": 0.10,
      "num_records": 240,
      "num_valid": 210,
      "num_detection_fail": 30,
      "num_embed_fail": 0,
      "num_error": 0,
      "overall_accuracy": 0.952,
      "known_accuracy": 1.0,
      "unknown_rejection_rate": 0.0
    },
    ...
  ]
}
```

#### Mode 2 — Single-Threshold Evaluation

Single-threshold mode is used for the **final test evaluation** using the threshold
selected during the sweep.

Records are built from the test split and unknown directory, `apply_threshold()` is
called once at the specified threshold, and metrics are computed.

Two files are written:

- **`<tag>_test_metrics.json`** — summary metrics and the threshold used.
- **`<tag>_test_records.json`** — full per-image records including `true_label`,
  `best_label`, `best_score`, `pred_label`, and `status`. This file is the primary
  artefact for post-hoc error analysis (e.g. confusion matrices, failure case review).

```json
{
  "split": "test",
  "db": "...",
  "threshold": 0.35,
  "metrics": {
    "num_records": 240,
    "num_valid": 215,
    "num_detection_fail": 25,
    "num_embed_fail": 0,
    "num_error": 0,
    "overall_accuracy": 0.958,
    "known_accuracy": 0.963,
    "unknown_rejection_rate": 0.933
  }
}
```

---

## Evaluation Workflow

```
enroll.py                 evaluate.py --sweep        evaluate.py (no sweep)
─────────────────         ──────────────────────     ──────────────────────
For each identity         Build records (val +        Build records (test +
in ENROLL_DIR:            unknown); no threshold      unknown); apply best
  detect → align          applied yet.                threshold from sweep.
  → embed → collect                                   
embeddings.               Evaluate at 11              Compute metrics.
                          thresholds [0.10…0.60].     
Compute mean embedding                                Write metrics.json +
per identity (prototype)  Pick best by:               records.json.
→ L2-normalise.           0.5×known_acc +
                          0.5×unk_rej_rate.
Write .npz gallery +      
summary.json.             Write sweep.json with
                          best_threshold.
```

The validation and test splits must come from non-overlapping sets of images — this
is enforced by the `split_dataset.py` script in `arcface_insightface`, which stratifies
images per identity before copying them into `ENROLL_DIR`, `VAL_DIR`, and `TEST_DIR`.

---

## Setup

```bash
# 1. Install pip dependencies
pip install -r requirements.txt

# 2. Install the mobilefacenet_face package from the local frentzen project
pip install -e <repo_root>/frentzen/frentzen

# 3. (Optional) override the checkpoint path if your layout differs
export MFN_CHECKPOINT=/absolute/path/to/mobilefacenet_best.pth
```

### Windows note for `insightface`

On Windows with Python 3.10, `insightface==0.7.3` is typically downloaded from
PyPI as a source archive rather than a prebuilt wheel. During installation, pip
builds the `insightface.thirdparty.face3d.mesh.cython.mesh_core_cython`
extension, which requires Microsoft Visual C++ 14.0 or newer (`cl.exe`).

If installation fails with:

```text
error: Microsoft Visual C++ 14.0 or greater is required
```

install **Visual Studio Build Tools 2022** with the **Desktop development with C++**
workload, then rerun:

```powershell
pip install insightface==0.7.3
```

This package currently shares the older ArcFace detector stack, so `numpy<2` is
recommended in the same environment.

## Running

All commands are run as modules from the **ISD repo root**:

```bash
# Phase 1 — enrollment
python -m face_detection_ee4228.project_2.models.mobilefacenet_insightface.enroll

# Phase 2 — threshold selection on validation set
python -m face_detection_ee4228.project_2.models.mobilefacenet_insightface.evaluate \
    --split val --sweep

# Phase 3 — final evaluation on test set
python -m face_detection_ee4228.project_2.models.mobilefacenet_insightface.evaluate \
    --split test --threshold <best_threshold_from_phase_2>
```

## CLI Reference

### `enroll.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--enroll-dir` | `arcface_insightface.ENROLL_DIR` | Enrollment images root |
| `--db-out` | `artifacts_mobilefacenet/embeddings/mobilefacenet_enroll.npz` | Gallery output path |
| `--checkpoint` | `MFN_CHECKPOINT` env var or `mobilefacenet_best.pth` | Model weights |

### `evaluate.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--split` | `val` | `val` or `test` |
| `--db` | `mobilefacenet_enroll.npz` | Gallery path |
| `--threshold` | `0.45` | Cosine similarity acceptance threshold |
| `--sweep` | off | Find best threshold over validation set |
| `--unknown-dir` | `arcface_insightface.UNKNOWN_DIR` | Unknown-identity images |
| `--checkpoint` | see enroll | Model weights |
| `--tag` | `mobilefacenet` | Output filename prefix |
