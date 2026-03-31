import os
from pathlib import Path

# ── Package anchor ────────────────────────────────────────────────────────────
# Always safe to anchor to this file's own directory — unaffected by how the
# repo is named or where it is cloned.
_PACKAGE_DIR = Path(__file__).resolve().parent

# ── Artifact output directories ───────────────────────────────────────────────
# Stored inside this package folder so outputs stay self-contained.
MFN_ARTIFACTS_DIR = _PACKAGE_DIR / "artifacts_mobilefacenet"
MFN_EMBEDDINGS_DIR = MFN_ARTIFACTS_DIR / "embeddings"
MFN_METRICS_DIR = MFN_ARTIFACTS_DIR / "metrics"

for _d in [MFN_ARTIFACTS_DIR, MFN_EMBEDDINGS_DIR, MFN_METRICS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── MobileFaceNet checkpoint ──────────────────────────────────────────────────
# Priority 1: environment variable  →  set this if your layout differs from the
#             standard repo structure.
#
#   export MFN_CHECKPOINT=/absolute/path/to/mobilefacenet_best.pth
#
# Priority 2: standard repo layout fallback.
#   _PACKAGE_DIR.parents:
#     [0] = models/
#     [1] = project_2/
#     [2] = face_detection_ee4228/
#     [3] = ISD/  (repo root)
_ckpt_env = os.environ.get("MFN_CHECKPOINT")
if _ckpt_env:
    MFN_DEFAULT_CHECKPOINT = Path(_ckpt_env)
else:
    _ISD_DIR = _PACKAGE_DIR.parents[3]
    MFN_DEFAULT_CHECKPOINT = (
        _ISD_DIR / "frentzen" / "frentzen" / "artifacts" / "checkpoints" / "mobilefacenet_best.pth"
    )
