from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Shared raw data outside project_2
RAW_DATA_DIR = Path("/Users/jessica/Documents/GitHub/face_detection_ee4228/augmented_data/training_images_augmented")  # CHANGE THIS

LOCAL_DATA_DIR = BASE_DIR / "data_augmented"
SPLIT_DIR = LOCAL_DATA_DIR / "split"
ENROLL_DIR = SPLIT_DIR / "enroll"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"
UNKNOWN_DIR = LOCAL_DATA_DIR / "unknown"

ARTIFACTS_DIR = BASE_DIR / "artifacts_augmented"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
LOGS_DIR = ARTIFACTS_DIR / "logs"

RANDOM_SEED = 42

ENROLL_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

UNKNOWN_LABEL = "Unknown"
DEFAULT_THRESHOLD = 0.45

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
CTX_ID = -1  # use -1 if CPU only

for d in [
    LOCAL_DATA_DIR,
    SPLIT_DIR,
    ENROLL_DIR,
    VAL_DIR,
    TEST_DIR,
    UNKNOWN_DIR,
    ARTIFACTS_DIR,
    EMBEDDINGS_DIR,
    METRICS_DIR,
    LOGS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
