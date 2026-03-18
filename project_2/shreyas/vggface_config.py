from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SHREYAS_ROOT = Path(__file__).resolve().parent

# Dataset root (folder contains one subfolder per person/class)
DATASET_DIR = REPO_ROOT / "training_images_augmented"

# Artifact outputs for this pipeline only
ARTIFACT_DIR = SHREYAS_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "vggface_friends.keras"
CLASS_INDEX_PATH = ARTIFACT_DIR / "class_indices.json"
DATASET_SUMMARY_PATH = ARTIFACT_DIR / "dataset_summary.json"
HISTORY_CSV_PATH = ARTIFACT_DIR / "training_history.csv"

# Model / data settings
INPUT_SIZE = (224, 224)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

# Training settings
LEARNING_RATE = 1e-4
EPOCHS = 25
EARLY_STOPPING_PATIENCE = 6

# Realtime inference settings
CONFIDENCE_THRESHOLD = 0.60
DETECTION_SCALE = 0.50
PROCESS_EVERY_N_FRAMES = 2
