from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
PACKAGE_DIR = SRC_DIR / "jetarm"

DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw" / "images"
RAW_VIDEOS_DIR = DATA_DIR / "raw" / "videos"
PROCESSED_IMAGES_DIR = DATA_DIR / "processed" / "images"
LABELS_DIR = DATA_DIR / "processed" / "labels"
YOLO_DATA_DIR = DATA_DIR / "yolo"
CALIBRATION_DIR = DATA_DIR / "calibration"

MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

