"""
Train the active LEGO YOLO model using the processed Roboflow export.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO  # noqa: E402

from jetarm.config.yolo_config import BATCH_SIZE, EPOCHS, IMAGE_SIZE, LEGO_DATASET_YAML_PATH  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train the JetArm LEGO YOLO model.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument("--data", type=Path, default=LEGO_DATASET_YAML_PATH, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "detect")
    parser.add_argument("--name", default="lego_yolo11n")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
    )


if __name__ == "__main__":
    main()
