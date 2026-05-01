from pathlib import Path

from ultralytics import YOLO

print("Loading YOLO model...")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolov8n.pt"
model = YOLO(str(MODEL_PATH))

print("Model loaded successfully")
