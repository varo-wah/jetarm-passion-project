from ultralytics import YOLO
import cv2
from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXHIBITION_PHASE = PROJECT_ROOT / "exhibition_phase"

sys.path.append(str(EXHIBITION_PHASE))

from coordinatelogic import pixel_to_robot


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "models" / "yolo11n.pt"
    
    if model_path.exists():
        print(f"Loading model from: {model_path}")
        return YOLO(str(model_path))

    print(f"Model not found at: {model_path}")
    print("Use this command from Machine_Learning_Model:")
    print("python3 -c \"from ultralytics import YOLO; YOLO('yolo11s.pt')\"")
    return None

    for model_path in candidate_paths:
        if model_path.exists():
            print(f"Loading model from: {model_path}")
            return YOLO(str(model_path))

    print("No local YOLO model found. Downloading yolo11s.pt from Ultralytics...")
    return YOLO("../models/yolo11s.pt")


def extract_detections(result):
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        robot_x, robot_y = pixel_to_robot(center_x, center_y)

        detections.append({
            "class_id": cls_id,
            "confidence": round(conf, 2),
            "center_x": center_x,
            "center_y": center_y,
            "robot_x": round(robot_x, 2),
            "robot_y": round(robot_y, 2),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        })  

    return detections


def main():
    model = load_model()
    if model is None:
        return
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("Starting YOLO live detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break

        results = model(frame, verbose=False)
        result = results[0]

        detections = extract_detections(result)
        annotated_frame = result.plot()

        for det in detections:
            cx = det["center_x"]
            cy = det["center_y"]

            # Draw center point
            cv2.circle(
                annotated_frame,
                (cx, cy),
                5,
                (0, 255, 255),
                -1
            )

            # Text to show near the detection
            text = f"px=({cx},{cy}) robot=({det['robot_x']},{det['robot_y']})"

            cv2.putText(
                annotated_frame,
                text,
                (det["x1"], max(20, det["y1"] - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
            
        cv2.imshow("YOLO Detection", annotated_frame)

        if detections:
            print(detections)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()