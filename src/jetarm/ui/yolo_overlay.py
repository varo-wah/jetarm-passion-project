import cv2

from jetarm.vision.yolo_detector import choose_target, detect_objects


def _draw_detection(frame, detection, is_target):
    x1 = detection["x1"]
    y1 = detection["y1"]
    x2 = detection["x2"]
    y2 = detection["y2"]
    center_x = detection["center_x"]
    center_y = detection["center_y"]
    robot_x = detection["robot_x"]
    robot_y = detection["robot_y"]
    angle = detection["angle"]
    confidence = detection["confidence"]
    color = detection.get("color", "NEUTRAL")

    box_color = (0, 0, 255) if is_target else (80, 220, 120)
    text_color = (0, 0, 255) if is_target else (0, 240, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)

    if is_target:
        cv2.circle(frame, (center_x, center_y), 14, (0, 0, 255), 2)

    labels = [
        f"{color} conf={confidence:.2f}",
        f"robot=({robot_x:.2f},{robot_y:.2f})",
        f"angle={angle:.1f}",
    ]
    text_x = x1
    text_y = max(22, y1 - 48)

    for index, label in enumerate(labels):
        cv2.putText(
            frame,
            label,
            (text_x, text_y + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            2,
        )


def annotate_yolo_frame(frame):
    if frame is None:
        return frame

    out = frame.copy()
    detections = detect_objects(frame)
    target = choose_target(detections)

    for detection in detections:
        _draw_detection(out, detection, target is not None and detection == target)

    if target is None:
        status = "YOLO TARGET: none"
    else:
        status = (
            f"YOLO TARGET: x={target['robot_x']:.2f} "
            f"y={target['robot_y']:.2f} "
            f"angle={target['angle']:.1f} "
            f"color={target.get('color', 'NEUTRAL')}"
        )

    cv2.putText(out, status, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return out
