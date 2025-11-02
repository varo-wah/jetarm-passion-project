import cv2

cap = cv2.VideoCapture(0)  # 0 = /dev/video0
if not cap.isOpened():
    raise Exception("Camera not found")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()