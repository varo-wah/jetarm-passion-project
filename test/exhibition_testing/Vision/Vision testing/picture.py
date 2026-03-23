import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

cv2.imwrite("snapshot.jpg", frame)   # optional: save for debugging
