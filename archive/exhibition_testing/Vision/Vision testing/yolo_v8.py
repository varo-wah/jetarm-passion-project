from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
assert cap.isOpened()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model(frame, imgsz=640, device=0, half=True, verbose=False)[0]
    # draw boxes
    frame = r.plot()
    # pick the largest detection
    if len(r.boxes):
        areas = (r.boxes.xyxy[:,2]-r.boxes.xyxy[:,0])*(r.boxes.xyxy[:,3]-r.boxes.xyxy[:,1])
        i = int(areas.argmax().item())
        x1,y1,x2,y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(frame,(cx,cy),6,(0,255,0),2)  # visualize target
        # TODO: map (cx, cy) -> arm coordinates and move

    cv2.imshow("YOLO-Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()