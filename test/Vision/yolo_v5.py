import torch, cv2
model = torch.hub.load('/home/ubuntu/jetarm-passion-project/test/Vision/yolov5', 'custom',
                       path='/home/ubuntu/jetarm-passion-project/test/Vision/yolov5/yolov5n.pt', source='local')
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    res = model(frame)
    cv2.imshow('YOLOv5', res.render()[0])
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()