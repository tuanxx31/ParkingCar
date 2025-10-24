import cv2
from ultralytics import YOLO

from main import colors

img_path = r"E:\Code\TGMT\ParkingCar\ParkingCar_Data\PKLotYoloData\HasXML\UFPR05\Sunny\2013-02-22\2013-02-22_06_05_00.jpg"

model = YOLO("best.pt")
names = model.names

results = model(img_path,device=0)

empty_count = 0

for result in results:
    boxes = result.boxes
    frame = result.orig_img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        class_id = int(box.cls.item())
        class_name = names[class_id]
        conf = float(box.conf.item())
        cx, cy, w, h = map(int, box.xywh[0])  
        
        if conf > 0.5 and class_id == 1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_id], 2)

            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
cv2.putText(frame,f"Empty : {empty_count}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
cv2.imshow("Parking Detection", frame)