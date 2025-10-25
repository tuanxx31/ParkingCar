import cv2
from ultralytics import YOLO

# video_path = "ParkingCar_Data\yolo11-parkinglot-main\parking1.mp4" 
video_path = "ParkingCar_Data\CarParkProject\carPark.mp4"
cap = cv2.VideoCapture(video_path)

model = YOLO(r"model\v2\best.pt")

names = model.names
colors = {
    1: (0, 255, 0),  
    0: (0, 0, 255),  
}

while True:
    ret, frame = cap.read()
    if not ret:
        break  
    frame = cv2.medianBlur(frame, 3)
    results = model(frame, device=0, iou=0.5, verbose=False)
    empty_count = 0
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            class_id = int(box.cls.item())
            class_name = names[class_id]
            conf = float(box.conf.item())
            cx, cy, w, h = map(int, box.xywh[0])  
            
            if class_id == 1:
                empty_count +=1
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_id], 2)

                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
    cv2.putText(frame,f"Empty : {empty_count}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    cv2.imshow("Parking Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
