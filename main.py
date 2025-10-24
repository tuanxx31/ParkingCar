import cv2
from ultralytics import YOLO

img_path = "ParkingCar_Data/CarParkProject/carParkImg.png"

model = YOLO("best.pt")

results = model(img_path)
names = model.names

for result in results:
    img = result.orig_img.copy()
    boxes = result.boxes
    for box in boxes:
        if box.conf >0.5:
            x1, y1, x2, y2 = map(int,box.xyxy[0])
            class_id = box.cls.item()
            class_name = names[class_id]
            print(f"Class: {class_name}, Bounding Box: ({x1}, {y1}, {x2}, {y2})")
            color = (0,255,0)
            cv2.rectangle(img,(x1,y1),(x2,y2),color)
            cv2.putText(img,class_name,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.7,color)

cv2.imshow("show",img)
cv2.waitKey(0)
cv2.destroyAllWindows()