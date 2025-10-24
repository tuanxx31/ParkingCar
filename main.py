import cv2
from ultralytics import YOLO

img_path = "ParkingCar_Data/CarParkProject/carParkImg.png"

model = YOLO("best.pt")

results = model(img_path,device="mps",iou=0.5)

names = model.names
colors = {
    1 : (0,255,0),
    0 : (0,0,255),
}

for result in results:
    img = result.orig_img.copy()
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int,box.xyxy[0])
        class_id = box.cls.item()
        class_name = names[class_id]
        if box.conf >0.2 and class_id ==1:
            print("conf ::",box.conf)
            print(f"Class: {class_name}, Bounding Box: ({x1}, {y1}, {x2}, {y2})")
            cv2.rectangle(img,(x1,y1),(x2,y2),colors[class_id],2)
            cv2.putText(img,class_name,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.7,colors[class_id])
            # cv2.putText(img,str(box.conf),(x2,y2),cv2.FONT_HERSHEY_SIMPLEX,1,colors[class_id])

cv2.imshow("show",img)
cv2.waitKey(0)
cv2.destroyAllWindows()