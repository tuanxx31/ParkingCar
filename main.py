import cv2
from ultralytics import YOLO

img_path = "ParkingCar_Data/CarParkProject/carParkImg.png"

model = YOLO("best.py")

results = model(img_path)
names = model.names
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        class_id = box.cls.int()
        