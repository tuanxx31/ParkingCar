from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

img_path = "ParkingCar_Data/CarParkProject/carParkImg_5.jpg"

results = model(img_path)

names = model.names
for res in results:
    img = res.orig_img.copy()

    for box in res.boxes:
        # Lấy toạ độ
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        label = f"{names[cls_id]} {conf:.2f}"

        # Chọn màu theo class
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # xanh cho empty, đỏ cho occupied

        # Vẽ bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    # Hiển thị ảnh
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()