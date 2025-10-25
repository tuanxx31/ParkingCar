import cv2
from ultralytics import YOLO
from conf import colors
from utils import draw_rectangle


def detect_image(img_path, model_path="model/v5/best.pt", threshold=0.25):
    model = YOLO(model_path)
    names = model.names

    results = model(img_path)
    empty_count = 0

    for result in results:
        boxes = result.boxes
        frame = result.orig_img.copy()

        for box in boxes:
            class_id = int(box.cls.item())
            class_name = names[class_id]
            conf = float(box.conf.item())

            if conf >= threshold and class_id == 1:
                empty_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_rectangle(frame, x1, y1, x2, y2, color=colors[class_id], label=class_name, conf=conf)

    cv2.putText(frame, f"Empty : {empty_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Parking Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
