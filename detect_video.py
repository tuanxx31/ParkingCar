import cv2
from ultralytics import YOLO
from conf import colors
from utils import  draw_rectangle



def detect_video(video_path, model_path="model/v5/best.pt", threshold=0.25):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    names = model.names

    frame_count = 0
    frame_skip = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # frame = cv2.medianBlur(frame, 3)
        results = model(frame, iou=0.5, verbose=False)

        empty_count = 0
        for result in results:
            boxes = result.boxes

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
