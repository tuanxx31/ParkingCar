import cv2
from ultralytics import YOLO
from conf import colors
from utils import draw_rectangle, get_center_rectangle, calc_distance


def detect_video(video_path, model_path="model/v5/best.pt", threshold=0.25, gate_rect=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        return

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

        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        results = model(frame, iou=0.5, verbose=False)

        empty_count = 0
        nearest_slot = None
        min_distance = float("inf")

        gate_center = None
        if gate_rect:
            gate_center = get_center_rectangle(*gate_rect)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = names[class_id]
                conf = float(box.conf.item())

                if conf >= threshold and class_id == 1:
                    empty_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    slot_top_left = (x1, y1)
                    slot_bottom_right = (x2, y2)

                    draw_rectangle(frame, x1, y1, x2, y2, color=colors[class_id], label=class_name, conf=conf)

        cv2.imshow("Parking Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
