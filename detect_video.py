import cv2
from ultralytics import YOLO
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

                    

                    slot_center = get_center_rectangle(slot_top_left, slot_bottom_right)

                    if gate_center:
                        dist = calc_distance(slot_center, gate_center)
                        if dist < min_distance:
                            min_distance = dist
                            nearest_slot = (slot_top_left, slot_bottom_right, slot_center)
                        
                    draw_rectangle(frame, slot_top_left, slot_bottom_right,
                                   color=(0, 255, 0),
                                   label=class_name,
                                   conf=conf)

        if gate_rect:
            draw_rectangle(frame, gate_rect[0], gate_rect[1],
                           color=(255, 0, 0),
                           label="Gate")
            if gate_center:
                cv2.circle(frame, gate_center, 6, (255, 0, 0), -1)

        if nearest_slot:
            slot_top_left, slot_bottom_right, slot_center = nearest_slot
            draw_rectangle(frame, slot_top_left, slot_bottom_right,
                           color=(0, 255, 255),
                           label="Nearest")
            cv2.circle(frame, slot_center, 6, (0, 255, 255), -1)

        cv2.putText(frame, f"Empty: {empty_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Parking Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
