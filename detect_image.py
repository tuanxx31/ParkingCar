import cv2
from ultralytics import YOLO
from conf import colors
from utils import draw_rectangle, get_center_rectangle, calc_distance



def detect_image(img_path, model_path="model/v5/best.pt", threshold=0.25, gate_rect_image=None):
    model = YOLO(model_path)
    names = model.names

    results = model(img_path)
    empty_count = 0
    nearest_slot = None
    min_distance = float("inf")
    gate_center = None
    if gate_rect_image:
        gate_center = get_center_rectangle(*gate_rect_image)

    for result in results:
        boxes = result.boxes
        image = result.orig_img.copy()

        for box in boxes:
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

                draw_rectangle(image, slot_top_left, slot_bottom_right,  color=colors[class_id], label=class_name, conf=conf)

                
        if(gate_rect_image):
            draw_rectangle(image, gate_rect_image[0], gate_rect_image[1],
                           color=(255, 0, 0),
                           label="Gate")
            if gate_center:
                cv2.circle(image, gate_center, 6, (255, 0, 0), -1)

        if nearest_slot:
            slot_top_left, slot_bottom_right, slot_center = nearest_slot
            draw_rectangle(image, slot_top_left, slot_bottom_right,
                           color=(0, 255, 255),
                           label="Nearest")
            cv2.circle(image, slot_center, 6, (0, 255, 255), -1)

    cv2.putText(image, f"Empty : {empty_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Parking Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
