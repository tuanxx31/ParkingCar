import cv2
import math

def draw_rectangle(img, top_left, bottom_right, color=(0, 255, 0), thickness=2, label=None, conf=None):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label is not None:
        text = f"{label} {conf:.2f}" if conf is not None else label
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_center_rectangle(top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def calc_distance(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)