import cv2
import math


def draw_rectangle(img, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, label=None, conf=None):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label is not None:
        text = f"{label} {conf:.2f}" if conf is not None else label
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
