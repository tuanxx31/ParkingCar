import cv2
import numpy as np

def draw_polygon(img, points, color=(0, 255, 0), thickness=2, is_closed=True, label=None, conf=None):
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=is_closed, color=color, thickness=thickness)

    if label is not None:
        text = f"{label} {conf:.2f}" if conf is not None else label
        x, y = pts[0][0]  # vẽ text ở đỉnh đầu tiên
        cv2.putText(img, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
