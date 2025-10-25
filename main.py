import cv2

from detect_image import detect_image
from detect_video import detect_video

model_path = "model/v5/best.pt"
video_path = "test/carPark.mp4"
import cv2


def select_gate(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được video.")
        return None

    roi = cv2.selectROI("Chọn vùng cổng (Enter để xác nhận)", frame)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("Chưa chọn vùng nào.")
        return None

    top_left = (int(x), int(y))
    bottom_right = (int(x + w), int(y + h))

    gate_rect = [top_left, bottom_right]
    print(gate_rect)
    return gate_rect

if __name__ == "__main__":
    # detect_image("test/0.png",model_path,0.4)
    # detect_video("test/testVideo.mp4",model_path,0.2)
    gate_point = select_gate(video_path)
    detect_video("test/carPark.mp4", model_path, 0.2, gate_point)
