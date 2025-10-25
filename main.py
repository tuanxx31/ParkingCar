

from detect_image import detect_image
from detect_video import detect_video

model_path = "model/v5/best.pt"

if __name__ =="__main__":
    detect_image(r"test/2013-03-13_07_10_01.jpg")
    # detect_video("test/parking1.mp4",model_path,0.5)