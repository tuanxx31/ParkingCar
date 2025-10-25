

from detect_image import detect_image
from detect_video import detect_video
 
model_path = "model/v5/best.pt"

if __name__ =="__main__":
    # detect_image("test/0.png")
    detect_video("test/car_test.mp4",model_path,0.2)