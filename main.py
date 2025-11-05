

from detect_image import detect_image
from detect_video import detect_video
 
model_path = r"model\v5\best.pt"

if __name__ =="__main__":
    detect_image(r"test\test3.png",model_path,0.4)
    # detect_video("test/testVideo.mp4",model_path,0.2)