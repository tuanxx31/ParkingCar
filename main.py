

from detect_image import detect_image
from detect_video import detect_video

model_path = r"model\v2\best.pt"

if __name__ =="__main__":
    # detect_image(r"ParkingCar_Data\CarParkProject\carParkImg_2.jpg")
    detect_video(r"test\car_test.mp4",model_path,0.5)