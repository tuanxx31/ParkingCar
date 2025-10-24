import cv2

cap = cv2.VideoCapture("ParkingCar_Data/CarParkProject/carPark.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

i = 0
for t in range(0, duration, 5):  # mỗi 5 giây
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)  # nhảy đến mốc giây
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"ParkingCar_Data/CarParkProject/carParkImg_{i}.jpg", frame)
        i += 1

cap.release()
