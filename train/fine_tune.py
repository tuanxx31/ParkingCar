from ultralytics import YOLO

def main():
    model = YOLO(r"E:\Code\TGMT\ParkingCar\model\v2\best.pt")
    model.train(
        data="data2.yaml",
        epochs=30,
        imgsz=640,
        batch=32,
        device=0,
        workers=4,
        cache=True,
        amp=True,
        resume=True   
    )

if __name__ == "__main__":
    main()
