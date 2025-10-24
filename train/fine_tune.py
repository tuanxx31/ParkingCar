from ultralytics import YOLO

def main():
    # Load lại model đã train trên Dataset 1
    model = YOLO(r"E:\Code\TGMT\ParkingCar\model\v2\best.pt")

    # Fine-tune trên Dataset 2
    model.train(
        data="data.yaml",  # YAML của dataset mới
        epochs=40,
        imgsz=640,
        batch=32,
        device=0,     # GPU RTX 3060
        workers=4,
        cache=True,
        amp=True,
        resume=False   # KHÔNG resume log cũ, mà train lại trên dữ liệu mới
    )

if __name__ == "__main__":
    main()
