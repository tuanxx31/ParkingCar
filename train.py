from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")
    model.train(
        data="pklot.yaml",
        epochs=10,
        imgsz=640,
        batch=32,
        device=0,     # GPU RTX 3060
        workers=4,    # fix thêm để tránh lỗi spawn
        cache=True,
        amp=True
    )

if __name__ == "__main__":
    main()