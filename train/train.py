from ultralytics import YOLO

def main():
    model = YOLO("yolo11l.pt")
    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=640,
        batch=20,
        device=0,     
        workers=4,    
        cache=True,
        amp=True
    )

if __name__ == "__main__":
    main()