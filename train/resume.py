from ultralytics import YOLO

def main():
    model = YOLO(r"E:\Code\TGMT\runs\detect\train15\weights\last.pt")

    model.train(
        data="data.yaml",
        epochs=20,      
        resume=False,   
        cos_lr=True,
        batch=20,
        device=0,
        workers=4,
        cache=True
    )

if __name__ == "__main__":
    main()
