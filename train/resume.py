from ultralytics import YOLO

def main():
    model = YOLO(r"E:\Code\TGMT\runs\detect\train14\weights\last.pt")
    model.train(
        data="data.yaml",
        epochs=15,       
        resume=False,
        cos_lr=False
    )

if __name__ == "__main__":
    main()
