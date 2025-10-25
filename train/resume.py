from ultralytics import YOLO

def main():
    model = YOLO(r"E:\Code\TGMT\runs\detect\train15\weights\last.pt")

    model.train(
        data="data.yaml",
        epochs=20,        # tổng số epoch = 30
        resume=False,     # không resume vì run trước đã kết thúc
        cos_lr=True,
        lr0=0.001,        # giảm LR để tinh chỉnh mượt
        batch=20,
        device=0,
        workers=4,
        cache=True
    )

if __name__ == "__main__":
    main()
