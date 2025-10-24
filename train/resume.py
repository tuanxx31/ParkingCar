from ultralytics import YOLO

def main():
    model = YOLO("last.pt")
    model.train(
        data="data.yaml",
        epochs=20,       # tổng số epoch muốn (10 cũ + 10 mới)
        resume=True      # quan trọng!
    )

if __name__ == "__main__":
    main()
