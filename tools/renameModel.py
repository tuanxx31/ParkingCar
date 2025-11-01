from ultralytics import YOLO

model = YOLO("best.pt")

model.model.names = {0: "occupied", 1: "empty"}

model.save("best_renamed.pt")
