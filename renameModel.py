from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Đổi tên class ngay trong model
model.model.names = {0: "occupied", 1: "empty"}

# Lưu lại thành file mới
model.save("best_renamed.pt")
