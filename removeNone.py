import os

label_dir = r"E:\Code\TGMT\ParkingLot\yolo_dataset\labels\val"

for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue
    path = os.path.join(label_dir, file)
    new_lines = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # loại bỏ dòng có 'None' hoặc không đủ 5 giá trị
            if "None" in parts:
                continue
            if len(parts) == 5:
                try:
                    # check xem có convert được thành float không
                    float(parts[1]); float(parts[2]); float(parts[3]); float(parts[4])
                    int(parts[0])
                    new_lines.append(line.strip())
                except:
                    continue
    # ghi lại file chỉ với các dòng hợp lệ
    with open(path, "w") as f:
        f.write("\n".join(new_lines))
