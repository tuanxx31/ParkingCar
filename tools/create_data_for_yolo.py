import os
import shutil
from glob import glob
import random

YoloFolder = r"E:\Code\TGMT\ParkingLot\ParkingCar_Data\PKLotYoloData\HasXML"
SubDirs = [
    "UFPR04/Sunny", "UFPR04/Rainy", "UFPR04/Cloudy",
    "UFPR05/Sunny", "UFPR05/Rainy", "UFPR05/Cloudy",
    "PUCPR/Sunny", "PUCPR/Rainy", "PUCPR/Cloudy"
]

output_dir = r"E:\Code\TGMT\ParkingLot\yolo_dataset"
image_train_dir = os.path.join(output_dir, "images", "train")
image_val_dir = os.path.join(output_dir, "images", "val")
label_train_dir = os.path.join(output_dir, "labels", "train")
label_val_dir = os.path.join(output_dir, "labels", "val")

for d in [image_train_dir, image_val_dir, label_train_dir, label_val_dir]:
    os.makedirs(d, exist_ok=True)

percentage_train = 90
data_list = {"train": [], "valid": []}

def copy_pair(img, split):
    base = os.path.splitext(os.path.basename(img))[0]
    label = os.path.splitext(img)[0] + ".txt"

    if split == "train":
        shutil.copy(img, os.path.join(image_train_dir, os.path.basename(img)))
        if os.path.exists(label):
            shutil.copy(label, os.path.join(label_train_dir, base + ".txt"))
    else:
        shutil.copy(img, os.path.join(image_val_dir, os.path.basename(img)))
        if os.path.exists(label):
            shutil.copy(label, os.path.join(label_val_dir, base + ".txt"))

for sub in SubDirs:
    folder_path = os.path.join(YoloFolder, sub)
    images = glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True)
    if not images:
        print(f"[⚠] Không tìm thấy ảnh trong {folder_path}")
        continue

    random.shuffle(images)
    total = len(images)
    train_size = round(total * percentage_train / 100)

    train_images = images[:train_size]
    val_images = images[train_size:]

    for img in train_images:
        copy_pair(img, "train")
    for img in val_images:
        copy_pair(img, "valid")

print("✅ Dữ liệu YOLO đã được chuẩn hóa tại:", output_dir)
