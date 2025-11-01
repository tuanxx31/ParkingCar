import os, shutil
import tkinter as tk
from tkinter import filedialog, messagebox

def check_label_format(file_path):
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            values = list(map(float, parts[1:]))
            if len(values) == 4:        
                return "Detection"
            elif len(values) > 4 and len(values) % 2 == 0:
                return "Segmentation"
    return "Unknown"

def move_yolo_segmentation(dataset_dir):
    if not dataset_dir:
        messagebox.showwarning("Cảnh báo", "Bạn chưa chọn thư mục dataset!")
        return
    
    out_dir = os.path.join(os.path.dirname(dataset_dir), "dataset_moved")
    os.makedirs(out_dir, exist_ok=True)

    seg_count = 0
    img_with_seg = 0
    img_no_label = 0
    skipped_det = 0

    for split in ["train", "valid", "test"]:
        img_split = os.path.join(dataset_dir, split, "images")
        lbl_split = os.path.join(dataset_dir, split, "labels")

        if not os.path.exists(img_split):
            continue

        for img_file in os.listdir(img_split):
            if os.path.splitext(img_file)[1].lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            base_name = os.path.splitext(img_file)[0]
            src_img = os.path.join(img_split, img_file)
            src_txt = os.path.join(lbl_split, base_name + ".txt")

            if os.path.exists(src_txt):
                fmt = check_label_format(src_txt)

                if fmt == "Segmentation":
                    dst_txt = os.path.join(out_dir, split, "labels", base_name + ".txt")
                    os.makedirs(os.path.dirname(dst_txt), exist_ok=True)
                    shutil.move(src_txt, dst_txt)
                    seg_count += 1

                    dst_img = os.path.join(out_dir, split, "images", img_file)
                    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                    shutil.move(src_img, dst_img)
                    img_with_seg += 1
                else:
                    skipped_det += 1
            else:
                dst_img = os.path.join(out_dir, split, "images", img_file)
                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                shutil.move(src_img, dst_img)
                img_no_label += 1

    messagebox.showinfo(
        "Hoàn tất",
        f"✅ {seg_count} segmentation labels đã move\n"
        f"✅ {img_with_seg} ảnh có segmentation label đã move\n"
        f"✅ {img_no_label} ảnh không có label đã move\n"
        f"⚠️ {skipped_det} ảnh có detection label (bbox) bị bỏ lại\n\n"
        f"📂 Dataset segmentation ở:\n{out_dir}"
    )

def select_folder():
    folder = filedialog.askdirectory(title="Chọn thư mục gốc dataset (chứa train/valid/test)")
    if folder:
        entry_dir.delete(0, tk.END)
        entry_dir.insert(0, folder)

# ==== GUI ====
root = tk.Tk()
root.title("Move YOLO Segmentation Dataset")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

tk.Label(frame, text="Thư mục gốc YOLO dataset:").pack(anchor="w")

entry_dir = tk.Entry(frame, width=50)
entry_dir.pack(side="left", padx=5)

btn_browse = tk.Button(frame, text="Chọn...", command=select_folder)
btn_browse.pack(side="left")

btn_move = tk.Button(root, text="Move Segmentation", 
                     command=lambda: move_yolo_segmentation(entry_dir.get()), 
                     bg="orange")
btn_move.pack(pady=15)

root.mainloop()
