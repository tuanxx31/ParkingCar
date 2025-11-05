import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
CLASSES = ["occupied", "empty"]

COLORS = {
    "occupied": (0, 0, 255),
    "empty": (0, 255, 0)
}

class YoloLabelViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Label Checker")

        self.img_label = tk.Label(root)
        self.img_label.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Chọn thư mục ảnh", command=self.load_dir).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Ảnh trước", command=self.prev_img).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Ảnh tiếp", command=self.next_img).pack(side=tk.LEFT)

        self.image_files = []
        self.index = 0

    def load_dir(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".jpg",".png"))]
        self.index = 0
        self.show_image()

    def show_image(self):
        if not self.image_files:
            return

        img_path = self.image_files[self.index]
        lbl_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        print(lbl_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls = int(float(parts[0]))
                    coords = list(map(float, parts[1:]))

                    if len(coords) == 4:  
                        x, y, bw, bh = coords
                        x1 = int((x - bw/2) * w)
                        y1 = int((y - bh/2) * h)
                        x2 = int((x + bw/2) * w)
                        y2 = int((y + bh/2) * h)
                        pts = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
                    else:
                        pts = []
                        for i in range(0, len(coords), 2):
                            px = int(coords[i] * w)
                            py = int(coords[i+1] * h)
                            pts.append((px, py))

                    class_name = CLASSES[cls]
                    color = COLORS.get(class_name, (255,255,255))

                    cv2.polylines(img, [np.array(pts, np.int32)], isClosed=True, color=color, thickness=2)
                    cv2.putText(img, class_name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=img)
        self.img_label.image = img
        self.root.title(f"{os.path.basename(img_path)} ({self.index+1}/{len(self.image_files)})")

    def next_img(self):
        if self.image_files and self.index < len(self.image_files)-1:
            self.index += 1
            self.show_image()

    def prev_img(self):
        if self.image_files and self.index > 0:
            self.index -= 1
            self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloLabelViewer(root)
    root.mainloop()
