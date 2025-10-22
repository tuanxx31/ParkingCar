from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best_renamed.pt")

img_path = "ParkingCar_Data/CarParkProject/carParkImg.png"

# Predict
results = model(img_path)

# img = cv2.imread(img_path)

i = 0
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    class_ids = result.boxes.cls.int().tolist()

    print(class_ids)

    empty_count = class_ids.count(1)      
    occupied_count = class_ids.count(0)   

    print("Empty slots:", empty_count)
    print("Occupied slots:", occupied_count)
    # result.show()  # display to screen
    img = result.orig_img.copy()



cv2.destroyAllWindows()
