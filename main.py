import cv2

img_path = "ParkingCar_Data/CarParkProject/carParkImg.png"

img = cv2.imread(img_path)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()