# from ultralytics import YOLO
# model = YOLO('yolo8n.pt')
# results= model("images/1.png",show=True)

from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model("Images/2.png", show=True)
cv2.waitKey(0)