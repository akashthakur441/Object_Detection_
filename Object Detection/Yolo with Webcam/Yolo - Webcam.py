#for webcam
from ultralytics import YOLO
import cv2
import cvzone
import math
from tkinter import *

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def open_camera():
    cap= cv2.VideoCapture(0) #for webcam
    cap.set(3,1280) #widht
    cap.set(4,720) #height
    while True:
        success, img= cap.read()
        results = model(img, stream = True)
        for r in results:
            boxes= r.boxes
            for box in boxes:

                 #bounding box
                x1,y1,x2,y2= box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1),int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

                w,h= x2-x1, y2-y1
                cvzone.cornerRect(img,(x1,y1,w,h)) #by click on cornerReact we can change the color and thickness of the boxes
                #confidence
                conf = math.ceil((box.conf[0]*100))/100

                #class Name
                cls= int(box.cls[0])

                # cvzone.putTextRect(img,f'{classNames[cls]}{conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(35, y1)), scale=3,thickness=3)


        cv2.imshow("Image",img)
        cv2.waitKey(1)


#for videos
def open_video():
    from ultralytics import YOLO
    import cv2
    import cvzone
    import math

    # cap= cv2.VideoCapture(0) #for webcam
    # cap.set(3,1280) #width
    # cap.set(4,720) #height

    cap = cv2.VideoCapture("../Videos/ppe-1-1.mp4")

    # Load the YOLOv8 model weights
    model = YOLO("../Yolo-Weights/yolov8n.pt")

    # List of class names (COCO dataset)
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    while True:
        success, img = cap.read()

        if not success:
            print("Failed to grab frame or video ended.")
            break

        # Run YOLO model on the frame
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes  # Get the bounding boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calculate width and height
                w, h = x2 - x1, y2 - y1

                # Draw rectangle using cvzone (custom corner rectangle)
                cvzone.cornerRect(img, (x1, y1, w, h))  # Optional color and thickness customization

                # Get confidence score
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Get class label index and convert to int
                cls = int(box.cls[0])

                # Display class name and confidence score on the image
                cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        # Display the image with bounding boxes
        cv2.imshow("Image", img)

        # Wait for 1ms before moving to the next frame, break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    root = Tk()
    root.title("Object Detection")
    root.geometry("400x566+500+100")
    try:
        root.attributes('-toolwindow', True)
    except TclError:
        print("hello")

    img_bck = PhotoImage(file="back.png")
    btn_img_ = PhotoImage(file ="vid_btn.png")
    web_img_ = PhotoImage(file="web_btn.png")

    _img = Label(root, image= img_bck).pack()

    video_btn = Button(root, image=btn_img_, border=0,highlightthickness=0,command=open_video)
    video_btn.place(x = 57, y = 280)

    camera_btn = Button(root, image=web_img_, border=0, highlightthickness=0, command=open_camera)
    camera_btn.place(x=250, y=280)

    video_btn = Button(root, text="Video", command=open_video)
    video_btn.pack()
    root.mainloop()
main()