import cv2
from utils import *
from darknet import Darknet

# Load YOLO model and other configurations
cfg_file = './cfg/yolov3.cfg'
weight_file = './weights/yolov3.weights'
namesfile = 'data/coco.names'
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
iou_thresh = 0.4
nms_thresh = 0.6

# Open the webcam
vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Perform object detection
    boxes = detect_objects(m, frame, iou_thresh, nms_thresh)

    # Draw bounding boxes and labels on the frame
    frame_with_boxes = plot_boxes(frame, boxes, class_names, plot_labels=True)

    # Display the resulting frame
    cv2.imshow('frame', frame_with_boxes)

    # Check if the 'q' button is pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV window
vid.release()
cv2.destroyAllWindows()
