import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils import *
from darknet import Darknet

# Function to select an image file
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_objects_and_display(file_path)

# Function to detect objects in the selected image and display results
def detect_objects_and_display(image_path):
    img = cv2.imread(image_path)
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(original_image, (m.width, m.height))
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
    print_objects(boxes, class_names)
    plot_img = plot_boxes(original_image, boxes, class_names, plot_labels=True)
    plot_img = Image.fromarray(plot_img)
    plot_img = ImageTk.PhotoImage(plot_img)
    panel.configure(image=plot_img)
    panel.image = plot_img

# Load YOLO model and other configurations
cfg_file = './cfg/yolov3.cfg'
weight_file = './weights/yolov3.weights'
namesfile = 'data/coco.names'
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
iou_thresh = 0.4
nms_thresh = 0.6

# Create a Tkinter window
root = tk.Tk()
root.title("YOLO Object Detection")

# Button to select an image
btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack(padx=10, pady=5)

# Panel to display the image with bounding boxes
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()
