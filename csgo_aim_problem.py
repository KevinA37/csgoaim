import torch
from mss import mss
import cv2 as cv
import numpy as np
import win32gui
import win32api
import keyboard
import time
import pydirectinput as pdi

# Model without specifying classes
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Confidence threshold
confidence_threshold = 0.4

# Define the bounding box
bounding_box = {'top': 340, 'left': 800, 'width': 350, 'height': 400}

# Obtain the window handle (HWND) for the target window
def find_window(title):
    hwnd = win32gui.FindWindow(None, title)
    return hwnd

target_window_title = "Counter-Strike 2"  # Replace with the actual window title
target_window_handle = find_window(target_window_title)

sct = mss()

while True:
    # Check if the active window matches the target window
    active_window_handle = win32gui.GetForegroundWindow()
    if active_window_handle == target_window_handle:
        sct_img = sct.grab(bounding_box)
        img = np.array(sct_img)

        # Inference
        results = model(img)

        # Get the annotated image with bounding boxes
        annotated_image = results.render()[0]

        # Draw filtered bounding boxes
        for detection in results.pred[0]:
            x1, y1, x2, y2, confidence, class_index = detection
            if class_index == 0 and confidence >= confidence_threshold:
                cv.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green rectangle
                
                # Calculate the center of the detected person
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Calculate screen coordinates
                screen_x = bounding_box['left'] + center_x
                screen_y = bounding_box['top'] + center_y

                # Define the number of steps for smoother movement
                steps = 3
                scale_factor = 1.7

                # Calculate the delta for each step
                delta_x = int((screen_x - win32api.GetCursorPos()[0]) / steps)
                delta_y = int((screen_y - win32api.GetCursorPos()[1]) / steps)

                # Perform smooth mouse movement
                for step in range(steps):
                    pdi.moveTo(win32api.GetCursorPos()[0] + delta_x, win32api.GetCursorPos()[1] + delta_y)
                    time.sleep(0.01)  # Adjust the sleep duration for smoother movement

        # Display the annotated image
        cv.imshow('Screenshot', annotated_image)

    if (cv.waitKey(1) & 0xFF) == ord('q'):
        cv.destroyAllWindows()
        break
