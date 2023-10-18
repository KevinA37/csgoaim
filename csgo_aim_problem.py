import torch
from mss import mss
import cv2 as cv
import numpy as np
import win32gui
import win32api
import time
import pydirectinput as pdi
import keyboard

# Model without specifying classes
model = torch.hub.load("ultralytics/yolov5", "yolov5n")

# Confidence threshold
confidence_threshold = 0.4

# Define the window title of the target application
target_window_title = "Counter-Strike 2"  # Replace with the actual window title

sct = mss()

frame_count = 0
start_time = time.time()

while True:
    # Find the target window by its title
    target_window_handle = win32gui.FindWindow(None, target_window_title)

    if target_window_handle:
        # Get the screen coordinates of the target window
        left, top, right, bottom = win32gui.GetWindowRect(target_window_handle)

        # Define the bounding box using the window coordinates
        bounding_box = {'left': left, 'top': top, 'width': right - left, 'height': bottom - top}

        # Capture the region of the target window
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
                screen_x = left + center_x
                screen_y = top + center_y

                # Define the number of steps for smoother movement
                steps = 2  # Increase the number of steps for smoother movement

                # Calculate the delta for each step
                delta_x = screen_x - win32api.GetCursorPos()[0]
                delta_y = screen_y - win32api.GetCursorPos()[1]

                # Perform smooth mouse movement
                for step in range(steps):
                    if keyboard.is_pressed('1'):
                        new_x = int(win32api.GetCursorPos()[0] + delta_x / steps)
                        new_y = int(win32api.GetCursorPos()[1] + delta_y / steps)
                        pdi.moveTo(new_x, new_y)
                        time.sleep(0.1)  # Adjust the sleep duration for smoother movement

        # Display the annotated image
        cv.imshow('Screenshot', annotated_image)

    if (cv.waitKey(1) & 0xFF) == ord('q'):
        cv.destroyAllWindows()
        break

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = current_time
