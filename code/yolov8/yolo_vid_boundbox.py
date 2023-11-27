

from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import cv2


# Load a pretrained YOLO model
model = YOLO('runs/detect/train4/weights/best.pt')


# This script will run predictions on each frame of the video, visualize the results, and display them in a window. The loop can be exited by pressing 'q'.


# Open the video file
video_path = "../vids/Auklab1_ROST3_2023-07-01_1440_1462.mp4"
video_path = "../vids/Auklab1_ROST6_2023-07-01_1530_1800.mp4"
cap = cv2.VideoCapture(video_path)


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Save (optional)
        #cv2.imwrite("../dump/frame%d.jpg" % success, frame) 

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reachedq11212
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

