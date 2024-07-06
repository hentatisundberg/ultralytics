from collections import defaultdict

import cv2
import json
import numpy as np

from ultralytics import YOLO


# Select tracker and adujt tracker parameters in their yaml files
tracker = "ultralytics/cfg/trackers/botsort_custom.yaml"
#tracker = "../../ultralytics/cfg/trackers/bytetrack_custom.yaml"

# Load the YOLOv8 model
model = YOLO('models/best_train48.pt')

# Open the video file
video_path = "Auklab1_FAR3_2022-07-09_04.00.00.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
last_frame, frame = None, None

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    last_frame = frame
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker=tracker)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        if results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            

            # Save tracks history
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

#Saving track history in json
with open('track_history.json','w') as json_file:
    json.dump(dict(track_history), json_file)

# Plot Tracks
for id, tracks in track_history.items():
    c = np.random.choice(range(256), size=3).to_list() # create list with unique track colors
    # Draw the tracking lines
    points = np.hstack(tracks).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(last_frame, [points], isClosed=False, color=tuple(c), thickness=10)

# Save the annotated frame
cv2.imwrite("tracks.jpg", annotated_frame)

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()