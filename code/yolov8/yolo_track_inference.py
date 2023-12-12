from ultralytics import YOLO
import os
import sys 
import pathlib
import numpy as np
from PIL import Image
import pandas as pd
import torch


# Select tracker and adujt tracker parameters in their yaml files
# tracker = "ultralytics/cfg/trackers/botsort_custom.yaml"
tracker = "ultralytics/cfg/trackers/bytetrack_custom.yaml"

tracker_name = tracker.split("/")[-1].split(".")[0]

# Load a pretrained YOLO model
model = YOLO('models/best_train53.pt')

# Perform object detection 
vids = ["Auklab1_FAR3_2022-07-09_04.00.00.mp4"]

# vids = os.listdir("vids/")

fps = 25

for vid in vids: 

    # Pick out relevant video information
    name = vid.split("_")
    time = name[3].split(".")
    ledge = name[1]

    starttime = pd.to_datetime(name[2]+" "+time[0]+":"+time[1]+":"+time[2])
    starttime_u = starttime.timestamp()

    results = model.track(vid, stream=True, tracker=tracker, save=True, show_labels=True, show_conf=True, show_boxes=True, device=1)

    # Process results list
    time = []
    boxes = []
    track_ids = []
    confs = []
    classes = []
    framenum = []
    counter = starttime_u
    counterx = 0

    for r in results:
        if not r.boxes.conf.nelement() == 0 :
            boxes.append(r.boxes.xyxy.tolist())
            if r.boxes.id is not None:
                track_ids.append(r.boxes.id.tolist())
            else:
                track_ids.append([-1 for _ in r.boxes])
            classes.append(r.boxes.cls.tolist())
            confs.append(r.boxes.conf.tolist())
            ndetect = len(r.boxes.conf.tolist())
            time.append([counter] * ndetect)
            framenum.append([counterx] * ndetect)

            # Plot image with boxes
            #im_array = r.plot()  # plot a BGR numpy array of predictions
            #im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            #im.show()  # show image

        counter += 1/fps
        counterx += 1

    # Concatenate outputs
    conf = sum(confs, [])
    classes = sum(classes, [])
    boxesx = sum(boxes, [])
    track_ids = sum(track_ids, [])
    times = sum(time, [])
    framenums = sum(framenum, [])


    # Save as data frames
    out1 = pd.DataFrame(boxesx, columns = ["x", "y", "w", "h"])
    out2 = pd.DataFrame(list(zip(track_ids, classes, conf, times, framenums)), columns = ["track_id","class", "conf", "time", "frame"])

    out = out1.merge(out2, left_index = True, right_index = True)
    out["ledge"] = ledge
    out["filename"] = vid

    out.to_csv(f'inference/{vid}_{tracker_name}.csv')