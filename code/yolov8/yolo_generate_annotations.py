


from ultralytics import YOLO
import os
import sys 
import pathlib
import numpy as np
from PIL import Image
import pandas as pd
import torch

# Load a pretrained YOLO model
model = YOLO('runs/XXX/best.pt')

# List of videos for inference 
vids = os.listdir("../vids")
fps = 25

# Run
for vid in vids: 

    results = model(f'../vids/{vid}', stream = True)

    # Process results list
    time = []
    boxes = []
    confs = []
    classes = []
    framenum = []
    counter = starttime_u
    counterx = 0

    for r in results:

        boxes.append(r.boxes.xyxy.tolist()) 
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
    times = sum(time, [])
    framenums = sum(framenum, [])


    # Save as data frames
    out1 = pd.DataFrame(boxesx, columns = ["x", "y", "w", "h"])
    out2 = pd.DataFrame(list(zip(classes, conf, times, framenums)), columns = ["class", "conf", "time", "frame"])

    out = out1.merge(out2, left_index = True, right_index = True)
    out["ledge"] = ledge
    out["filename"] = vid

    out.to_csv(f'inference/{vid}.csv')




