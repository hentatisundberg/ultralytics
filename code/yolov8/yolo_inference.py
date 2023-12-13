

from ultralytics import YOLO
import os
import sys 
import pathlib
import numpy as np
from PIL import Image
import pandas as pd
import torch
import glob
from pathlib import Path

# Decide where to run
torch.cuda.device_count()

# Load a pretrained YOLO model
#model = YOLO('models/best_train53.pt')
model = YOLO('../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train53.pt')

#files = os.listdir("images/")
#files = [pathlib.Path("images/"+item) for item in files]


# Perform object detection 
parent_folder = pathlib.Path("../../../../../mnt/BSP_NAS2/Video/Video2022/BONDEN3/2022-07-04/").absolute()

vids = list(parent_folder.glob("*.mp4"))

fps = 25

for vid in vids: 

    # Pick out relevant video information
    name = vid.name.split("_")
    time = name[3].split(".")
    ledge = name[1]

    starttime = pd.to_datetime(name[2]+" "+time[0]+":"+time[1]+":"+time[2])
    starttime_u = starttime.timestamp()

    results = model(parent_folder.joinpath(vid), stream = True)

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
    out["filename"] = vid.name

    out.to_csv(f'inference/{vid.name}.csv')





