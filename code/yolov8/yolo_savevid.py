import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import sys

# Select tracker and adujt tracker parameters in their yaml files
tracker = "ultralytics/cfg/trackers/bytetrack.yaml"

tracker_name = tracker.split("/")[-1].split(".")[0]

# Load a pretrained YOLO model
model = YOLO("../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train57.pt")

# Output folder
#save_dir = Path("../../../../../../mnt/BSP_NAS2_work/fish_model/clips_annot")

#vid_dir = Path(filelist.iloc[it]["paths"])
vid_dir = Path("../../../../../../mnt/BSP_NAS2_work/fish_model/clips1")
vids = list(vid_dir.glob("*.mp4"))

for vid in vids: 

    # Pick out relevant video information
    results = model(vid, 
                    stream=True,  
                    save = True,
                    show = False, 
                    save_frames = False)
    for result in results: 
        result.boxes
    
