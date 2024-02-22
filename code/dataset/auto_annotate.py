import os
import pandas as pd
import sys
import numpy as np
import cv2
import yaml
from pathlib import Path



# Read arguments
video_meta_path = sys.argv[1]
vid_sourcefold = sys.argv[2]
vid_outfold = sys.argv[3]
im_outfold = sys.argv[4]
yaml_outfold = sys.argv[5]
yolo_model = sys.argv[6]


# Read metadata on interesting videos
video_meta = pd.read_csv(video_meta_path)


## RUN

# Run video cutting
#results = cut_vid() 

# Extract frames from all vids 
#for file in os.listdir(vid_outfold):
#    save_all_frames()

results = annotate_images()



# Run example (Sprattus/Larus)
#python3 code/dataset/auto_annotate.py "data/fishvids.csv" "../../../../../../../mnt/BSP_NAS2/Video/" "vids/" "images/" "data/annotations_yaml/" "../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train55.pt"



