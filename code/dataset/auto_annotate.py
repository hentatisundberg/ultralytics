import sys
from pathlib import Path
import os

# Append paths to sys.path
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/") # Sprattus
#sys.path.append("/home/jonas/Documents/vscode/ultralytics/") # Sprattus
#sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/") # Mac

# Print sys.path for debugging
#print("sys.path:", sys.path)

import pandas as pd
from functions import save_all_frames, cut_vid_simpler, remove_similar_images, annotate_images

# Read arguments
video_dir = sys.argv[1]
video_meta_path = sys.argv[2]
vid_outfold = sys.argv[3]
im_outfold = sys.argv[4]
yaml_outfold = sys.argv[5]
yolo_model = sys.argv[6]

# Read metadata on interesting videos
video_meta = pd.read_csv(video_meta_path, sep=";")

# Run video cutting
for row in video_meta.index:
    results = cut_vid_simpler(video_dir, video_meta.loc[row], vid_outfold, 10)


# Extract frames from all vids
#for file in list(Path(vid_outfold).glob("*.mp4")):
#    save_all_frames(file, im_outfold)

# Remove similar images
#remove = remove_similar_images(im_outfold, 250000)
#[os.remove(file) for file in remove]

# Remove video...


# Annotate images
#results = annotate_images(yolo_model, im_outfold, yaml_outfold)

# Run example (Sprattus/Larus)
# python3 code/dataset/auto_annotate.py "../../../../../../mnt/BSP_NAS2_vol3/Video/Video2024/" "data/eidervids2.csv" "vids/" "images/" "data/annotations_yaml/" "../../../../../../mnt/BSP_NAS2/Software_Models/Eider_model/models/eider_model_medium_v5852.pt"

