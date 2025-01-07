import sys
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/") # Sprattus
sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/") # Mac
import pandas as pd
from pathlib import Path
from functions import cut_vid, save_all_frames, cut_vid_simpler

# Read arguments
video_meta_path = sys.argv[1]
vid_outfold = sys.argv[2]
im_outfold = sys.argv[3]
yaml_outfold = sys.argv[4]
yolo_model = sys.argv[5]

# Read metadata on interesting videos
video_meta = pd.read_csv(video_meta_path, sep = ";")

# Run video cutting
for row in video_meta.index:
    results = cut_vid_simpler(video_meta.loc[row], vid_outfold, 0) 

# Extract frames from all vids 
for file in list(Path(vid_outfold).glob("*.mp4")):
    save_all_frames(file, im_outfold)

# Remove similar images
#remove_similar_images(im_outfold, 10000)

# Annotate
#results = annotate_images()

# Run example (Sprattus/Larus)
#python3 code/dataset/auto_annotate.py "data/fishvids.csv" "../../../../../../../mnt/BSP_NAS2/Video/" "vids/" "images/" "data/annotations_yaml/" "../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train55.pt"
#python3 code/dataset/auto_annotate.py "data/eidervids.csv" "../../../../Downloads/vids/" "../../../../Downloads/ims/" "data/annotations_yaml/" "../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train55.pt"

