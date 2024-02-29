
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
import sys
import numpy as np
import cv2
from pathlib import Path
from functions import cut_vid, save_frames

# Cut 
cut_vid("data/eider_vids.csv", Path("../../../../../../mnt/BSP_NAS2/Video/"), "vids/")

# Extract  
files = list(Path("vids/").glob("*.mp4"))
for file in files: 
    print(f'Extracting frames from {file}')
    save_frames(file, Path("images/"), 50)

