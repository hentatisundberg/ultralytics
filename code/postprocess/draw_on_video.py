
import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import shutil
import cv2
import os
import sqlite3
import numpy as np
import sys
from functions import create_connection, df_from_db, cut_vid, compress_annotate_vid



# Read databases  
df_raw = df_from_db("inference/Inference_raw_nomergeZ.db", f'ledge == "FAR3"', f'strftime("%Y-%m-%d", time2) != "XYZ"', False)
df_stats = df_from_db("inference/Inference_stats_nomergeZ.db", f'nframes > 0', f'strftime("%Y-%m-%d", start) != "XYZ"', True)


# Cut vid
for row in df_stats.index:
    input = df_stats.iloc[row]
    print(f'starting with {input.track}')
    vid = cut_vid(input, "../../../../../../Volumes/JHS-SSD2/full_vid/", "../../../../../../Volumes/JHS-SSD2/cut_vid/", "track") 
    print("cut finished")


# Annotate and compress
allfiles = list(Path("../../../../../../Volumes/JHS-SSD2/cut_vid/").glob("*.mp4"))
for file in allfiles[360:]:
    print(f'processing {file} ...')
    compress_annotate_vid(file, "../../../../../../Volumes/JHS-SSD2/annot_merge/")


