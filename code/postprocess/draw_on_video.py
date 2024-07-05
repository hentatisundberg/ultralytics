
import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import shutil
import cv2
import os
import sqlite3
import numpy as np
import sys

sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/")
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/")
sys.path.append("/home/jonas/Documents/python/ultralytics-1/code/generic_functions/") # Larus

from functions import create_connection, df_from_db, cut_vid, compress_annotate_vid_nodetect


# Read databases  
df_raw = df_from_db("../../../../../../mnt/BSP_NAS2_work/fish_model/databases/Inference_raw_mergeTRI3_compl.db", f'ledge == "TRI3"', f'strftime("%Y-%m-%d", time2) != "XYZ"', False)
df_stats = df_from_db("../../../../../../mnt/BSP_NAS2_work/fish_model/databases/Inference_stats_mergeTRI3_compl.db", f'nframes > 0', f'strftime("%Y-%m-%d", start) != "XYZ"', True)


# Cut vid
# Combine with classification
df_class = pd.read_csv("inference/merged_fishTRI3compl.csv", sep = ";", decimal = ",")
df_stats = df_stats.merge(df_class, on = "track", how = "left")
df_stats = df_stats[df_stats["multi"] > 0]

print(len(df_stats))
count = 0
for row in list(range(0, len(df_stats))):
    input = df_stats.iloc[count]
    print(f'starting with {input.track}')
#   vid = cut_vid(input, "../../../../../../Volumes/JHS-SSD2/full_vid/", "../../../../../../Volumes/JHS-SSD2/cut_vid/", "track") 
    vid = cut_vid(input, "../../../../../../mnt/BSP_NAS2/Video/", "../../../../../../mnt/BSP_NAS2_work/fish_model/cut_vids/", 2) 
    print("cut finished")
    count += 1


# Annotate and compress
#allfiles = list(Path("../../../../../../mnt/BSP_NAS2_work/fish_model/cut_vids").glob("*.mp4"))
#for file in allfiles:
#    print(f'processing {file} ...')
#    compress_annotate_vid_nodetect(df_raw, file, "../../../../../../mnt/BSP_NAS2_work/fish_model/cut_vids/")


