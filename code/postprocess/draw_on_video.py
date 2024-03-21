
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
df_raw = df_from_db("inference/Inference_raw_mergeZ.db", f'ledge == "FAR3"', f'strftime("%Y-%m-%d", time2) != "XYZ"', False)
df_stats = df_from_db("inference/Inference_stats_mergeZ.db", f'nframes > 0', f'strftime("%Y-%m-%d", start) != "XYZ"', True)


# Cut vid
# Combine with classification
df_class = pd.read_csv("inference/unmerged_fish.csv", sep = ";", decimal = ",")
df_stats = df_stats.merge(df_class, on = "track", how = "left")
df_stats = df_stats[df_stats["multi"] > 0]

count = 0
for row in list(range(0, len(df_stats))):
    input = df_stats.iloc[count]
    print(f'starting with {input.track}')
    vid = cut_vid(input, "../../../../../../Volumes/JHS-SSD2/full_vid/", "../../../../../../Volumes/JHS-SSD2/cut_vid/", "track") 
    print("cut finished")
    count += 1


# Annotate and compress
allfiles = list(Path("../../../../../../Volumes/JHS-SSD2/cut_vid/").glob("*.mp4"))
for file in allfiles:
    print(f'processing {file} ...')
    compress_annotate_vid(df_raw, file, "../../../../../../Volumes/JHS-SSD2/annot_merge/")


