

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime 
from functions import euclidean, create_connection, merge_tracks, associate_points_before, associate_points_within, calc_stats2, modify_output, prep_data, insert_to_db, predict_from_classifier, df_from_db


# Classify all original tracks with existing model

#dataset = df_from_db("inference/Inference_stats_nomerge.db")
#pred = predict_from_classifier("inference/Inference_stats_nomerge.db") 


dates = pd.date_range(start='6/20/2022', end='7/10/2022')

for date in dates: 
    date = date.date()
    print(f'processing {date}')    
    orig_data = df_from_db("inference/Inference_raw_nomerge.db", f'Ledge == "FAR3"', f'strftime("%Y-%m-%d", time2) == "{date}"', False)
    
    if len(orig_data) > 0: 
        init_class = pd.read_csv("inference/unmerged_nofish.csv", sep = ";", decimal = ",")
        orig_data = orig_data.merge(init_class, on = "track", how = "left")

        # Merge
        #Params: merge_tracks(input_data, size, chunksize, track_merge_thresh, time_scaling):
            
        v1 = merge_tracks(orig_data, 30, 10, 200, 10)
        v2 = merge_tracks(v1, 30, 10, 200, 10)
        v3 = merge_tracks(v2, 30, 10, 200, 10)
        v4 = merge_tracks(v3, 30, 10, 200, 10)  
        v5 = merge_tracks(v4, 30, 10, 200, 10)  
        v6 = v5[["track", "x", "y", "conf", "time2", "width", "height", "maxdim", "mindim", "xdiff", "ydiff","multi", "ledge", "filename"]]

        stats = calc_stats2(v5, "track")
        #plot_tracks2(v5)
        insert_to_db(v6, "inference/Inference_raw_merge.db")
        insert_to_db(stats, "inference/Inference_stats_merge.db")





