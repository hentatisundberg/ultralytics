

import pandas as pd
from datetime import datetime 
import numpy as np
from functions import merge_tracks, calc_stats2, insert_to_db, df_from_db


# Classify all original tracks with existing model

#dataset = df_from_db("inference/Inference_stats_nomerge.db")
#pred = predict_from_classifier("inference/Inference_stats_nomerge.db") 


dates = pd.date_range(start='6/15/2022', end='6/15/2022')

for date in dates: 
    date = date.date()
    print(f'processing {date}')    
    orig_data = df_from_db("inference/Inference_raw_nomergeZ.db", f'Ledge == "FAR3"', f'strftime("%Y-%m-%d", time2) == "{date}"', False)
    
    if len(orig_data) > 0: 
        init_class = pd.read_csv("inference/unmerged_fish.csv", sep = ";", decimal = ",")
        orig_data = orig_data.merge(init_class, on = "track", how = "left")
     
        v1 = merge_tracks(orig_data, 30, 10, 200, 10)
        v2 = merge_tracks(v1, 30, 10, 200, 10)
        v3 = merge_tracks(v2, 30, 10, 200, 10)
        v4 = merge_tracks(v3, 30, 10, 200, 10)  
        v5 = merge_tracks(v4, 30, 10, 200, 10)  
        v6 = v5[["track", "x", "y", "conf", "time2", "width", "height", "maxdim", "mindim", "xdiff", "ydiff","multi", "ledge", "filename"]]

        stats = calc_stats2(v5, "track")
        print(stats)
        insert_to_db(v6, "inference/Inference_raw_mergeZ.db")
        insert_to_db(stats, "inference/Inference_stats_mergeZ.db")





