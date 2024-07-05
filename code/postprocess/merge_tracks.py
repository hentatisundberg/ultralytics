

import pandas as pd
from datetime import datetime 
import numpy as np
import sys
sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/") # Mac
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/") # Sprattus
sys.path.append("/home/jonas/Documents/python/ultralytics-1/code/generic_functions/") # Larus
from functions import merge_tracks, calc_stats2, insert_to_db, df_from_db


# Dates 
dates = pd.date_range(start='6/12/2022', end='7/20/2022')
dates2 = pd.date_range(start='6/12/2023', end='7/23/2023')
dates = pd.concat([pd.Series(dates), pd.Series(dates2)], axis = 0)

for date in dates: 
    date = date.date()
    print(f'processing {date}')    
    orig_data = df_from_db("inference/Inference_raw_nomergeCOMPL.db", 
                           f'Ledge == "TRI3"', 
                           f'strftime("%Y-%m-%d", time2) == "{date}"', 
                           False)
    
    if len(orig_data) > 0: 
        init_class = pd.read_csv("inference/unmerged_fishALL.csv", sep = ";", decimal = ",")
        orig_data = orig_data.merge(init_class, on = "track", how = "left")
        orig_data = orig_data[orig_data["multi"] > 0] # Only those selected in first classification

        v1 = merge_tracks(orig_data, 30, 10, 200, 10)
        v2 = merge_tracks(v1, 30, 10, 200, 10)
        v3 = merge_tracks(v2, 30, 10, 200, 10)
        v4 = merge_tracks(v3, 30, 10, 200, 10)  
        v5 = merge_tracks(v4, 30, 10, 200, 10)  
        v6 = v5[["track", "x", "y", "conf", "time2", "width", "height", "maxdim", "mindim", "xdiff", "ydiff","multi", "ledge", "filename"]]

        stats = calc_stats2(v5, "track")
        insert_to_db(v6, "../../../../../../mnt/BSP_NAS2_work/fish_model/databases/Inference_raw_mergeTRI_compl.db")
        insert_to_db(stats, "../../../../../../mnt/BSP_NAS2_work/fish_model/databases/Inference_stats_mergeTRI3_compl.db")




