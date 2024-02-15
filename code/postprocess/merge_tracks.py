

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime 
from functions import euclidean, create_connection, merge_tracks, associate_points_before, associate_points_within, calc_stats, modify_output, plot_tracks, prep_data, insert_to_db, predict_from_classifier, df_from_db, plot_tracks2


# Classify all original tracks with existing model

#dataset = df_from_db("inference/Inference_stats_nomerge.db")
#pred = predict_from_classifier("inference/Inference_stats_nomerge.db") 

orig_data = df_from_db("inference/Inference_raw_nomerge.db", f'Ledge == "FAR3"', f'strftime("%Y-%m-%d", time2) == "2022-06-23"')
init_class = pd.read_csv("inference/Unmerged_nofish.csv", sep = ";")
orig_data = orig_data.merge(init_class, on = "track", how = "left")

#Merge params

# Merge
v1 = merge_tracks(orig_data, 30, 10, 300, .2)
v2 = merge_tracks(v1, 30, 10, 300, .2)
v3 = merge_tracks(v2, 30, 10, 300, .2)
v4 = merge_tracks(v3, 30, 10, 300, .2)
v5 = merge_tracks(v4, 30, 10, 300, .2)

plot_tracks2(v5)



# 1. Classify all original tracks (based on summary database) - remove those few models are considering fish 
# 1.1. Save in database 
# 2. Read 50 tracks at the time (sorted by time and ledge, only valid tracks ) with 10 tracks overlap (with several iterations), do merging (should be possible for a data subset such as one day and ledge)
# Do this day by day for each ledge (as there is a break during night anyway)
# 3. Save in new database
# 4. Calc stats and plot for defined time interval and ledge ()


# Set params for run 


# Run multiple
#multpath = "../../../../../mnt/BSP_NAS2_work/fish_model/inference"
#multpath = "inference/orig"
#run_multiple(multpath)


# Run single 
#file = "inference/orig/Auklab1_FAR3_2022-06-28_16.00.00.csv"
#output8 = run_single(file)
#all_data = pd.read_csv("inference/orig/Auklab1_FAR3_2022-06-16_04.00.00.csv")

# Test
#test = pd.read_csv("../../../../../mnt/BSP_NAS2_work/fish_model/inference_merged/Auklab1_BONDEN3_2022-06-15_04.00.00.csv")
#test2 = modify_output(test)



