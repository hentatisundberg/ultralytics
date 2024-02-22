


import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import sqlite3
from datetime import datetime 
from functions import euclidean, create_connection, associate_points_before, calc_stats2, modify_output, prep_data, insert_to_db, df_from_db, modify_input



def plot_raw_by_time(): 
    df["time2"] = pd.to_datetime(df["time"]*1000*1000*1000)
    assigned["time2"] = pd.to_datetime(assigned["time"]*1000*1000*1000)
    tracks = assigned["track_id"].unique()
    dat_notrack = df[df["track_id"] == -1]
    dat_track = df[df["track_id"] != -1]

    # Plot 
    fig, axs = plt.subplots(figsize=(15, 8))

    for track in tracks: 
        td = assigned[assigned["track_id"] == track]
        col = np.random.rand(3,)
        axs.scatter(td["time2"], td["y"], s=80, c = col)    

    axs.scatter(dat_notrack["time2"], dat_notrack["y"], c = "black", s = 3, marker = ".")
    axs.scatter(dat_track["time2"], dat_track["y"], c = "red", s = 7, marker = "*")    

    axs.grid(False)
    
    plt.show()


# This scripts runs assign_points_before on raw output and saves a db

path = Path("inference/orig/")
files = list(path.glob("*.csv"))
for file in files: 
    df = pd.read_csv(file)
    if len(df) > 0: 
        print(file)
        assigned = associate_points_before(df, 1, 200, 20)
        assigned = modify_input(assigned)
        stats = calc_stats2(assigned, "track_id")
        insert_to_db(assigned, "inference/Inference_raw_nomergeV2.db")
        insert_to_db(stats, "inference/Inference_stats_nomergeV2.db")

