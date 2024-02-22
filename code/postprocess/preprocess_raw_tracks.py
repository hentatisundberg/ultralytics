


import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import sqlite3
from datetime import datetime 
from functions import euclidean, create_connection, associate_points_before, calc_stats2, modify_output, prep_data, insert_to_db, df_from_db, modify_input
from plot_functions import plot_raw_by_time


# This scripts runs assign_points_before on raw output and saves a db

path = Path("inference/orig/")
files = list(path.glob("*.csv"))
for file in files: 
    df = pd.read_csv(file)
    df_track = df[df["track_id"] != -1]
    if len(df_track) > 0: 
        print(file)
        assigned = associate_points_before(df, 1, 200, 20)
        assigned = modify_input(assigned)
        stats = calc_stats2(assigned, "track")
        insert_to_db(assigned, "inference/Inference_raw_nomergeV2.db")
        insert_to_db(stats, "inference/Inference_stats_nomergeV2.db")

