


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


# This scripts runs assign_points_before on raw output (with tracks for some but not all detections) and saves a db

path = Path("inference/orig/")
files = list(path.glob("*FAR3_2023-07-01_17*"))
for file in files: 
    df = pd.read_csv(file)
    df_track = df[df["track_id"] != -1]
    if len(df_track) > 0: 
        print(file)
        assigned = associate_points_before(df, 1, 200, 20)
        assigned2 = modify_input(assigned)
        stats = calc_stats2(assigned2, "track")
        insert_to_db(assigned2, "inference/Inference_raw_nomergeV3.db")
        insert_to_db(stats, "inference/Inference_stats_nomergeV3.db")

#t1 = assigned[assigned["track_id"] == 78952.0]
#t1["frame"]
#test = t1["frame"]

#newvals = test.diff().tolist()[1:]
#newvals.insert(0, 1.0)
#pnewvals = pd.Series(newvals)
#newtest = pd.Series(newvals) > 0

#df[np.array([0,1,0,0,1,1,0,0,0,1],dtype=bool)]

#t1.loc[:, pnewvals > 0]

#t1.loc[newtest]


#fig, ax = plt.subplots()
#ax.scatter(list(range(0, 26)), t1["frame"])
#plt.show()