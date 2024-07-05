

import sys
sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/") # Mac
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/") # Sprattus
sys.path.append("/home/jonas/Documents/python/ultralytics-1/code/generic_functions/") # Larus

import pandas as pd
from pathlib import Path
from functions import associate_points_before, calc_stats2, insert_to_db, modify_input


# This scripts runs assign_points_before on raw output (with tracks for some but not all detections) and saves a db

path = Path("../../../../../../mnt/BSP_NAS2_work/fish_model/inference2")
files = list(path.glob("*.csv"))
for file in files: 
    df = pd.read_csv(file)
    df_track = df[df["track_id"] != -1]
    if len(df_track) > 0: 
        print(file)
        assigned = associate_points_before(df, 1, 200, 20)
        assigned2 = modify_input(assigned)
        stats = calc_stats2(assigned2, "track")
        insert_to_db(assigned2, "inference/Inference_raw_nomergeCOMPL.db")
        insert_to_db(stats, "inference/Inference_stats_nomergeCOMPL.db")
