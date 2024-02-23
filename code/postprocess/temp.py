



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


trackdata = df_from_db("inference/Inference_stats_nomerge.db", "track != 'rrr'", "track != 'rrr'", True)
t1 = pd.read_csv("data/tracks_temp.csv")

new = t1.merge(trackdata, on = "track", how = "left")
new.to_csv("data/additional_annotations.csv", sep = ";", decimals = ",")