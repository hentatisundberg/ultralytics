


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import sqlite3
from PIL import Image
from functions import df_from_db
from plot_functions import plot_all_from_db, plot_raw_by_time, plot_track_from_db


# Read background image
bg = np.asarray(Image.open('data/bg.jpg'))


#plot_track_from_db()
plot_raw_by_time()
#plot_all_from_db()

#plot_tracks(dat, "FAR3", "2022-06-27")