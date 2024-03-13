
from functions import df_from_db
import numpy as np
import matplotlib.pyplot as plt


testdat = df_from_db("inference/Inference_stats_nomergeZ.db", "ledge != 'X'", "ledge != 'X'", True)

deg = []

for row in list(range(0, 10000)): 
    xs = testdat.iloc[row][["x_first", "x_nth"]].tolist()
    ys = testdat.iloc[row][["y_first", "y_nth"]].tolist()
    #print(xs)
    # Scale to -1, 1 for the coordinate system of the image 
    xs_adj = [(xs[0]/2592)-.5, (xs[1]/2592)-.5]
    ys_adj = [(ys[0]/1520)-.5, (ys[1]/1520)-.5]
    val = np.degrees(np.arctan2(xs_adj, ys_adj))[1]
    val_out = np.where(val < 0, 360+val, val)
    # Append
    deg.append(val_out)

