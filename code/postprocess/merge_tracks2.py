


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

# Read tracks from bytetrack, botsort or similar
# First merge similar tracks based on threshold
# Then possibly assign untracked detections to existing tracks 
# Saving new csv in the end  


def calc_dist(file,start,end):

    dat = pd.read_csv(file)
    
    # Stats for each track 
    dat[dat["track_id"] != -1]

    tr = dat.groupby(["track_id"], as_index = False).aggregate({
        "frame": ["first", "last"], 
        "x": ["first", "last"],
        "y": ["first", "last"],
        })

    x1 = tr["x"]["first"][start]-tr["x"]["last"][end]
    x2 = tr["x"]["last"][start]-tr["x"]["first"][end]
    y1 = tr["y"]["first"][start]-tr["y"]["last"][end]
    y2 = tr["y"]["last"][start]-tr["y"]["first"][end]
    
    
        z1 = trackstats["frame"]["first"][row]-trackstats["frame"]["last"][row-1]
        z2 = trackstats["frame"]["last"][row]-trackstats["frame"]["first"][row-1]
        dist0 = np.sqrt(x1**2 + y1**2)
        dist1 = np.sqrt(x2**2 +y2**2)
        mindist = time_space_scale*(min(dist0, dist1))
        minelapse = min(abs(z1), abs(z2))
        metric.append(mindist*minelapse)

    trackstats["metric"] = metric
    trackstats["merge"] = np.where(trackstats["metric"] < track_merge_thres, True, False)

    # Merge 

    rows = range(1, len(trackstats))

    newtrack = [trackstats["track_id"][0]]
    for row in rows:
        if trackstats["merge"][row] == True: 
            newtrack.append(newtrack[row-1])
        else: 
            newtrack.append(trackstats["track_id"][row])
        
    trackstats["newtrack"] = newtrack
    df = pd.DataFrame(trackstats[["track_id", "newtrack"]]).droplevel(1, axis = 1)

    # Combine with original df
    dat = dat.merge(df, on = "track_id", how = "left")

    # Remove unassigned
    dat = dat[dat["track_id"] > 0]

    # Save
    dat.to_csv("inference/merged/"+newname)

