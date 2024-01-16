


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

# Read tracks from bytetrack, botsort or similar
# First merge similar tracks based on threshold
# Then possibly assign untracked detections to existing tracks 
# Saving new csv in the end  


# FIX
# Distance scaling
# All combinations or tracks to start with
# How tracks are actually merged
# How this is iterated over until no more tracks are merged


dat = pd.read_csv("inference/orig/Auklab1_FAR3_2022-06-19_17.00.00.csv")
dat[dat["track_id"] != -1]


start = 4599
end = 4588
size = 6
d1 = dat[dat["track_id"] == start][["x", "y", "frame"]]
d2 = dat[dat["track_id"] == end][["x", "y", "frame"]]
if len(d1) < size: 
    ss1 = len(d1)
else: 
    ss1 = size

if len(d2) < size: 
    ss2 = len(d2)
else: 
    ss2 = size

d1s = d1.sample(ss1)
d2s = d2.sample(ss2)

d1l = d1s.values.tolist()
d2l = d2s.values.tolist()


# All combinations

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

dist = []
for i in d1l:
    foo = [euclidean(i, j) for j in d2l]
    dist.append(foo)

dist2 = []
for xs in dist:
    for x in xs:
        dist2.append(x)

min(dist2)





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

