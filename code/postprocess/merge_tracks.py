



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dat = pd.read_csv("inference/Auklab1_FAR3_2022-07-08_05.00.00_560_580.mp4_bytetrack_custom.csv")
dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)

# Stats for each track 
trackdat = dat[dat["track_id"] != -1]

trackstats = trackdat.groupby(["track_id"]).aggregate({
    "frame": ["first", "last"], 
    "x": ["first", "last"],
    "y": ["first", "last"],
    }).reset_index()

dist = [0]
elapse = [0]
comb = [0]
metric = [0]
rows = range(1, len(trackstats))
for row in rows:
    x1 = trackstats["x"]["first"][row]-trackstats["x"]["last"][row-1]
    x2 = trackstats["x"]["last"][row]-trackstats["x"]["first"][row-1]
    y1 = trackstats["y"]["first"][row]-trackstats["y"]["last"][row-1]
    y2 = trackstats["y"]["last"][row]-trackstats["y"]["first"][row-1]
    z1 = trackstats["frame"]["first"][row]-trackstats["frame"]["last"][row-1]
    z2 = trackstats["frame"]["last"][row]-trackstats["frame"]["first"][row-1]
    dist0 = np.sqrt(x1**2 + y1**2)
    dist1 = np.sqrt(x2**2 +y2**2)
    mindist = .1*(min(dist0, dist1))
    minelapse = min(abs(z1), abs(z2))
    metric.append(mindist*minelapse)



trackstats["metric"] = metric
trackstats["merge"] = np.where(trackstats["metric"] < 100, True, False)

# Merge 

rows = range(1, len(trackstats))

newtrack = [trackstats["track_id"][0]]
for row in rows:
    if trackstats["merge"][row] == True: 
        newtrack.append(newtrack[row-1])
    else: 
        newtrack.append(trackstats["track_id"][row])
    
trackstats["newtrack"] = newtrack




# Plot 
fig, ax = plt.subplots()

ax.scatter(trackdat["time2"], trackdat["x"], color = "black")
ax.scatter(trackdat["time2"], trackdat["y"], color = "red")
plt.show()


