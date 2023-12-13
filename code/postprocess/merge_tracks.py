



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
rows = range(1, len(trackstats))
for row in rows:
    dist.append(np.sqrt((trackstats["x"]["first"][row]-trackstats["x"]["last"][row-1])**2 + (trackstats["y"]["first"][row]-trackstats["y"]["last"][row-1])**2))
    elapse.append(trackstats["frame"]["first"][row]-trackstats["frame"]["last"][row-1])    

trackstats["dist"] = dist
trackstats["elapse"] = elapse


# Plot 
fig, ax = plt.subplots()

ax.scatter(trackdat["time2"], trackdat["x"], color = "black")
ax.scatter(trackdat["time2"], trackdat["y"], color = "red")
plt.show()


