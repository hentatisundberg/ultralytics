

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path



# Create initial tracks based on conservative detection threshold
# Assign points around those to existing tracks 
# Merge tracks 



# Find all consecutive detections over a certain threshold 
# Name tho 
# Apply track merge from previous step


conf_thres = .8
track_assign_thres = 1000 
time_space_scale = 10



inputfile = "inference/tracking/botsort_custom2_____20231217T211805/Auklab1_FAR3_2023-07-05_06.00.00.mp4_botsort_custom2.csv"
dat = pd.read_csv(inputfile)
dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)


# Initital track creation

dat = dat[dat["conf"] > conf_thres]

start = 6
rows = range(start, len(dat))
dat["track_temp"] = range(0, len(dat))

track = [0]*start
current_track = 0

for row in rows:

    current = dat.iloc[row]
    previous = dat.iloc[(row-5):(row-1)]
    x, y, frame = previous["x"]-current["x"], previous["y"]-current["y"], previous["frame"]-current["frame"]

    dist0 = np.sqrt(x**2 + y**2)
    elapse0 = abs(frame)*time_space_scale
    score = dist0+elapse0
    minval = min(score)
    nearest = np.argwhere(score == minval)[0][0]

    if minval < track_assign_thres: 
        current_track = track[-1] 
    else: 
        current_track = current["track_temp"]
    track.append(current_track)

dat["nt3"] = track


# Plot most recent track 
palette = sns.color_palette("bright")
sns.set(rc = {'axes.facecolor': 'white'})
ax = sns.scatterplot(x= dat["time2"], y=dat["x"], hue = dat["nt3"].astype("int"), palette = palette)
ax.invert_yaxis()
ax.grid(False)
plt.show()
