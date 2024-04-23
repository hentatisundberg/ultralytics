
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import random
import numpy as np
import sys
sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/") # Mac
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/") # Sprattus
sys.path.append("/home/jonas/Documents/python/ultralytics-1/code/generic_functions/") # Larus
from functions import df_from_db
from plot_functions import plot_results, plot_orig_data


# Set path to inference data  
wpath = "inference/"


# Plot 
#fish = plot_results("inference/Inference_stats_merge.db", "inference/merged_fish.csv", "init_move", "conf_mean", True, False, 6)


# Stats
plotdat = df_from_db(wpath + "Inference_stats_mergeFAR3.db", 'ledge != "!"', 'ledge != "Y"', True)
class_res = pd.read_csv(wpath + "merged_fish.csv", sep = ";", decimal = ",")
plotdat = plotdat.merge(class_res, on = "track")
plotdat = plotdat[plotdat["multi"] > 0]

# Save plotdat to Mica
plotdat.to_csv("dump/FAR3StatsMerge_toMica.csv", sep = ";", decimal = ",")


# Raw dat

rawdat = df_from_db(wpath + "Inference_raw_mergeFAR3.db", f'ledge != ""',  'ledge != "XXX"', False)
#rawdat = df_from_db("inference/Inference_raw_merge.db", f'ledge == "{ledge}"',  f'strftime("%Y-%m-%d", time2) == "{date}"', False)
rawdat = rawdat.merge(class_res, on = "track", how = "left")
rawdat = rawdat[rawdat["multi_y"] > 0]
rawdat["date"] = pd.to_datetime(rawdat['time2']).dt.date
rawdat["H"] = rawdat["time2"].dt.strftime("%H").astype("int")
rawdat["Yr"] = rawdat["time2"].dt.year


rawraw = df_from_db(wpath + "Inference_raw_nomergeALL.db", f'ledge == "FAR3"',  'ledge != "XXX"', False)
rawraw["date"] = pd.to_datetime(rawraw['time2']).dt.date
rawraw["H"] = rawraw["time2"].dt.strftime("%H").astype("int")
rawraw["Yr"] = rawraw["time2"].dt.year





# Fix some columns
plotdat["date"] = plotdat["start"].dt.date
plotdat["H"] = plotdat["start"].dt.strftime("%H").astype("int")
plotdat["Yr"] = plotdat["start"].dt.year


# Test selection 
cond1 = plotdat["y_first"] < 250
cond2 = plotdat["init_move"] > 300
yr = 2022
cond3 = plotdat["Yr"] == yr
td = plotdat[cond1 & cond2 & cond3]


# Histogram
fig, ax = plt.subplots()
rangex = list(range(0, 23))
ax.hist(td["H"], color = "darkred", width = 0.9, bins = rangex)
ax.set_xlabel("Hour")
ax.set_ylabel("Number of fish")
fig.suptitle("Time of fish deliveries (H), FAR3, 2022-2023")
#plt.legend()
plt.show()



# Fish size?
fig, ax = plt.subplots()
ax.scatter(plotdat["maxdim_mean"], plotdat["mindim_mean"], c = "r", alpha = .3, s = 4)
ax.set_xlabel("fish length (pixels)")
ax.set_ylabel("fish width (pixels)")
plt.show()



# Trend fish size?
td["size"] = td["maxdim_mean"]*td["mindim_mean"]
fig, ax = plt.subplots()
ax.scatter(td["start"], td["size"], c = "r", alpha = .5, label = "Fish", s = 12)
plt.legend()
plt.show()

# End positions of fish tracks
bg22 = cv2.imread("data/bg2022.jpg")
bg23 = cv2.imread("data/bg2023.jpg")

fig, ax = plt.subplots()
imgplot = ax.imshow(bg23)
yr = 2023
dx = plotdat[plotdat["Yr"] == yr]
ax.scatter(dx["x_last"], dx["y_last"], c = "y", alpha = .5, s = 12)
fig.suptitle(f"End position {yr}")
plt.show()


# Example tracks on FAR3

date = "2023-07-04"
ledge = "FAR3"
all_tracks = rawdat["track"].unique().tolist()
random.shuffle(all_tracks)

fig, ax = plt.subplots()
imgplot = ax.imshow(bg23)

# Plot tracks in space 
for track in all_tracks[0:9]: 
    data = rawdat[rawdat["track"] == track]        
    col = np.random.rand(3,)
    x = data["x"]+(.5*data["width"])
    y = data["y"]+(.5*data["height"])
    ax.plot(x, y, c = col, alpha = .8)
    ax.grid(False)

plt.suptitle(f'{ledge}, {date} (10 random tracks)')
plt.show()


# Tracks over time, one day

fig, ax = plt.subplots()

# Plot tracks in space 
for track in all_tracks: 
    data = rawdat[rawdat["track"] == track]        
    col = np.random.rand(3,)
    x = data["x"]+(.5*data["width"])
    y = -1*(data["y"]+(.5*data["height"]))
    ax.scatter(data["time2"], y, c = col, alpha = .8)
    ax.grid(False)
    #ax.invert_yaxis()

plt.suptitle(f'{ledge}, {date} (all tracks)')
plt.show()


# Initial dist and angle 

fig, ax = plt.subplots()
ax.scatter(td["init_move"], td["y_first"], c = "r", alpha = .5, s = 12)
plt.legend()
plt.show()



fig, ax = plt.subplots()
imgplot = ax.imshow(bg22)
ax.scatter(td["x_last"], td["y_last"], c = "y", s = 15)
fig.suptitle(f"End position {yr}")
plt.show()



td[td["y_last"] < 400]["track"]




all_tracks = td["track"].unique().tolist()

fig, ax = plt.subplots()
imgplot = ax.imshow(bg22)

# Plot tracks in space 
for track in all_tracks: 
    data = rawdat[rawdat["track"] == track]        
    #col = np.random.rand(3,)
    x = data["x"]+(.5*data["width"])
    y = data["y"]+(.5*data["height"])
    ax.plot(x[0:20], y[0:20], c = "yellow", alpha = .8, linewidth = .2)
    ax.grid(False)

plt.suptitle(f'All tracks 2022')
plt.show()


# Plot orig data 

date = "2022-06-27"
limit = 1
cond1 = rawdat["date"] == pd.Timestamp(date)
cond2 = rawdat["multi_y"] > limit
cond3 = plotdat["date"] == pd.Timestamp(date)
cond4 = plotdat["multi"] > limit
cond5 = rawraw["date"] == pd.Timestamp(date)

xx = rawdat.loc[cond1 & cond2]
xs = plotdat.loc[cond3 & cond4]
xr = rawraw[cond5]
all_tracks = xx["track"].unique().tolist()


fig, ax = plt.subplots()
y = -1*(xr["y"]+(.5*xr["height"]))
ax.scatter(xr["time2"], y, s = 1)

# Plot tracks in space 
for track in all_tracks: 
    data = xx[xx["track"] == track]        
    col = np.random.rand(3,)
    x = data["x"]+(.5*data["width"])
    y = data["y"]+(.5*data["height"])
    time = data["time2"]
    ax.plot(time[0:30], -y[0:30], c = "black", alpha = .8, linewidth = 2)
    ax.annotate(data["track"].iloc[0], (time.iloc[0], -y.iloc[0]), fontsize = 8)

plt.suptitle(f'2022-06-27')
plt.show()
















