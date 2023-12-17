



import matplotlib.pyplot as plt
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


# Arguments: 
inputfold = "inference/tracking/botsort_custom2_____20231217T205907/"
#inputfile = "inference/Auklab1_FAR3_2022-07-08_05.00.00_560_580.mp4_bytetrack_custom.csv"

files = list(Path(inputfold).glob('*.csv'))

for file in files: 

    filedata = file.name.split("_")
    newname = filedata[1]+"_"+filedata[2]+"_"+filedata[3]

    track_merge_thres = 1000
    track_assign_thres = 500 
    time_space_scale = .1

    dat = pd.read_csv(file)
    dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)

    # Stats for each track 
    trackdat = dat[dat["track_id"] != -1]

    trackstats = trackdat.groupby(["track_id"], as_index = False).aggregate({
        "frame": ["first", "last"], 
        "x": ["first", "last"],
        "y": ["first", "last"],
        })

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


    # Go through df, check if unassigned points are close enough to tracks before or after

    rows = range(0, len(dat))
    newtrack2 = []
    rowprocess = []
    for row in rows:
        if dat["track_id"][row] < 0:
            temp = dat.iloc[row]
            first, last = np.nan, np.nan 
            if len(dat[row:][dat["track_id"] > 0]):
                first = dat[row:][dat["track_id"] > 0].iloc[0]
                x1, y1, z1 = temp["x"]-first["x"], temp["y"]-first["y"], temp["frame"]-first["frame"]
            else: x1, y1, z1 = np.nan, np.nan, np.nan
            if len(dat[:row][dat["track_id"] > 0]):
                last = dat[:row][dat["track_id"] > 0].iloc[0]
                x2, y2, z2 = temp["x"]-last["x"], temp["y"]-last["y"], temp["frame"]-last["frame"]
            else: x2, y2, z2 = np.nan, np.nan, np.nan
            
            dist0 = np.sqrt(x1**2 + y1**2)
            dist1 = np.sqrt(x2**2 +y2**2)
            elapse0 = abs(z1)
            elapse1 = abs(z2)
            scores = [np.nanmean([dist0, elapse0*10]), np.nanmean([dist1, elapse1*10])]
            minval = min(scores)
            mincategory = str(np.where(scores.index(minval) == 0, "first", "last"))
            if minval < track_assign_thres: 
                if mincategory == "first":
                    newtrack2.append(first["newtrack"])
                else: 
                    newtrack2.append(last["newtrack"])
            else: 
                newtrack2.append(temp["track_id"])
        else: 
            newtrack2.append(dat["newtrack"][row])
        rowprocess.append(row)

    dat["newtrack2"] = newtrack2

    dat.to_csv("inference/_-"+newname+".csv")


    # Plot 

    # Plot most recent track 
    palette = sns.color_palette("bright")
    sns.set(rc = {'axes.facecolor': 'white'})
    ax = sns.scatterplot(x= dat["x"], y=dat["y"], hue = dat["newtrack2"].astype("int"), palette = palette)
    ax.invert_yaxis()
    ax.grid(False)
    plt.savefig("temp/"+"tracks_space_"+newname+".jpg")
    plt.close()



    # Plot most recent track 
    palette = sns.color_palette("bright")
    sns.set(rc = {'axes.facecolor': 'white'})
    ax = sns.lineplot(x= dat["time2"], y=dat["y"], hue = dat["track_id"].astype("int"), palette = palette)
    ax.invert_yaxis()
    ax.grid(False)
    plt.savefig("temp/"+"tracks_time_"+newname+".jpg")
    plt.close()



