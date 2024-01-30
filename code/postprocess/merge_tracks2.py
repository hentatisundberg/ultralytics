

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
from itertools import combinations
from itertools import product
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3


# Read tracks from bytetrack, botsort or similar
# First merge similar tracks based on threshold
# Then possibly assign untracked detections to existing tracks 
# Saving new csv in the end  


# FIX
# How tracks are actually merged
# How this is iterated over until no more tracks are merged
# pairw = pd.DataFrame(list(product(t1, t2)), columns = ["t1", "t2"])


# Functions 
def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


def merge_tracks(input_data):

    dat = input_data
    dat = dat[dat["track_id"] != -1]
    ids = dat["track_id"].unique()
    outdata = pd.DataFrame()
    
    if len(ids) > chunksize:
        multiple = 1
    else: 
        multiple = 0

    if multiple == 1:
        n_it = int(np.ceil(len(ids)/chunksize))
        res = []
        for i in list(range(0, n_it)):
            for ele in range(chunksize):
                res.append(i)
        res = res[0:len(ids)] # How to split dataset
        df = pd.DataFrame(list(ids), columns = ["ids"])
        df["res"] = res
    else:
        n_it = 1
        df = pd.DataFrame(list(ids), columns = ["ids"])
        df["res"] = 0
    
    for j in range(0, n_it): 

        iterate = 1 # Initiate loop
        current = df[df["res"] == j]["ids"]
        dx = dat[dat["track_id"].isin(current)]

        while iterate == 1: 

            ids = dx["track_id"].unique()
            
            if len(ids) > 1:
                comblist = pd.DataFrame(combinations(ids, 2))
                ntracks = len(ids)
                ncombs = len(comblist)
                #print(f'Chunk {j} of {n_it}')
                #print(f'Number of tracks = {ntracks}')
                distance = []

                for i in range(0, ncombs):

                    d1 = dx[dx["track_id"] == comblist.iloc[i][0]][["x", "y", "frame"]]
                    d2 = dx[dx["track_id"] == comblist.iloc[i][1]][["x", "y", "frame"]]
                    
                    if len(d1) < size: 
                        ss1 = len(d1)
                    else: 
                        ss1 = size

                    if len(d2) < size: 
                        ss2 = len(d2)
                    else: 
                        ss2 = size

                    d1first = d1.iloc[0:1]
                    d1last = d1.iloc[-1:]
                    
                    d2first = d2.iloc[0:1]
                    d2last = d2.iloc[-1:]

                    d1sample = d1.sample(ss1)
                    d2sample = d2.sample(ss2)

                    d1s = pd.concat([d1first, d1sample, d1last])
                    d2s = pd.concat([d2first, d2sample, d2last])

                    d1s["frame"] = d1s["frame"]*time_scaling
                    d2s["frame"] = d2s["frame"]*time_scaling

                    d1l = d1s.values.tolist()
                    d2l = d2s.values.tolist()

                    # All combinations

                    dist = []
                    for k in d1l:
                        foo = [euclidean(k, j) for j in d2l]
                        dist.append(foo)

                    dist2 = []
                    for xs in dist:
                        for x in xs:
                            dist2.append(x)

                    distance.append(min(dist2))

                comblist["distance"] = distance
                nearest = comblist[comblist["distance"] == min(comblist["distance"])][0:1]

                if nearest["distance"].item() < track_merge_thresh:
                    oldtrack = nearest[1].item()
                    newtrack = nearest[0].item()
                    dx.loc[dx["track_id"] == oldtrack, "track_id"] = newtrack
                    #print(f'Track {oldtrack} merged with track {newtrack} inside chunk {j}')
                else:
                    #print("No more tracks to merge")
                    iterate = 0
            else: 
                #print("All tracks merged...")
                iterate = 0
        outdata = pd.concat([outdata, dx])
    return outdata


def associate_points(track_data, all_data):
    tracks = track_data["track_id"].unique().astype("int")
    
    unassoc = all_data
    unassoc = unassoc[unassoc["track_id"] == -1]
    outdata = pd.DataFrame()
    
    for track in tracks: 

        track_temp = track_data[track_data["track_id"] == track]
        minf, maxf = np.min(track_temp["frame"]), np.max(track_temp["frame"])
        candidates = unassoc.loc[(unassoc["frame"] > minf-framedist) & (unassoc["frame"] < maxf+framedist)]

        iterate = 1 # Initiate loop       
        while iterate == 1: 
    
            d1 = track_temp[["x", "y", "frame"]]
            d2 = candidates[["x", "y", "frame"]]

            if len(d1) < size: 
                ss1 = len(d1)
            else: 
                ss1 = size

            d1first = d1.iloc[0:1]
            d1last = d1.iloc[-1:]
            d1sample = d1.sample(ss1)
            d1s = pd.concat([d1first, d1sample, d1last])

            d1s["frame"] = d1s["frame"]*time_scaling_assign
            d2["frame"] = d2["frame"]*time_scaling_assign

            # Min distance per point to track
            dist = []

            # Loop through each candidate point, recover its min distance 
            points = range(0, len(d2))
            for point in points: 
                p = np.array(d2.iloc[point].tolist())
                d = np.linalg.norm(p - np.array(d1s.values.tolist()), axis=1)
                dist.append(np.min(d))
                
            #print(track)
            #print(dist)
            nearest = np.min(dist)

            if nearest < track_assign_thresh:
                minpos = candidates.loc[dist == nearest]
                minpos["track_id"] = track
                track_temp = pd.concat([track_temp, minpos]) # Update track data
                candidates.drop(minpos.index, inplace = True) # Delete from candidates
                nrow = len(track_temp) 
                if len(candidates) == 0:
                    iterate = 0
                    outdata = pd.concat([outdata, track_temp])
                #print(f'Track {track} now includes {nrow} points')
            else:
                #print("No more tracks to merge")
                iterate = 0
                outdata = pd.concat([outdata, track_temp])
    return outdata


def calc_stats(input_data, orig_file): 
    name = orig_file
    dat = input_data
    dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)
    dat["width"] = abs(dat["x"]-dat["w"])
    dat["height"] = abs(dat["y"]-dat["h"])
    maxdim = []
    mindim = []
    for i in range(0, len(dat)):
        maxdim.append(max(dat.iloc[i]["width"], dat.iloc[i]["height"]))
        mindim.append(min(dat.iloc[i]["width"], dat.iloc[i]["height"]))
    dat["maxdim"] = maxdim
    dat["mindim"] = mindim

    dat["xdiff"] = dat["x"].diff().abs()
    dat["ydiff"] = dat["y"].diff().abs()

    stats = dat.groupby(["track_id"]).agg({"time2": ["min", "max"], 
                                            "frame": "count", 
                                            "x": ["first", "std", "last"],
                                            "y": ["first", "std", "last"], 
                                            "conf": ["min", "mean", "max"], 
                                            "mindim": ["mean", "std"], 
                                            "maxdim": ["mean", "std"],
                                            "xdiff": "sum",
                                            "ydiff": "sum"})
    
    stats = stats.droplevel(1, axis = 1).reset_index()
    
    cols = ["track", 'start','end', 'nframes', "x_first", "x_std", "x_last", "y_first", "y_std", "y_last", "conf_min", "conf_mean", "conf_max", "mindim_mean", "mindim_std", "maxdim_mean", "maxdim_std", "x_dist", "y_dist"]
    stats.columns = cols
    
    timeelapse = stats["end"]-stats["start"]
    dur = []
    for i in timeelapse:
        dur.append(i.total_seconds())    
    stats["dur_s"] = dur

    xx = name.stem

    stats["file"] = xx
    
    ledge = xx.split("_")[1]
    dates = xx.split("_")[2].split("-")
    dateval = dates[0][2:4]+dates[1]+dates[2]
    time = xx.split("_")[3][0:2]
    a = pd.Series(stats.index).astype("int").astype("str")
    stats["file_id"] = ledge+dateval+time
    b = stats["file_id"].reset_index()["file_id"]
    stats["track_id"] = b+"-"+a

    stats["Ledge"] = ledge    
    stats["detect_dens"] = stats["nframes"]/stats["dur_s"]

    printstats = stats[["track_id", "start", "end", "nframes", "conf_mean", "x_dist", "y_dist", "dur_s"]]
    print(printstats.sort_values(by = ["conf_mean"], ascending = False))
    return(stats)


def plot_tracks(track_data, all_data):
    all_data = pd.read_csv(all_data)
    dat = track_data
    dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)
    all_data["time2"] = pd.to_datetime(all_data["time"]*1000*1000*1000)

    # General plotting features
    palette = sns.color_palette("bright")
    sns.set(rc = {'axes.facecolor': 'white'})
    
    # Plot new tracks in space 
    #ax = sns.scatterplot(x= dat["x"], y=dat["y"], hue = dat["track_id"].astype("int"), palette = palette)
    #ax.invert_yaxis()
    #ax.grid(False)
    #plt.show()
    #plt.savefig("temp/"+"tracks_space_"+file_name+"orig.jpg")
    #plt.close()

    # Plot tracks over time 
    ax = sns.scatterplot(x= dat["time2"], y=dat["x"], hue = dat["track_id"].astype("int"), palette = palette)
    ax = sns.lineplot(x= dat["time2"], y=dat["x"], hue = dat["track_id"].astype("int"), palette = palette)
    ax = sns.scatterplot(x = all_data["time2"], y = all_data["x"], size = .1, color = "black", marker = "+")
    ax.invert_yaxis()
    ax.grid(False)
    plt.show()
    #plt.savefig("temp/"+"tracks_time_"+file_name+".jpg")
    #plt.close()

def prep_data(input):
    if input.is_file:
        output = pd.read_csv(input)
    else: 
        output = input
    return output 

def insert_to_db(file):
    file = file 
    #file["file"] = file
    file = file.reset_index()
    file = file[file["nframes"] > 10]
    con_local = create_connection("inference/Inference.db")
    file.to_sql("Inference", con_local, if_exists='append')

def run_multiple(dir):
    dir = Path(dir)
    allfiles = list(dir.glob("*"))
    nfiles = len(allfiles)
    counter = 0
    for file in allfiles:
        orig_file = file
        #file_name = file.stem
        #print(file_name)
        precheck = prep_data(orig_file)
        if len(precheck["track_id"].unique()) > 1:
            output1 = merge_tracks(prep_data(orig_file))
            output2 = merge_tracks(output1)
            output3 = merge_tracks(output2)
            output4 = merge_tracks(output3)
            output5 = merge_tracks(output4)
            output6 = associate_points(output5, prep_data(orig_file))
            output7 = merge_tracks(output6)
            ss = calc_stats(output7, orig_file)
            insert_to_db(ss)
            print(f'Finished with file {counter} of {nfiles}')
        counter += 1

def run_single(file):
    orig_file = Path(file)
    print(orig_file)
    precheck = prep_data(orig_file)
    if len(precheck["track_id"].unique()) > 1:
        output1 = merge_tracks(prep_data(orig_file))
        output2 = merge_tracks(output1)
        output3 = merge_tracks(output2)
        output4 = merge_tracks(output3)
        output5 = merge_tracks(output4)
        output6 = associate_points(output5, prep_data(orig_file))
        output7 = merge_tracks(output6)
        ss = calc_stats(output7, orig_file)
        plot_tracks(output7, orig_file)
        return(ss, output7)

# Set params
size = 30
track_merge_thresh = 300
track_assign_thresh = 100
time_scaling = .1
time_scaling_assign = 1
chunksize = 10
framedist = 200

# Run multiple
multpath = "../../../../../mnt/BSP_NAS2_work/fish_model/inference"
run_multiple("inference/orig")

# Run one 
#run_single("inference/orig/Auklab1_FAR3_2022-06-27_22.00.00.csv")

