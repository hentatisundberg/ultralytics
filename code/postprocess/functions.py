

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
from datetime import datetime 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


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



def merge_tracks(input_data, size, chunksize, track_merge_thresh, time_scaling):

    dat = input_data
    dat = dat[dat["multi"] > 0]
    ids = dat["track"].unique()
    ninp = len(ids)
    print(f'number of input tracks = {ninp}')
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
        dx = dat[dat["track"].isin(current)]

        while iterate == 1: 

            ids = dx["track"].unique()
            
            if len(ids) > 1:
                comblist = pd.DataFrame(combinations(ids, 2))
                ntracks = len(ids)
                ncombs = len(comblist)
                #print(f'Chunk {j} of {n_it}')
                #print(f'Number of tracks = {ntracks}')
                distance = []

                for i in range(0, ncombs):

                    d1 = dx[dx["track"] == comblist.iloc[i][0]][["x", "y", "time"]]
                    d2 = dx[dx["track"] == comblist.iloc[i][1]][["x", "y", "time"]]
                    
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

                    d1s["time"] = d1s["time"]*time_scaling
                    d2s["time"] = d2s["time"]*time_scaling

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
                    dx.loc[dx["track"] == oldtrack, "track"] = newtrack
                    #print(f'Track {oldtrack} merged with track {newtrack} inside chunk {j}')
                else:
                    #print("No more tracks to merge")
                    iterate = 0
            else: 
                #print("All tracks merged...")
                iterate = 0
        outdata = pd.concat([outdata, dx])
    nout = len(outdata["track"].unique())
    print(f'number of output tracks = {nout}')
    return outdata




def associate_points_before(detection_data, time_scaling_assign, track_assign_thresh, framedist):
    print("ADDING POINTS BEFORE TRACKS")
    tracks = detection_data["track_id"].unique().astype("int")
    track_data = detection_data[detection_data["track_id"] != -1]
    unassoc = detection_data[detection_data["track_id"] == -1]
    outdata = pd.DataFrame()
    
    for track in tracks: 
        track_temp = track_data[track_data["track_id"] == track]
        minf, maxf = np.min(track_temp["frame"]), np.max(track_temp["frame"])
        
        # Points before 
        cand_bef = unassoc.loc[(unassoc["frame"] < minf) & (unassoc["frame"] > minf-framedist)]

        if len(cand_bef) > 0:
            iterate = 1
            while iterate == 1: 
        
                d1 = track_temp[["x", "y", "frame"]]
                d2 = cand_bef[["x", "y", "frame"]]

                d1s = d1.iloc[0:10]

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
                    
                nearest = np.min(dist)

                if nearest < track_assign_thresh:
                    minpos = cand_bef.loc[dist == nearest]
                    minpos["track_id"] = track
                    track_temp = pd.concat([minpos, track_temp]) # Update track data
                    cand_bef.drop(minpos.index, inplace = True) # Delete from candidates
                    #d1s = pd.concat([d1s, minpos[["x", "y", "frame"]]])
                    track_temp = pd.concat([track_temp, minpos])
                    if len(cand_bef) == 0:
                        iterate = 0
                        #outdata = pd.concat([outdata, track_temp])
                    #print(f'Track {track} now includes {nrow} points')
                else:
                    iterate = 0
        outdata = pd.concat([outdata, track_temp])
    return outdata




def associate_points_within(track_data, all_data):
    print("ADDING POINTS WITHIN TRACKS")
    start = datetime.now()
    tracks = track_data["track_id"].unique().astype("int")
    unassoc = all_data
    unassoc = unassoc[unassoc["track_id"] == -1]
    outdata = pd.DataFrame()
    
    for track in tracks: 

        track_temp = track_data[track_data["track_id"] == track]
        minf, maxf = np.min(track_temp["frame"]), np.max(track_temp["frame"])
        
        # Points within

        cand_within = unassoc.loc[(unassoc["frame"] > minf) & (unassoc["frame"] < maxf+framedist)]

        ids = cand_within.index
        n_it = int(np.ceil(len(ids)/chunksize))
        #print(f'total number of chunks for track {track} = {n_it}')
        res = []
    
        for i in list(range(0, n_it)):
            for ele in range(chunksize):
                res.append(i)

        res = res[0:len(ids)] # How to split dataset
        df = pd.DataFrame(list(ids), columns = ["ids"])  # Index of unassociated point that will be checked
        df["res"] = res
        
        for j in range(0, n_it): 
            
            #print(f'starting chunk = {j}')
            current = df[df["res"] == j]["ids"]
            d2 = cand_within[cand_within.index.isin(current)][["frame", "x", "y"]]
            d1 = track_temp[["frame", "x", "y"]]

            if len(track_temp) < size: 
                ss1 = len(d1)
            else: 
                ss1 = size
            
            d1first = d1.iloc[0:1]
            d1last = d1.iloc[-1:]
            d1sample = d1.sample(ss1)
            d1s = pd.concat([d1first, d1sample, d1last])

            d1s["frame"] = d1s["frame"]*time_scaling_assign
            d2["frame"] = d2["frame"]*time_scaling_assign

            iterate = 1
            while iterate == 1:

                # Loop through each candidate point, recover its min distance 

                dist = []
                counter = 0                
                for point in d2.index: 
                    p = np.array(d2.iloc[counter].tolist())
                    d = np.linalg.norm(p - np.array(d1s.values.tolist()), axis=1)
                    dist.append(np.min(d))
                    counter += 1

                nearest = np.min(dist)
                if nearest < track_assign_thresh:
                    minpos = d2.loc[dist == nearest]
                    minposx = cand_within[cand_within.index == minpos.index.to_list()[0]]
                    minposx["track_id"] = track
                    track_temp = pd.concat([track_temp, minposx]) # Update track data
                    cand_within.drop(minposx.index, inplace = True) # Delete from candidates
                    d2.drop(minpos.index, inplace = True) # Delete from candidates
                    if len(d2) == 0:
                        iterate = 0
                else:
                    iterate = 0
        outdata = pd.concat([outdata, track_temp])
    nrows = len(outdata)
    nrows0 = len(track_data)
    end = datetime.now()
    elapsed = end-start
    print(f'input had {nrows0} rows')
    print(f'output has {nrows} rows')
    print(f'time elapsed: {elapsed}')
    return outdata


def calc_stats2(input_data, trackname): 
    dat = input_data
    dat["time2"] = pd.to_datetime(dat["time2"])
    stats = dat.groupby([trackname]).agg({"time2": ["min", "max"], 
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
    stats["detect_dens"] = stats["nframes"]/stats["dur_s"]

    #printstats = stats[["track", "start", "end", "nframes", "conf_mean", "x_dist", "y_dist", "dur_s"]]
    #print(printstats.sort_values(by = ["conf_mean"], ascending = False))
    return(stats)

def modify_output(file):
    stats = file.groupby(["track_id"]).agg({"time2": "min"})    
    print(stats)
    stats = stats.reset_index()
    cols = ["track", 'start']
    stats.columns = cols

    xx = file["filename"].iloc[0]

    stats["file"] = xx  
    ledge = xx.split("_")[1]
    dates = xx.split("_")[2].split("-")
    dateval = dates[0][2:4]+dates[1]+dates[2]
    time = xx.split("_")[3][0:2]
    a = pd.Series(stats.index).astype("int").astype("str")
    stats["file_id"] = ledge+dateval+time
    b = stats["file_id"].reset_index()["file_id"]
    stats["track_id"] = b+"-"+a
    
    out = file.merge(stats[["track", "track_id"]], left_on = "track_id", right_on = "track", how = "right")
    out["date"] = out["time2"].dt.date
    out = out[["track_id_y", "ledge", "date", "time2", "x", "y", "width", "height", "maxdim", "mindim", "xdiff", "ydiff"]]
    cols = ["track_id", "ledge", "date", "time", "x", "y", "width", "height", "maxdim", "mindim", "xdiff", "ydiff"]
    out.columns = cols
    return(out)


def plot_tracks(track_data, all_data):
    all_data = pd.read_csv(all_data)
    dat = track_data[track_data["track_id"] != -1]
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
    ax = sns.scatterplot(x= dat["time2"], y=dat["y"], color = "red", size = dat["conf"], palette = palette)
    #ax = sns.lineplot(x= dat["time2"], y=dat["y"], color = "red", palette = palette)
    ax = sns.scatterplot(x = all_data["time2"], y = all_data["y"], size = .1, color = "black", marker = "+")
    ax.invert_yaxis()
    ax.grid(False)
    plt.show()
    #plt.savefig("temp/"+"tracks_time_"+file_name+".jpg")
    #plt.close()


def plot_tracks2(track_data):
    
    # General plotting features
    fig, axs = plt.subplots(2)
    tracks = track_data["track"].unique()
    track_data["time2"] = pd.to_datetime(track_data["time2"])
    date = track_data.iloc[0]["time2"].date()
    ledge = track_data.iloc[0]["ledge"]

    # Plot tracks in space 
    for track in tracks: 
        data = track_data[track_data["track"] == track]        
        col = np.random.rand(3,)
        axs[0].plot(data["x"], data["y"], c = col)
        axs[0].grid(False)
        axs[0].text(data.iloc[0]["x"], data.iloc[0]["y"], data.iloc[0]["track"], fontsize = 'xx-small', c = col)
        axs[0].invert_yaxis()
        
        axs[1].plot(data["time2"], data["y"], c = col)
        axs[1].scatter(data["time2"], data["y"], c = "black", s = 10, marker = "|", alpha = .5)
        axs[1].grid(False)
        axs[1].text(data.iloc[0]["time2"], data.iloc[0]["y"], data.iloc[0]["track"], fontsize = 'xx-small')
        axs[1].invert_yaxis()

    fig.suptitle(f'{date}, {ledge}')
    plt.show()
    

def prep_data(input):
    if input.is_file:
        output = pd.read_csv(input)
    else: 
        output = input
    return output 

def insert_to_db(input, output): 
    input = input.reset_index()
    #file = file[file["nframes"] > 5]
    con_local = create_connection(output)
    input.to_sql("Inference", con_local, if_exists='append')

    
def df_from_db(db, cond1, cond2, stats):
    con = create_connection(db)

    sql = (f'SELECT * '
           f'FROM Inference '
           f'WHERE {cond1} AND {cond2};')
   
    if stats: 
        df = pd.read_sql_query(
            sql,
            con, 
            parse_dates = {"start": "%Y-%m-%d %H:%M:%S.%f", "end": "%Y-%m-%d %H:%M:%S.%f"}
            )
    else: 
        df = pd.read_sql_query(
        sql,
        con, 
        parse_dates = {"time2": "%Y-%m-%d %H:%M:%S.%f"}
        )
    return(df)

def predict_from_classifier(dataset):
    
    RandFor = pickle.load(open("models/unmerged_tracks/RandomForests.sav", 'rb'))
    KNear = pickle.load(open("models/unmerged_tracks/KNearest.sav", 'rb'))
    NaiveBayes = pickle.load(open("models/unmerged_tracks/NaiveBayes.sav", 'rb'))
    DecisionTree = pickle.load(open("models/unmerged_tracks/DecisionTrees.sav", 'rb'))
    SVM = pickle.load(open("models/unmerged_tracks/SVM.sav", 'rb'))
    LogReg = pickle.load(open("models/unmerged_tracks/LogisticRegression.sav", 'rb'))

    con = create_connection(dataset)
    cond1 = f'nframes > 1'
    
    sql = (f'SELECT * FROM Inference '
            f'WHERE {cond1};')
    
    dataset = pd.read_sql_query(
        sql, 
        con)
    
    preddata = dataset[["nframes","x_first","x_std","x_last","y_first","y_std","y_last", "conf_min","conf_mean","conf_max","mindim_mean","mindim_std","maxdim_mean","maxdim_std","x_dist","y_dist","dur_s", "detect_dens"]]

    # Transform indata
    ss_pred = StandardScaler()
    preddata_transform = ss_pred.fit_transform(preddata)

    # Make predictions 
    pred_randfor=pd.Series(RandFor.predict(preddata_transform), name = "RandFor")
    pred_knear=pd.Series(KNear.predict(preddata_transform), name = "KNear")
    pred_naive=pd.Series(NaiveBayes.predict(preddata_transform), name = "NaBayes")
    pred_dectree=pd.Series(DecisionTree.predict(preddata_transform), name = "DecTree")
    pred_svm=pd.Series(SVM.predict(preddata_transform), name = "SVM")
    pred_logreg=pd.Series(LogReg.predict(preddata_transform), name = "LogReg")

    preds = pd.concat([dataset["track"], pred_randfor, pred_knear, pred_naive, pred_dectree, pred_svm, pred_logreg], axis = 1)
    preds["multi"] = preds["RandFor"]+preds["KNear"]+preds["NaBayes"]+preds["DecTree"]+preds["SVM"]+preds["LogReg"]
    preds["nofish"] = np.where(preds["multi"] < 3, 1, 0)

    # Combine with original data
#    out = pd.merge(preds, dataset, left_index = True, right_index = True)
    
    preds.to_csv("inference/Unmerged_nofish.csv", sep = ";", decimal = ",")
    return(preds)


# Modify input

def modify_input(dat):
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

    return(dat)


# Run functions
def run_multiple(dir):
    dir = Path(dir)
    allfiles = list(dir.glob("*"))
    nfiles = len(allfiles)
    counter = 0
    for file in allfiles:
        orig_file = file
        file_name = file.stem
        print(file_name)
        precheck = prep_data(orig_file)
        if len(precheck["track_id"].unique()) > 1:
            output1 = merge_tracks(precheck)
            output2 = merge_tracks(output1)
            output3 = merge_tracks(output2)
            output4 = merge_tracks(output3)
            output5 = merge_tracks(output4)
            output6 = associate_points_before(output5, precheck)
            output7 = associate_points_within(output6, precheck)
            output8 = merge_tracks(output7)
            ss = calc_stats(output8, precheck)
            #insert_to_db(ss, "inference/Inference_stats.db")
            output9 = modify_output(output8)
            insert_to_db(output9, "inference/Inference_raw.db")
            #output8.to_csv(f'inference/merged/{file_name}.csv')
            print(f'Finished with file {counter} of {nfiles}')
        counter += 1


def run_single(file):
    orig_file = Path(file)
    print(orig_file)
    precheck = prep_data(orig_file)
    if len(precheck["track_id"].unique()) > 1:
        output1 = merge_tracks(precheck)
        #output2 = merge_tracks(output1)
        #output3 = merge_tracks(output2)
        #output4 = merge_tracks(output3)
        #output5 = merge_tracks(output4)        
        output6 = associate_points_before(precheck, precheck)
        #output7 = associate_points_within(output6, precheck)
        #output8 = merge_tracks(output7)
        ss = calc_stats(output6, precheck)
        ss.to_csv(f'inference/test/{orig_file.name}', sep = ";", decimal = ",")
        plot_tracks(output6, orig_file)
        return(output6)

