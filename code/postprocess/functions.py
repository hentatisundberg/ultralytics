

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


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




def cut_vid(row, vidpath, savepath, trackname): 

    datefold = str(row["start"])[0:10]

    starttime_vid = row["start"].floor("H")
    startclip = row["start"]
    endclip = row["end"]
    starttime_name = starttime_vid.strftime("%Y-%m-%d_%H.%M.%S")

    if any(pd.isnull([startclip, endclip, starttime_vid])):
        print("skip")

    else: 
        startsec = (row["start"]-starttime_vid)/np.timedelta64(1,'s')
        endsec = (row["end"]-starttime_vid)/np.timedelta64(1,'s')

        ledge = row[trackname].split("_")[0]
        vid_rel_path = f"{vidpath}{datefold}/"
        full_path = f'{vid_rel_path}Auklab1_{ledge}_{starttime_name}.mp4'
        print(full_path)

        if os.path.isfile(full_path):
            tracknamex = row[trackname]
            filename_out = f"{savepath}{tracknamex}.mp4"
            ffmpeg_extract_subclip(
                full_path,
                startsec,
                endsec,
                targetname = filename_out
            )
            #print(filename_out)
            return(filename_out)



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
    stats["ledge"] = dat["ledge"].iloc[0]
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


def prep_training_data(tracks, valid):
    con = create_connection(tracks)
    tracks = pd.read_sql_query(
    """SELECT * FROM Inference""",
    con)

    # Prepare Valid data and merge with track data
    valid = pd.read_csv(valid, sep = ";")
    valid = pd.merge(valid, tracks, on = "track", how = "left")
    valid = valid[valid['ledge'].isin(["FAR3", "FAR6", "TRI3", "TRI6"])]
    
    # Remove tracks with one detection and only include annotated tracks 
    valid = valid[valid["nframes"] > 1] 
    valid = valid[valid["Valid"] > -1]

    return(valid)


def train_classifier(dataset, merge):

    if merge: 
        fold = "merged"
    else: 
        fold = "unmerged"

    #dataset = pd.read_csv(dataset, sep = ";", decimal = ",")  
    dataset = dataset[~dataset["Valid"].isna()]

    # Define response and target
    X_expl = dataset[["track", "start", "end"]]
    X = dataset[["nframes",	"x_first","x_std","x_last","y_first","y_std","y_last", "conf_min","conf_mean","conf_max","mindim_mean","mindim_std","maxdim_mean","maxdim_std","x_dist","y_dist","dur_s", "detect_dens"]]
    y = dataset['Valid']

    # Define data sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)

    # Save orig data for later validation
    X_expl = X_expl.merge(X, left_index = True, right_index = True)

    # Normalize data
    ss_train = StandardScaler()
    X_train = ss_train.fit_transform(X_train)

    ss_test = StandardScaler()
    X_test = ss_test.fit_transform(X_test)

    # Train models
    models = {}
    models['Logistic Regression'] = LogisticRegression()
    models['Support Vector Machines'] = LinearSVC()
    models['Decision Trees'] = DecisionTreeClassifier()
    models['Random Forest'] = RandomForestClassifier()
    models['Naive Bayes'] = GaussianNB()
    models['K-Nearest Neighbor'] = KNeighborsClassifier()

    accuracy, precision, recall = {}, {}, {}
    df_out = pd.DataFrame()

    for key in models.keys():
        
        # Fit the classifier
        models[key].fit(X_train, y_train)
        
        # Make predictions
        predictions = models[key].predict(X_test)
        df_out[key] = predictions

        # Calculate metrics
        accuracy[key] = accuracy_score(predictions, y_test)
        precision[key] = precision_score(predictions, y_test)
        recall[key] = recall_score(predictions, y_test)

    temp = pd.merge(y_test, X_expl["track"], left_index = True, right_index = True, how = "left").reset_index()
    df_out = pd.merge(temp, df_out, left_index = True, right_index = True)

    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()

    print(df_model)
    df_out.to_csv("inference/multimod_valid_merge.csv", sep = ";", decimal = ",")
    
    # Save model
    pickle.dump(models['Random Forest'], open(f"models/{fold}_tracks/RandomForests.sav", 'wb'))
    pickle.dump(models['K-Nearest Neighbor'], open(f"models/{fold}_tracks/KNearest.sav", 'wb'))
    pickle.dump(models['Naive Bayes'], open(f"models/{fold}_tracks/NaiveBayes.sav", 'wb'))
    pickle.dump(models['Decision Trees'], open(f"models/{fold}_tracks/DecisionTrees.sav", 'wb'))
    pickle.dump(models['Support Vector Machines'], open(f"models/{fold}_tracks/SVM.sav", 'wb'))
    pickle.dump(models['Logistic Regression'], open(f"models/{fold}_tracks/LogisticRegression.sav", 'wb'))



def predict_from_classifier(dataset, merge):
    
    if merge: 
        fold = "merged"
    else: 
        fold = "unmerged"

    RandFor = pickle.load(open(f"models/{fold}_tracks/RandomForests.sav", 'rb'))
    KNear = pickle.load(open(f"models/{fold}_tracks/KNearest.sav", 'rb'))
    NaiveBayes = pickle.load(open(f"models/{fold}_tracks/NaiveBayes.sav", 'rb'))
    DecisionTree = pickle.load(open(f"models/{fold}_tracks/DecisionTrees.sav", 'rb'))
    SVM = pickle.load(open(f"models/{fold}_tracks/SVM.sav", 'rb'))
    LogReg = pickle.load(open(f"models/{fold}_tracks/LogisticRegression.sav", 'rb'))

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

    # Combine with original data
#    out = pd.merge(preds, dataset, left_index = True, right_index = True)
    
    preds.to_csv(f'inference/{fold}_fish.csv', sep = ";", decimal = ",")
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
    dat["date"] = dat["time2"].dt.date
    
    # New track ids
    track_id = dat["track_id"].unique()
    nlist = list(range(1, len(track_id)+1)) 
    newnum = [str(item).zfill(4) for item in nlist]
    
    time = dat["time2"].iloc[0].strftime('%Y-%m-%d_%H-')
    ledge = dat["ledge"].iloc[0]
    track = pd.Series([ledge + "_" + time + i for i in newnum], name = "track")
    track_id = pd.Series(track_id, name = "track_id")
    df = pd.concat([track_id, track], axis = 1) 
    dat = dat.merge(df, on = "track_id", how = "left")

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

