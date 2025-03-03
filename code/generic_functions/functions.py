
import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path
#from itertools import combinations
#from itertools import product
#import sys
import sqlite3
from datetime import datetime 
import pickle
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import LinearSVC
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score, precision_score, recall_score
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from ultralytics import YOLO
import yaml

# Functions 

def init_dir(x):
    p1 = x[["x_first", "x_nth"]].tolist()
    p2 = x[["y_first", "y_nth"]].tolist()
    xs_adj = [(p1[0]/2592)-.5, (p1[1]/2592)-.5]
    ys_adj = [(p2[0]/1520)-.5, (p2[1]/1520)-.5]
    val = np.degrees(np.arctan2(xs_adj, ys_adj))[1]
    val_out = np.where(val < 0, 360+val, val).tolist()
    return val_out

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5


def movement(x, nframes):
    ndata = len(x)
    start = 0
    if ndata < nframes:
        end = ndata-2
    else: 
        end = nframes
    point1 = (x.iloc[start][["x", "y"]]).tolist()
    point2 = (x.iloc[end][["x", "y"]]).tolist()
    eu = euclidean(point1, point2)
    return(eu)


def nth10(x):
    if len(x) < 10:
        return(x.iloc[-1])
    else: 
        return(x.iloc[9]) 
  

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
                    #print("minpos is")
                    #print(minpos)
                    track_temp = pd.concat([minpos, track_temp]) # Update track data
                    #print("track temp is")
                    #print(track_temp)
                    cand_bef.drop(minpos.index, inplace = True) # Delete from candidates
                    #print(track_temp)
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


# Minutes and seconds to seconds
def minsec2sec(x):
    return int(x.split(":")[0])*60+int(x.split(":")[1])



def cut_vid_simpler(video_dir, row, savepath, addseconds): 

    # Build path to video
    video = row["filename"]
    print(video)
    video_station = video.split("_")[-3]
    video_date = video.split("_")[-2]

    full_path = f'{video_dir}{video_station}/{video_date}/{video}'
    print(full_path)

    startclip = row["start"]
    endclip = row["end"]
    
    if any(pd.isnull([startclip, endclip, video])):
        print("skip")

    else: 
        startsec = minsec2sec(startclip)-addseconds
        endsec = minsec2sec(endclip)+addseconds
        print(startsec)
        print(endsec)

        if os.path.isfile(full_path):
            filename_out = f"{savepath}{Path(video).stem}_{startsec}_{endsec}.mp4"
            ffmpeg_extract_subclip(
                full_path,
                startsec,
                endsec+startsec,
                filename_out
            )
            return(filename_out)
        else: 
            print("file not found")


#rx = {"filename": "../../../../Downloads/NVR_Hien_EJDER7_2023-05-07_19.00.00.mp4", "start": "00:00", "end": "01:00"}
#row = pd.DataFrame(rx, index = [0])
#cut_vid_simpler(row, "../../../../Downloads/", 0)



def cut_vid(row, vidpath, savepath, addseconds): 

    datefold = str(row["start"])[0:10]

    starttime_vid = row["start"].floor("H")
    startclip = row["start"]
    endclip = row["end"]
    starttime_name = starttime_vid.strftime("%Y-%m-%d_%H.%M.%S")

    if any(pd.isnull([startclip, endclip, starttime_vid])):
        print("skip")

    else: 
        startsec = (row["start"]-starttime_vid)/np.timedelta64(1,'s')-addseconds
        endsec = (row["end"]-starttime_vid)/np.timedelta64(1,'s')+addseconds

        ledge = row["track"].split("_")[0]
        yrtext = row["start"].year 
        vid_rel_path = f"{vidpath}Video{yrtext}/{ledge}/{datefold}"
        full_path = f'{vid_rel_path}/Auklab1_{ledge}_{starttime_name}.mp4'
        print(full_path)

        if os.path.isfile(full_path):
            tracknamex = row["track"]
            filename_out = f"{savepath}{tracknamex}.mp4"
            ffmpeg_extract_subclip(
                full_path,
                startsec,
                endsec,
                targetname = filename_out
            )
            return(filename_out)



def calc_stats2(input_data, trackname): 
    dat = input_data
    dat["time2"] = pd.to_datetime(dat["time2"])
    stats = dat.groupby([trackname]).agg({"time2": ["min", "max"], 
                                            "frame": "count", 
                                            "x": ["first", "std", "last", nth10],
                                            "y": ["first", "std", "last", nth10], 
                                            "conf": ["min", "mean", "max"], 
                                            "mindim": ["mean", "std"], 
                                            "maxdim": ["mean", "std"],
                                            "xdiff": "sum",
                                            "ydiff": "sum"})
    
    stats = stats.droplevel(1, axis = 1).reset_index()
    
    cols = ["track", 'start','end', 'nframes', "x_first", "x_std", "x_last", "x_nth", "y_first", "y_std", "y_last", "y_nth", "conf_min", "conf_mean", "conf_max", "mindim_mean", "mindim_std", "maxdim_mean", "maxdim_std", "x_dist", "y_dist"]
    stats.columns = cols
    
    timeelapse = stats["end"]-stats["start"]
    
    dur = []
    
    for i in timeelapse:
        dur.append(i.total_seconds())    
    stats["dur_s"] = dur
    stats["detect_dens"] = stats["nframes"]/stats["dur_s"]
    
    # Initital movement 
    init_movement = []
    for row in stats.index:
        p1 = stats[["x_first", "y_first"]].iloc[row].tolist()
        p2 = stats[["x_nth", "y_nth"]].iloc[row].tolist()
        init_movement.append(euclidean(p1, p2))
    stats["init_move"] = init_movement

    # Initital direction 
    init_direction = []
    for row in stats.index:
        p1 = stats[["x_first", "x_nth"]].iloc[row].tolist()
        p2 = stats[["y_first", "y_nth"]].iloc[row].tolist()
        xs_adj = [(p1[0]/2592)-.5, (p1[1]/2592)-.5]
        ys_adj = [(p2[0]/1520)-.5, (p2[1]/1520)-.5]
        val = np.degrees(np.arctan2(xs_adj, ys_adj))[1]
        val_out = np.where(val < 0, 360+val, val).tolist()
        init_direction.append(val_out)
    stats["init_dir"] = init_direction    

    # Ledge info
    stats["ledge"] = dat["ledge"].iloc[0]
    
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
    #input = input.reset_index()
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
            parse_dates = {"start": "%Y-%m-%dT%H:%M:%S.%f", "end": "%Y-%m-%dT%H:%M:%S.%f"}
            )
    else: 
        df = pd.read_sql_query(
        sql,
        con, 
        parse_dates = {"time2": "%Y-%m-%dT%H:%M:%S.%f"}
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
    X = dataset[["nframes",	"x_first","x_std","x_last","x_nth", "y_first","y_std","y_last", "y_nth", "conf_min","conf_mean","conf_max","mindim_mean","mindim_std","maxdim_mean","maxdim_std","x_dist","y_dist","dur_s", "detect_dens", "init_move", "init_dir"]]
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

    #print(df_model)
    df_out.to_csv(f"inference/multimod_valid_{fold}.csv", sep = ";", decimal = ",")
    
    # Save model
    pickle.dump(models['Random Forest'], open(f"models/{fold}_tracks/RandomForests.sav", 'wb'))
    pickle.dump(models['K-Nearest Neighbor'], open(f"models/{fold}_tracks/KNearest.sav", 'wb'))
    pickle.dump(models['Naive Bayes'], open(f"models/{fold}_tracks/NaiveBayes.sav", 'wb'))
    pickle.dump(models['Decision Trees'], open(f"models/{fold}_tracks/DecisionTrees.sav", 'wb'))
    pickle.dump(models['Support Vector Machines'], open(f"models/{fold}_tracks/SVM.sav", 'wb'))
    pickle.dump(models['Logistic Regression'], open(f"models/{fold}_tracks/LogisticRegression.sav", 'wb'))



def predict_from_classifier(dataset, merge, ledge):
    
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
    
    preddata = dataset[["nframes","x_first","x_std","x_last", "x_nth", "y_first","y_std","y_last", "y_nth", "conf_min","conf_mean","conf_max","mindim_mean","mindim_std","maxdim_mean","maxdim_std","x_dist","y_dist","dur_s", "detect_dens", "init_move", "init_dir"]]

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
    
    preds.to_csv(f'inference/{fold}_fish{ledge}.csv', sep = ";", decimal = ",")
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



def compress_annotate_vid_nodetect(inputdata, file, savepath):
    
    name = file.name
    
    if name[0] != ".":
        track = file.stem
        output = savepath+name

        plotdata = inputdata[inputdata["track"] == track][["track", "x", "y", "width", "height"]].reset_index()
        ndetections = len(plotdata)
        print(f'number of detections = {ndetections}')

        if ndetections > 3 & ndetections < 1000:
        
            cap = cv2.VideoCapture(str(file))
            nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            print(f'number of frames = {nframes}')
            if not cap.isOpened():
                print("Error: Could not open the input video file")
                exit()
            # XVID better than MJPG. DIVX = XVID
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change this to your desired codec
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            font = cv2.FONT_HERSHEY_SIMPLEX 

            out = cv2.VideoWriter(output, fourcc, frame_rate, frame_size, isColor=True)

            count = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret==True:
                    
                    # Filename
                    if count > (ndetections-2):
                        count = 0
                    #print(count)
                    cv2.putText(frame, f'{name}',  
                        (50, 150),  
                        font, 3,  
                        (255, 255, 255),  
                        3,  
                        cv2.LINE_4) 
                    
                    #for row in range(len(plotdata)-1):
                        #print(row)
                        #print(row+1)
                    x1 = int(plotdata.iloc[count]["x"]+(.5*plotdata.iloc[count]["width"]))
                    y1 = int(plotdata.iloc[count]["y"]+(.5*plotdata.iloc[count]["height"]))                   
                    x2 = int(plotdata.iloc[(count+1)]["x"]+(.5*plotdata.iloc[(count+1)]["width"]))
                    y2 = int(plotdata.iloc[(count+1)]["y"]+(.5*plotdata.iloc[(count+1)]["height"]))                   
                    
                    frame = cv2.circle(frame, (x1, y1), 100, (255, 255, 255), 1)
                    #frame = cv2.line(frame, startpoint, endpoint, (255, 255, 255), 20)                   

                    out.write(frame)
                    count += 1
                else:
                    break

        # Release everything if job is finished
            cap.release()
            out.release()
            cv2.destroyAllWindows()



# Read a video and save all frames as images
def save_all_frames(video_path, image_folder):
    try: 
        vidname = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open the input video file")
            exit()
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                countnum = str(count).zfill(4)
                cv2.imwrite(f'{image_folder}/{vidname}_{countnum}.png', frame)
                count += 1
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        print(f'number of frames = {count}')
    except: 
        print("Error: Could not open the input video file")
        pass




def euclidean_images(img1, img2):
    
    # Flatten the images
    image1_flat = img1.flatten()
    image2_flat = img2.flatten()
    
    # Compute the Euclidean distance
    distance = np.linalg.norm(image1_flat - image2_flat)
    return distance


# Function for looking through images in a folder and remove those that are very similar to the previous one
def remove_similar_images(folder, similarity_thresh):
    files = list(Path(folder).glob("*.png"))
    files.sort()
    remove = []
    for i in range(0, len(files)-1):
        print(f'reading {files[i+1]}')
        img1 = cv2.imread(files[i])
        img2 = cv2.imread(files[i+1])
        dist = euclidean_images(img1, img2)
        print(f'distance to {files[i]} = {dist}')
        if dist < similarity_thresh:
            remove.append(files[i+1])
            print(f'added {files[i+1]} to remove list')
        else: 
            pass
    return remove





def annotate_images(yolo_model, im_outfold, yaml_outfold):

    # Load a pretrained YOLO model
    model = YOLO(yolo_model)

    # List of videos for inference 
    ims = os.listdir(im_outfold)

    # Run
    for im in ims: 

        if len(im) > 20:

            results = model(f'{im_outfold}/{im}')

            # Width and height
            imread = cv2.imread(f'{im_outfold}/{im}')
            width = imread.shape[1]
            height = imread.shape[0]

            # Process results list
            boxes = []
            boxesxyxy = []
            classes = []
            confs = []

            for r in results:
                boxes.append(r.boxes.xywh.tolist()) 
                boxesxyxy.append(r.boxes.xyxy.tolist())
                classes.append(r.boxes.cls.tolist())
                confs.append(r.boxes.conf.tolist())

            # Concatenate outputs
            boxesx = sum(boxes, [])
            boxesxyxy2 = sum(boxesxyxy, [])

            # Save as data frames
            nobj = len(boxesx)

            filename = im.replace(".jpg", ".txt")
            filename_simpl = im.replace(".png", "")

            d = {"name": ['crow', 'eider_female', 'eider_male', 'gull', 'razorbill'], 
                "class": [0, 1, 2, 3, 4]}
            class_ids = pd.DataFrame(d)    

            # .yaml
            # Always in file
            data_dict = {}
            data_dict["image"] = filename
            data_dict["size"] = {"depth": 3, "height": height, "width": width}
            data_dict["source"] = {"framenumber": 0, "path": "na", "video": "na"}
            data_dict["state"] = {"verified": False, "warnings": 0}


            if nobj > 0:

                data_dict["objects"] = []

                for row in range(0, nobj):
                    print(f'classes = {classes}')
                    print(f'nobj = {nobj}')
                    print(f'row = {row}')
                    tdat = boxesxyxy2[row]
                    #classdat = row #works
                    classdat = classes[0][row]
                    print(f'classdat = {classdat}')
                    classname = class_ids[class_ids["class"] == classdat]["name"].item()
                    print(f'classname = {classname}')
                    #confdat = confs[row]
                    
                    data_dict["objects"].append(
                        {
                            "bndbox": {
                                "xmax": tdat[2],
                                "xmin": tdat[0],
                                "ymax": tdat[3],
                                "ymin": tdat[1],
                            },
                            "name": classname

                        }
                    )
            
            write_yaml_to_file(yaml_outfold, data_dict, filename_simpl)


            # Plain annotation
            #if nobj > 0:

                #y = np.empty([nobj, 5], dtype = float)
                #for row in range(0, nobj):
                #    y[row, 1] = (boxesx[row][0]+(.5*boxesx[row][2]))/width # x 
                #    y[row, 2] = (boxesx[row][1]+(.5*boxesx[row][3]))/height # y 
                #    y[row, 3] = (boxesx[row][2])/width # w 
                #    y[row, 4] = (boxesx[row][3])/height # h 
                
                #np.savetxt(f'../dataset/annotations/{filename}', y, fmt="%i %1.4f %1.4f %1.4f %1.4f")
            
            #else:
             #   open(f'../dataset/annotations/{filename}', 'a').close()

def write_yaml_to_file(yaml_outfold, py_obj,filename_simpl):
    with open(f'{yaml_outfold}{filename_simpl}.yaml', 'w',) as f :
        yaml.dump(py_obj,f,sort_keys=False) 


