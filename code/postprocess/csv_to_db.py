


import pandas as pd
import numpy as np
import os
from pathlib import Path
import sqlite3



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


def calc_stats(input_data): 
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

    stats = dat.groupby(["track"]).agg({"time2": ["min", "max"], 
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

    xx = dat["filename"][0]

    stats["file"] = xx
    
    ledge = xx.split("_")[1]
    dates = xx.split("_")[2].split("-")
    dateval = dates[0][2:4]+dates[1]+dates[2]
    time = xx.split("_")[3][0:2]

    stats["Ledge"] = ledge    
    stats["detect_dens"] = stats["nframes"]/stats["dur_s"]

    return(stats)


def insert_to_db(input_file, output_raw, output_stats): 
    
    con_raw = create_connection(output_raw)
    con_stats = create_connection(output_stats)
    
    # Create unique track names    
    file = pd.read_csv(input_file)

    tracks = file[file["track_id"] != -1]

    tracklength = len(tracks)
    
    if tracklength > 0: 
        print(f'number of detections for {input_file} with assigned tracks: {tracklength}')
        stem = Path(input_file).stem
        id0 = stem.split("_")[1]+"_"+stem.split("_")[2]+"_"+stem.split("_")[3][0:2]
        
        id = tracks["track_id"].unique()
        id1 = list(range(0, len(id)))
        
        new = [id0+"-"+str(i) for i in id1]
        
        newnames = pd.DataFrame({"A": id, "track": new})

        tracks = tracks.merge(newnames, left_on = "track_id", right_on = "A", how = "left")

        stats = calc_stats(tracks)

        tracks.to_sql("Inference", con_raw, if_exists='append')
        stats.to_sql("Inference", con_stats, if_exists='append')


allfiles = list(Path("inference/orig/").glob("*FAR3*"))

for file in allfiles:
    insert_to_db(file, "inference/Inference_raw_nomerge.db", "inference/Inference_stats_nomerge.db")


