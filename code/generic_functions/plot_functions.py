

import pandas as pd
import matplotlib.pyplot as plt
from functions import create_connection, df_from_db
import numpy as np
import cv2

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
    
    


def plot_all_from_db():
    date = "2022-06-27"
    dat = df_from_db("inference/Inference_raw_nomerge.db", f'ledge == "FAR3"',  f'strftime("%Y-%m-%d", time2) == "{date}"', False)

    tracks = dat["track"].unique()
    # Plot 
    fig, axs = plt.subplots(figsize=(15, 8))
    axs.scatter(dat["time2"], dat["y"], c = "black", s = 8, marker = "*")
    
    for track in tracks: 
        d2 = dat[dat["track"] == track]
        axs.plot(d2["time2"], d2["y"], c = "red")
        
    axs.grid(False)
    plt.show()



def plot_track_from_db(): 
    track = "FAR3_2022-06-26_17-10"
    dat = df_from_db("inference/Inference_raw_nomerge.db", f'ledge == "FAR3"', f'track == "{track}"', False)

    # Plot 
    fig, ax = plt.subplots(figsize=(15, 10))
    imgplot = ax.imshow(bg)
    ax.plot(dat["x"]+(.5*dat["width"]), dat["y"]+(.5*dat["height"]), c = "yellow", linewidth = 5)
    #ax.invert_yaxis()
    ax.grid(False)
    plt.show()


def plot_raw_by_time(): 
    df["time2"] = pd.to_datetime(df["time"]*1000*1000*1000)
    assigned["time2"] = pd.to_datetime(assigned["time"]*1000*1000*1000)
    tracks = assigned["track_id"].unique()
    dat_notrack = df[df["track_id"] == -1]
    dat_track = df[df["track_id"] != -1]

    # Plot 
    fig, axs = plt.subplots(figsize=(15, 8))

    for track in tracks: 
        td = assigned[assigned["track_id"] == track]
        col = np.random.rand(3,)
        axs.scatter(td["time2"], td["y"], s=80, c = col)    

    axs.scatter(dat_notrack["time2"], dat_notrack["y"], c = "black", s = 3, marker = ".")
    axs.scatter(dat_track["time2"], dat_track["y"], c = "red", s = 7, marker = "*")    

    axs.grid(False)
    
    plt.show()





def plot_orig_data(db, preddata, date, ledge, fishlimit):
    
    preddata = pd.read_csv(preddata, sep = ";", decimal = ",")
    
    con = create_connection(db)
    cond1 = f"ledge == '{ledge}'"
 
    sql = (f'SELECT * '
        f'FROM Inference '
        f'WHERE {cond1};')

    dataset = pd.read_sql_query(
    sql,
    con, 
    parse_dates = {"time2": "%Y-%m-%d %H:%M:%S.%f"})


    dataset["date"] = dataset["time2"].dt.date
    
    #dataset = dataset[dataset["date"] == date]
    dataset = dataset[dataset["ledge"] == ledge]
    pred_raw = pd.merge(dataset, preddata[["track", "multi"]], on = "track", how = "right")

    fish = pred_raw[pred_raw["multi_y"] > fishlimit]
    nofish = pred_raw[pred_raw["multi_y"] <= fishlimit]

    trackids = list(fish["track"].unique())
    fish = fish[fish["track"].isin(trackids[0:9])]
    
    # Plot 1
    fig, ax = plt.subplots()

    ax.plot(fish["x"], fish["y"], c = "r", alpha = .5, label = "Fish")
    #ax.scatter(nofish["x"], nofish["y"], c = "b", alpha = .3, label = "No Fish", s = 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    plt.suptitle(f'{date}, fishlimit = {fishlimit}')
    plt.legend()
    plt.show()

    return(dataset)


def plot_results(db, inference, x, y, logx, logy, fishlimit):

    stats = df_from_db(db, "ledge != 'X'", "ledge != 'X'", True)
    inference = pd.read_csv(inference, sep = ";", decimal = ",")
    data = pd.merge(stats, inference, on = "track", how = "right")

    fish = data[data["multi"] > fishlimit]
    nofish = data[data["multi"] <= fishlimit]

    if logx: 
        x1 = np.log(fish[x])
        x2 = np.log(nofish[x])
    else: 
        x1 = fish[x]
        x2 = nofish[x]

    if logy: 
        y1 = np.log(fish[y])
        y2 = np.log(nofish[y])
    else: 
        y1 = fish[y]
        y2 = nofish[y]

    fig, ax = plt.subplots()
    ax.scatter(x1, y1, c = "r", alpha = .3, label = "Fish", s = 1)
    ax.scatter(x2, y2, c = "b", alpha = .3, label = "No Fish", s = 1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.legend()
    plt.show()
    return(fish)



def plot_annotations(data, x, y, logx, logy):
    
    fish = data[data["Valid"] == 1]
    nofish = data[data["Valid"] == 0]

    if logx: 
        x1 = np.log(fish[x])
        x2 = np.log(nofish[x])
    else: 
        x1 = fish[x]
        x2 = nofish[x]

    if logy: 
        y1 = np.log(fish[y])
        y2 = np.log(nofish[y])
    else: 
        y1 = fish[y]
        y2 = nofish[y]

    fig, ax = plt.subplots()
    ax.scatter(x1, y1, c = "r", alpha = .3, label = "Fish", s = 10)
    ax.scatter(x2, y2, c = "b", alpha = .3, label = "No Fish", s = 10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.legend()
    plt.show()



