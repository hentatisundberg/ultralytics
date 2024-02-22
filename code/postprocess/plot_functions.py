

import pandas as pd
import matplotlib.pyplot as plt


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



