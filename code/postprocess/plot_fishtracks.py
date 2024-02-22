


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import sqlite3
from PIL import Image
from functions import df_from_db

# Read background image
bg = np.asarray(Image.open('data/bg.jpg'))


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
    dat = pd.read_csv("inference/orig/Auklab1_FAR3_2022-06-27_04.00.00.csv")
    dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)
    dat_notrack = dat[dat["track_id"] == -1]
    dat_track = dat[dat["track_id"] != -1]

    # Plot 
    fig, ax = plt.subplots(figsize=(15, 10))
    imgplot = ax.imshow(bg)
    ax.scatter(dat_notrack["x"], dat_notrack["y"], c = "yellow", s = 3, marker = ".")
    ax.scatter(dat_track["x"], dat_track["y"], c = "red", s = 7, marker = "*")    
    ax.grid(False)
    plt.show()

    # Plot 
    fig, axs = plt.subplots(2, figsize=(15, 8))
    axs[0].scatter(dat_notrack["time2"], dat_notrack["y"], c = "black", s = 3, marker = ".")
    axs[0].scatter(dat_track["time2"], dat_track["y"], c = "red", s = 7, marker = "*")    
    axs[0].grid(False)
    axs[1].scatter(dat_notrack["time2"], dat_notrack["x"], c = "black", s = 3, marker = ".")
    axs[1].scatter(dat_track["time2"], dat_track["x"], c = "red", s = 7, marker = "*")    
    axs[1].grid(False)
    
    plt.show()


#plot_track_from_db()
plot_raw_by_time()
#plot_all_from_db()

#plot_tracks(dat, "FAR3", "2022-06-27")