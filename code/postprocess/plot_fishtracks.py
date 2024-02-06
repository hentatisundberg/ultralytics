


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import sqlite3
import seaborn as sns

# Read data 
dat = pd.read_csv("inference/Validated_fishtracks.csv")


def plot_tracks(track_data, ledge, date):
    dat = track_data
    dat["start"] = pd.to_datetime(dat["start"])
    dat["end"] = pd.to_datetime(dat["end"])
    dat["date"] = dat["start"].dt.date

    dati = dat.loc[(dat["Ledge"] == ledge) & (dat["date"] == pd.to_datetime(date))]
    print(dat)

    # General plotting features
    palette = sns.color_palette("bright")
    sns.set(rc = {'axes.facecolor': 'white'})
    
    # Plot tracks over time 
    ax = sns.scatterplot(x= dati["x_last"], y=dati["y_last"], hue = dati["track_id"], palette = palette)
    ax = sns.lineplot(x= dati["x_last"], y=dati["y_last"], hue = dati["track_id"], palette = palette)
    ax.invert_yaxis()
    ax.grid(False)
    plt.show()

    # Fish sizes
    ax = sns.scatterplot(x= dat["maxdim_mean"], y=dat["mindim_mean"], palette = palette)
    ax.invert_yaxis()
    ax.grid(False)
    plt.show()


plot_tracks(dat, "FAR3", "2022-06-27")