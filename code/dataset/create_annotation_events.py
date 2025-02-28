
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys 

# Read data
out = pd.read_csv("data/compiled_nanov5852_v3.csv")
out["datetime"] = pd.to_datetime(out["datetime"])

# One station at the time
stat = "EJDER1"
out = out[out["station"] == stat & out["class"] != 1]

# Summarize frame counts for all classes combined except for class 1
dx2 = out.groupby("datetime").sum(["frame"]).reset_index()

# Subset based on confidence level
#dx2 = dx2[dx2["conf"] > 0.3]

# Full date time sequence
date_rng = pd.date_range(start=dx2["datetime"].min(), end=dx2["datetime"].max(), freq='2S')

# Fill missing values
dx2 = dx2.set_index('datetime').reindex(date_rng, fill_value=0).reset_index().rename(columns={'index': 'datetime'})

# Plot time series of one station at the time, with datetime as time series and bars of bars for counts
fig, ax = plt.subplots(1, 1)

# Create a new y series which is a 10 point running mean of the original y series
y = dx2["frame"]/50
y_rolling = y.rolling(window=10).mean()

ax.plot(dx2["datetime"], y, c = "red", alpha = 0.4)
ax.plot(dx2["datetime"], y_rolling, color = "black", alpha = 0.9) 

plt.savefig("dump/eider1.png")
plt.show()
