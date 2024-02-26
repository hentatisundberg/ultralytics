
import pandas as pd
import numpy as np
import sqlite3
from functions import create_connection, prep_training_data, train_classifier, predict_from_classifier
from plot_functions import plot_annotations, plot_orig_data, plot_results
import matplotlib.pyplot as plt

# Prep training data 
valid = prep_training_data("inference/Inference_stats_merge.db", "data/fish_track_merge_annotations.csv")

# Train model 
#train_classifier(valid, True)

# Predict 
inf = predict_from_classifier("inference/Inference_stats_merge.db", True)

# Plot 
dataset = plot_orig_data("inference/Inference_raw_merge.db", "inference/merged_fish.csv", "2022-06-20", "FAR3", 1)
print(dataset["ledge"].value_counts())

# Plot annotations
plot_annotations(valid, "nframes", "conf_mean", False, False)

# Plot 
fish = plot_results("inference/Inference_stats_merge.db", "inference/merged_fish.csv", "conf_mean", "nframes", False, True, 4)


fish["date"] = fish["start"].dt.date
fish["H"] = fish["start"].dt.strftime("%H").astype("int")

rangex = range(2, 23)
xlist = list(rangex)
xlist2 = [x + 0.5 for x in xlist]

fig, ax = plt.subplots()
ax.hist(fish["H"], color = "darkred", label = "Fish", width = 0.9, bins = rangex)
ax.set_xlabel("Hour")
ax.set_xticks(xlist)
ax.set_ylabel(y)
fig.suptitle("Time of fish deliveries (H), FAR3, 2022")
#plt.legend()
plt.show()


# Fish size?

fig, ax = plt.subplots()
ax.scatter(fish["maxdim_mean"], fish["mindim_mean"], c = "r", alpha = .3, label = "Fish", s = 4)
plt.legend()
plt.show()

# Trend fish size
fish["size"] = fish["maxdim_mean"]*fish["mindim_mean"]
fig, ax = plt.subplots()
ax.scatter(fish["start"], fish["size"], c = "r", alpha = .5, label = "Fish", s = 12)
plt.legend()
plt.show()




# 

