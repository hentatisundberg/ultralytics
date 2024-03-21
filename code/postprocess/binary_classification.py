
import pandas as pd
import numpy as np
import random
from functions import create_connection, prep_training_data, train_classifier, predict_from_classifier, df_from_db
from plot_functions import plot_annotations, plot_orig_data, plot_results
import matplotlib.pyplot as plt
import cv2

# Prep training data 
#valid = prep_training_data("inference/Inference_stats_merge.db", "data/fish_tracks_merge_annotations.csv")

# Train model 
#train_classifier(valid, True)

# Predict 
inf = predict_from_classifier("inference/Inference_stats_merge.db", True)

# Plot 
#dataset = plot_orig_data("inference/Inference_raw_merge.db", "inference/merged_fish.csv", "2022-06-20", "FAR3", 1)
#print(dataset["ledge"].value_counts())

# Plot annotations
#plot_annotations(valid, "nframes", "conf_mean", False, False)

