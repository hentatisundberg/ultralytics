
import pandas as pd
import numpy as np
import random
import sys
sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/") # Mac
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/") # Sprattus
sys.path.append("/home/jonas/Documents/python/ultralytics-1/code/generic_functions/") # Larus

from functions import create_connection, prep_training_data, train_classifier, predict_from_classifier, df_from_db

# Prep training data 
valid = prep_training_data("../../../../../../mnt/BSP_NAS2_work/fish_model/inference/Inference_stats_nomergeALL.db", 
                           "data/fish_tracks_nomerge_annotations.csv")

# Train model 
train_classifier(valid, False)

# Predict 
inf = predict_from_classifier("../../../../../../mnt/BSP_NAS2_work/fish_model/inference/Inference_stats_nomergeALL.db", False, "ALL")

# Plot 
#dataset = plot_orig_data("inference/Inference_raw_merge.db", "inference/merged_fish.csv", "2022-06-20", "FAR3", 1)
#print(dataset["ledge"].value_counts())

# Plot annotations
#plot_annotations(valid, "nframes", "conf_mean", False, False)

