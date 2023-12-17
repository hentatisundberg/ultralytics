
from ultralytics import YOLO

# Load a model
model = YOLO('../../../../../../mnt/BSP_NAS2_work/seabirds/models/best_train58.pt')  # load the best model so far

# Tune hyper-parameters 
model.tune(data='../config/config_seabirds.yaml', 
    epochs=30, 
    iterations = 30, 
    optimizer = 'AdamW', 
    plots = False, 
    save = False, 
    val = False, 
    device = [0, 1])


