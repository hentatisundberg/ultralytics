
from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train20/weights/best.pt')  # load the best model so far

# Tune hyper-parameters 
model.tune(data='../config/data.yaml', 
    epochs=20, 
    iterations = 30, 
    optimizer = 'AdamW', 
    plots = False, 
    save = True, 
    val = False, 
    device = [0, 1])


