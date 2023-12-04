
from ultralytics import YOLO

# Load a model
model = YOLO('../runs/train15/weights/best.pt')  # load a pretrained model (recommended for training)

# Tune hyper-parameters 
model.tune(data='../config/data.yaml', 
    epochs=30, 
    iterations = 300, 
    optimizer = 'AdamW', 
    plots = False, 
    save = False, 
    val = False)

