
from ultralytics import YOLO

# Load a model
model = YOLO('../models/yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='../config/data.yaml', epochs=50, imgsz=960, device = [0, 1])



