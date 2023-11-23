
from ultralytics import YOLO
import os
import sys 
import pathlib
import numpy as np
from PIL import Image
import pandas as pd
import torch



from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
#model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='../data/data.yaml', epochs=5, imgsz=640)


