from ultralytics import YOLO

# Load a model
model = YOLO('../models/yolov8m.pt')  # load the best model thus far

# Train the model
results = model.train(data='../config/config_seabirds.yaml', 
    epochs=50, 
    imgsz=960, 
    device = [1] 
    #lr0 = 0.00782,
    #lrf = 0.01181,
    #momentum= 0.96128,
    #weight_decay= 0.00048,
    #warmup_epochs= 4.04741,
    #warmup_momentum= 0.74238,
    #box= 8.91645,
    #cls= 0.41067,
    #dfl= 1.77856,
    #hsv_h= 0.01073,
    #hsv_s= 0.83354,
    #hsv_v= 0.32446,
    #degrees= 0.0,
    #translate= 0.10994,
    #scale= 0.36825,
    #shear= 0.0,
    #perspective= 0.0,
    #flipud= 0.0,
    #fliplr= 0.3682,
    #mosaic= 1.0,
    #mixup= 0.0,
    #copy_paste= 0.0
    )



