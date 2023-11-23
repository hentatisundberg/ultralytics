


from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO('model/yolov8n.pt')

# Perform object detection on an image using the model
results = model.track(source="https://youtu.be/LNwODJXcvt4")

## Process results list
#for r in results:
#    boxes = r.boxes.xyxy  # Boxes object for bbox outputs
    


