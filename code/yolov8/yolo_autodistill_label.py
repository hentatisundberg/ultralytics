from autodistill_yolov8 import YOLOv8Base
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our YOLOv8 classes
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model

# replace weights_path with the path to your YOLOv8 weights file
base_model = YOLOv8Base(ontology=CaptionOntology({"fish": "fish"}), weights_path="../models/best.pt")

# run inference on a single image
#results = base_model.predict("../images/bus.jpg")

# Predict for a whole data set
base_model.label(
  input_folder="../images",
  output_folder="../dataset"
)