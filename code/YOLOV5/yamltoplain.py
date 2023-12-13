import os
import yaml
import numpy as np
from pathlib import Path

def xyxy2xywhn(x: np.ndarray, width: int, height: int) -> np.ndarray:
    assert x.ndim == 2 
    assert x.shape[1] == 4
    y = np.empty(x.shape)
    y[:, 0] = np.clip((x[:, 0] + x[:, 2]) * 0.5 / width, 0, 1)  # x center
    y[:, 1] = np.clip((x[:, 1] + x[:, 3]) * 0.5 / height, 0, 1)  # y center
    y[:, 2] = np.clip((x[:, 2] - x[:, 0]) / width, 0, 1)  # width
    y[:, 3] = np.clip((x[:, 3] - x[:, 1]) / height, 0, 1)  # height
    return y

#labels_dir = './labels'
#labels_dir = "../../../../../mnt/BSP_NAS2_work/fish_model/annotations/train/"
labels_dir = "../data/train"


onces = True

for train_val_dir in os.listdir(labels_dir):
    file_list = os.listdir(os.path.join(labels_dir, train_val_dir))
    plain_annotation = None
    for file in file_list:
        file_path = os.path.join(labels_dir, train_val_dir, file)
        
        yaml_dict = None
        with open(file_path) as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        #print(type(yaml_dict), "----------------------")
        #verified = yaml_dict["state"]["verified"]
        #warnings = yaml_dict["state"]["warnings"]

        width = yaml_dict["size"]["width"]
        height = yaml_dict["size"]["height"]

        bounding_boxes = []
        if "objects" in yaml_dict:
            #if len(yaml_dict["objects"]) > 1:
            #print(file_path)
            for obj in yaml_dict["objects"]:
                name = obj["name"]
                xmin = obj["bndbox"]["xmin"]
                ymin = obj["bndbox"]["ymin"]
                xmax = obj["bndbox"]["xmax"]
                ymax = obj["bndbox"]["ymax"]

                assert xmin < xmax
                assert ymin < ymax

                bounding_boxes.append([0, xmin, ymin, xmax, ymax])

        if len(bounding_boxes) > 0:
            x = np.array(bounding_boxes)
            plain_annotation = np.empty(x.shape)
            #print(plain_annotation, "---------------------")
            plain_annotation[:, 0] = x[:, 0]
            plain_annotation[:, 1::] = xyxy2xywhn(x[:, 1::], width, height)
            #print(plain_annotation)

            labels_dst_path = file_path.replace('yaml', "txt")
            np.savetxt(labels_dst_path, plain_annotation, fmt="%i %1.4f %1.4f %1.4f %1.4f")
        else:
            open(file_path.replace('yaml','txt'), 'a').close()
            if onces:
                print(file_path)
                onces = False
