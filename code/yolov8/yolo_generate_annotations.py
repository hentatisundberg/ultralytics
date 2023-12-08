


from ultralytics import YOLO
import os
import sys 
import pathlib
import numpy as np
import cv2
import pandas as pd
import torch
import yaml


def write_yaml_to_file(py_obj,filename_simpl):
    with open(f'../dataset/annotations_yaml/{filename_simpl}.yaml', 'w',) as f :
        yaml.dump(py_obj,f,sort_keys=False) 


def annotate_images():

    # Load a pretrained YOLO model
    model = YOLO('../models/best.pt')

    # List of videos for inference 
    ims = os.listdir("../images")

    # Run
    for im in ims: 

        if len(im) > 20:

            results = model(f'../images/{im}')

            # Width and height
            imread = cv2.imread(f'../images/{im}')
            width = imread.shape[1]
            height = imread.shape[0]

            # Process results list
            boxes = []
            boxesxyxy = []

            for r in results:
                boxes.append(r.boxes.xywh.tolist()) 
                boxesxyxy.append(r.boxes.xyxy.tolist())

            # Concatenate outputs
            boxesx = sum(boxes, [])
            boxesxyxy2 = sum(boxesxyxy, [])

            # Save as data frames
            nobj = len(boxesx)

            filename = im.replace(".jpg", ".txt")
            filename_simpl = im.replace(".jpg", "")


            # .yaml
            # Always in file
            data_dict = {}
            data_dict["image"] = filename
            data_dict["size"] = {"depth": 3, "height": height, "width": width}
            data_dict["source"] = {"framenumber": 0, "path": "na", "video": "na"}
            data_dict["state"] = {"verified": False, "warnings": 0}


            if nobj > 0:

                data_dict["objects"] = []

                for row in range(0, nobj):
                    tdat = boxesxyxy2[row]
                    
                    data_dict["objects"].append(
                        {
                            "bndbox": {
                                "xmax": tdat[2],
                                "xmin": tdat[0],
                                "ymax": tdat[3],
                                "ymin": tdat[1],
                            },
                            "name": "fish",

                        }
                    )
            
            write_yaml_to_file(data_dict, filename_simpl)


            # Plain annotation
            #if nobj > 0:

                #y = np.empty([nobj, 5], dtype = float)
                #for row in range(0, nobj):
                #    y[row, 1] = (boxesx[row][0]+(.5*boxesx[row][2]))/width # x 
                #    y[row, 2] = (boxesx[row][1]+(.5*boxesx[row][3]))/height # y 
                #    y[row, 3] = (boxesx[row][2])/width # w 
                #    y[row, 4] = (boxesx[row][3])/height # h 
                
                #np.savetxt(f'../dataset/annotations/{filename}', y, fmt="%i %1.4f %1.4f %1.4f %1.4f")
            
            #else:
             #   open(f'../dataset/annotations/{filename}', 'a').close()

results = annotate_images()


