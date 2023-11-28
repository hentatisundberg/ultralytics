
import numpy as np
from pathlib import Path

# import shutil
import sys
import os
import yaml

# Where to get and save files
sourcefold = "data/valid/labels/"
savefold = "data/annotations/yaml/"

def write_yaml_to_file(py_obj,filename):
    with open(f'{savefold}{filename}.yaml', 'w',) as f :
        yaml.dump(py_obj,f,sort_keys=False) 
    #print('Written to file successfully')

# Loop through dir
files = os.listdir(sourcefold)

for filename in files: 

    pred = np.loadtxt(Path(sourcefold+filename))

    width = 2560
    height = 1440

    # Always in file
    data_dict = {}
    data_dict["image"] = filename
    data_dict["size"] = {"depth": 3, "height": height, "width": width}
    data_dict["source"] = {"framenumber": 0, "path": "na", "video": "na"}
    data_dict["state"] = {"verified": False, "warnings": 0}

    if len(pred) > 0:
        data_dict["objects"] = []

        for ind in range(0, pred.ndim):
            if pred.ndim == 1: 
                tdat = pred
            else:
                tdat = pred[ind]
            data_dict["objects"].append(
                {
                    "bndbox": {
                        "xmax": int(np.clip((tdat[1] + tdat[3] * 0.5) * width, 0, width - 1)),
                        "xmin": int(np.clip((tdat[1] - tdat[3] * 0.5) * width, 0, width - 1)),
                        "ymax": int(np.clip((tdat[2] + tdat[4] * 0.5) * height, 0, height - 1)),
                        "ymin": int(np.clip((tdat[2] - tdat[4] * 0.5) * height, 0, height - 1)),
                    },
                    "name": "fish",

                }
            )

    write_yaml_to_file(data_dict, filename[:-4])

