#!/usr/bin/env python

import argparse
from dataclasses import astuple
import numpy as np
from pathlib import Path
from PIL import Image

# import shutil
import sys
import yaml

# setting module path
import config
import bounding_box_mappings



sourcefold = sys.argv[0]
savefold = sys.argv[1]


w, h = 0, 0
filename = Path("dataset/txt/Auklab1_TRI3_2023-07-03_03.00.00_1795_1815_307.txt")
pred = np.loadtxt(filename)
if pred.ndim == 1:
    pred = np.expand_dims(pred, axis=0)

data_dict = {}
data_dict["image"] = filename
data_dict["size"] = {"depth": 3, "height": h, "width": w}
data_dict["source"] = {"framenumber": 0, "path": "na", "video": "na"}
data_dict["state"] = {"verified": False, "warnings": 0}

if pred.shape[0] > 0:
    data_dict["objects"] = []
    labels_xyxy = bounding_box_mappings.xywhn2xyxy(pred[:, 1:], w, h)
    for i, pi in enumerate(pred):
        type_idx = int(pi[0])
        assert type_idx >= 0 and type_idx < 1
        data_dict["objects"].append(
            {
                "name": "fish",
                "bndbox": {
                    "xmax": int(labels_xyxy[i, 2]),
                    "xmin": int(labels_xyxy[i, 0]),
                    "ymax": int(labels_xyxy[i, 3]),
                    "ymin": int(labels_xyxy[i, 1]),
                },
            }
        )

#file = open(dst_path.joinpath(plain_path.stem + ".yaml"), "w")
#print(file)
#yaml.dump(data_dict, file, allow_unicode=True)#



#def main() -> int:
#    args = parse_args()
#    src_root_path = args.src_root_path
#    for plain_path in src_root_path.glob("**/*.txt"):
#        dst_path = args.dst_root_path.joinpath(
#            plain_path.relative_to(src_root_path).parent
#        )
#        print(dst_path)
#        if not dst_path.exists():
#            dst_path.mkdir(parents=True, exist_ok=False)###
#
#        data_dict = plain2yaml2(
#            plain_path=plain_path,
#            images_path=images_path,
#            types=args.types,
#        )
#
#        file = open(dst_path.joinpath(plain_path.stem + ".yaml"), "w")
#        print(file)
#        yaml.dump(data_dict, file, allow_unicode=True)#
#
#    return 0


#if __name__ == "__main__":
#    sys.exit(main())
