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

_seabird_types = astuple(config.SeabirdTypes())


def parse_args():
    argparser = argparse.ArgumentParser(
        description="A script to convert plain labels to yaml labels."
    )
    argparser.add_argument(
        "src_root_path",
        type=Path,
        help="Source root path.",
    )
    argparser.add_argument(
        "--dst_root_path",
        type=Path,
        default=Path.cwd().joinpath("output"),
        help="Destination, output, root path.",
    )
    argparser.add_argument(
        "--types",
        default=astuple(config.Fish()),
        choices=_seabird_types,
        nargs="+",
        type=str,
        help="Seabird types.",
    )

    args = argparser.parse_args()

    return args


def plain2yaml(
    plain_path: Path,
    images_path: Path,
    types: tuple,
):
    img = Image.open(images_path)
    w, h = img.size

    pred = np.loadtxt(plain_path)
    if pred.ndim == 1:
        pred = np.expand_dims(pred, axis=0)

    data_dict = {}
    try:
        data_dict["image"] = images_path.name
        data_dict["size"] = {"depth": 3, "height": h, "width": w}
        data_dict["source"] = {"framenumber": 0, "path": "na", "video": "na"}
        data_dict["state"] = {"verified": False, "warnings": 0}

        if pred.shape[0] > 0:
            data_dict["objects"] = []
            labels_xyxy = boundig_box_mappings.xywhn2xyxy(pred[:, 1:], w, h)
            for i, pi in enumerate(pred):
                type_idx = int(pi[0])
                assert type_idx >= 0 and type_idx < len(types)
                data_dict["objects"].append(
                    {
                        "name": types[type_idx],
                        "bndbox": {
                            "xmax": int(labels_xyxy[i, 2]),
                            "xmin": int(labels_xyxy[i, 0]),
                            "ymax": int(labels_xyxy[i, 3]),
                            "ymin": int(labels_xyxy[i, 1]),
                        },
                    }
                )
    except ValueError as e:
        sys.exit(f"ValueError: {e}")
    except KeyError as e:
        sys.exit(f"KeyError: {e}")

    return data_dict


def main() -> int:
    args = parse_args()
    try:
        src_root_path = args.src_root_path
        for plain_path in src_root_path.glob("**/*.txt"):
            images_path = plain_path.parent.joinpath(plain_path.stem + ".png")
            if images_path.exists():
                dst_path = args.dst_root_path.joinpath(
                    plain_path.relative_to(src_root_path).parent
                )
                if not dst_path.exists():
                    dst_path.mkdir(parents=True, exist_ok=False)

                data_dict = plain2yaml(
                    plain_path=plain_path,
                    images_path=images_path,
                    types=args.types,
                )

                file = open(dst_path.joinpath(plain_path.stem + ".yaml"), "w")
                yaml.dump(data_dict, file, allow_unicode=True)
    except yaml.YAMLError as e:
        sys.exit(f"YAMLError: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
