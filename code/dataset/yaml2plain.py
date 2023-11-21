#!/usr/bin/env python


import argparse
from dataclasses import astuple
import numpy as np
from pathlib import Path

# import shutil
import sys
import yaml

# setting module path
import config
import boundig_box_mappings

_seabird_types = astuple(config.SeabirdTypes())


def parse_args():
    argparser = argparse.ArgumentParser(
        description="A script to convert yaml labels to plain labels."
    )
    argparser.add_argument(
        "src_root_path",
        type=Path,
        help="Source root path, images and labels.",
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


def _plain_paths(dst_path: Path):
    images_path = dst_path.joinpath(config.YolovDataDirs.IMAGES)
    labels_path = dst_path.joinpath(config.YolovDataDirs.LABELS)
    return images_path, labels_path


def yaml2plain(yaml_path: Path, types: tuple):
    plain_annotation = None
    try:
        file = open(yaml_path)
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

        verified = yaml_dict["state"]["verified"]
        warnings = yaml_dict["state"]["warnings"]

        width = yaml_dict["size"]["width"]
        height = yaml_dict["size"]["height"]

        bounding_boxes = []
        if "objects" in yaml_dict:
            for obj in yaml_dict["objects"]:
                name = obj["name"]
                if not name in types:
                    print(name, yaml_path)
                    continue
                assert name in types
                # if not name in names:
                #     print(name, yaml_file_path)
                #     # continue
                #     return plain_annotation, False, False
                # #    return
                # #    # sys.exit()

                xmin = obj["bndbox"]["xmin"]
                ymin = obj["bndbox"]["ymin"]
                xmax = obj["bndbox"]["xmax"]
                ymax = obj["bndbox"]["ymax"]

                assert xmin < xmax
                assert ymin < ymax

                try:
                    id = types.index(name)
                except ValueError as e:
                    print(f"ValueError {e}.")
                    sys.exit()

                bounding_boxes.append([id, xmin, ymin, xmax, ymax])

        if len(bounding_boxes) > 0:
            x = np.array(bounding_boxes)
            plain_annotation = np.empty(x.shape)
            plain_annotation[:, 0] = x[:, 0]
            plain_annotation[:, 1::] = boundig_box_mappings.xyxy2xywhn(
                x[:, 1::], width, height
            )

    except OSError as e:
        sys.exit(f"OSError: {e}")
    except KeyError as e:
        sys.exit(f"KeyError: {e}, parsing dict {str(yaml_path)}")
    except yaml.YAMLError as e:
        sys.exit(f"YAMLError: {e}")

    return plain_annotation, verified, warnings


def main() -> int:
    args = parse_args()
    try:
        src_root_path = args.src_root_path
        for yaml_path in src_root_path.glob("**/*.yaml"):
            dst_path = args.dst_root_path.joinpath(
                yaml_path.relative_to(src_root_path).parent
            )
            images_path, labels_path = _plain_paths(dst_path)
            # if not images_path.exists():
            #     images_path.mkdir(parents=True, exist_ok=False)
            if not labels_path.exists():
                labels_path.mkdir(parents=True, exist_ok=False)

            plain_annotation, verified, warnings = yaml2plain(
                yaml_path=yaml_path,
                types=args.types,
            )

            if plain_annotation is not None:  #  and verified and not warnings:
                labels_dst_path = labels_path.joinpath(yaml_path.stem + ".txt")
                np.savetxt(
                    labels_dst_path,
                    plain_annotation,
                    fmt="%i %1.4f %1.4f %1.4f %1.4f",
                )
    except OSError as e:
        sys.exit(f"OSError: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
