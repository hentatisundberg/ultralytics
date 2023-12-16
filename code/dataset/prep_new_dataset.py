import os 
import numpy as np 
import sys
from pathlib import Path
import yaml


# First convert yaml to txt 
# Then split data set, create new folders and thus prepare for upload 
# Files are copied (not removed) from original directory
# Labels and images may or may not be in the same folder originally


# Directories 
input_yaml = sys.argv[1]
input_images = sys.argv[2]
new_base = sys.argv[3]


def xyxy2xywhn(x: np.ndarray, width: int, height: int) -> np.ndarray:
    assert x.ndim == 2 
    assert x.shape[1] == 4
    y = np.empty(x.shape)
    y[:, 0] = np.clip((x[:, 0] + x[:, 2]) * 0.5 / width, 0, 1)  # x center
    y[:, 1] = np.clip((x[:, 1] + x[:, 3]) * 0.5 / height, 0, 1)  # y center
    y[:, 2] = np.clip((x[:, 2] - x[:, 0]) / width, 0, 1)  # width
    y[:, 3] = np.clip((x[:, 3] - x[:, 1]) / height, 0, 1)  # height
    return y


def create_plain():
    
    files = list(Path(input_yaml).glob('*.yaml'))

    for filename in files: 

        yaml_dict = None
        yaml_dict = yaml.load(open(filename), Loader=yaml.FullLoader)

        width = yaml_dict["size"]["width"]
        height = yaml_dict["size"]["height"]

        bounding_boxes = []
        if "objects" in yaml_dict:
                for obj in yaml_dict["objects"]:
                    name = obj["name"]
                    xmin = obj["bndbox"]["xmin"]
                    ymin = obj["bndbox"]["ymin"]
                    xmax = obj["bndbox"]["xmax"]
                    ymax = obj["bndbox"]["ymax"]

                    #assert xmin < xmax
                    #assert ymin < ymax

                    bounding_boxes.append([0, xmin, ymin, xmax, ymax])
        
        labels_dst_path = Path(new_base+filename.stem+".txt")

        if len(bounding_boxes) > 0:
            x = np.array(bounding_boxes)
            plain_annotation = np.empty(x.shape)
            plain_annotation[:, 0] = x[:, 0]
            plain_annotation[:, 1::] = xyxy2xywhn(x[:, 1::], width, height)
            np.savetxt(labels_dst_path, plain_annotation, fmt="%i %1.4f %1.4f %1.4f %1.4f")
        else:   # If there are no bounding boxes in the file... 
            open(labels_dst_path, "w")


def split_dataset():
    files = list(Path(new_base).glob('*.txt'))
    tot = len(files)
    split_val_test = [.1, .1]
    nval, ntest = int(round(split_val_test[0]*tot,0)), int(round(split_val_test[1]*tot,0))
    ntrain = tot - (nval+ntest)
    rand = [list(np.repeat(0, ntrain)), list(np.repeat(1, nval)), list(np.repeat(2, ntest))]
    rand = [item for sublist in rand for item in sublist]
    np.random.shuffle(rand)
    count = 0

    folds = ["train/", "validate/", "test/"]
    
    #for fold in folds: 
    #    os.mkdir(new_base+fold)
    #    os.mkdir(new_base+fold+"images/")
    #    os.mkdir(new_base+fold+"labels/")

    for item in files: 
        label = item
        image = Path(input_images).joinpath(item.stem+".jpg")

        if rand[count] == 0: fold = new_base+"train/"
        elif rand[count] == 1: fold = new_base+"validate/"
        elif rand[count] == 2: fold = new_base+"test/"

        label_new = Path(fold+"labels/"+item.name)
        image_new = Path(fold+"images/"+item.stem+".jpg")
        
        try: 
            os.rename(label, label_new)
            os.rename(image, image_new)
        except: pass

        count += 1


# Run 
results = create_plain()
results = split_dataset()


#Run example
#python3 code/dataset/prep_new_dataset.py "../../../../../mnt/BSP_NAS2_work/seabirds/annotations/seabirds1" "../../../../../mnt/BSP_NAS2_work/seabirds/annotations/seabirds1" "../../../../../mnt/BSP_NAS2_work/seabirds/annotations/"


