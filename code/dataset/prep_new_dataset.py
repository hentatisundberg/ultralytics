import os 
import numpy as np 
import sys
from pathlib import Path
import yaml


# First convert yaml to txt 
# Then split data set, create new folders and thus prepare for upload 

# In folder 
input_folder = sys.argv[1]


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
    
    files = list(Path(input_folder).glob('*.yaml'))

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

                    assert xmin < xmax
                    assert ymin < ymax

                    bounding_boxes.append([0, xmin, ymin, xmax, ymax])

        if len(bounding_boxes) > 0:
            x = np.array(bounding_boxes)
            plain_annotation = np.empty(x.shape)
            plain_annotation[:, 0] = x[:, 0]
            plain_annotation[:, 1::] = xyxy2xywhn(x[:, 1::], width, height)

            labels_dst_path = Path(input_folder+filename.stem+".txt")
            np.savetxt(labels_dst_path, plain_annotation, fmt="%i %1.4f %1.4f %1.4f %1.4f")
            #np.savetxt(labels_dst_path, plain_annotation)
        else:   # If there are no bounding boxes in the file... 
            labels_dst_path = Path(input_folder+filename.stem+".txt")
            open(labels_dst_path, "w")


def split_dataset():
    files = list(Path(input_folder).glob('*.txt'))
    tot = len(files)
    split_val_test = [.1, .1]
    nval, ntest = int(round(split_val_test[0]*tot,0)), int(round(split_val_test[1]*tot,0))
    ntrain = tot - (nval+ntest)
    rand = [list(np.repeat(0, ntrain)), list(np.repeat(1, nval)), list(np.repeat(2, ntest))]
    rand = [item for sublist in rand for item in sublist]
    np.random.shuffle(rand)
    count = 0

    folds = ["train/", "val/", "test/"]
    
    for fold in folds: 
        os.mkdir(input_folder+fold)
        os.mkdir(input_folder+fold+"images/")
        os.mkdir(input_folder+fold+"labels/")

    for item in files: 
        label = item
        image = item.parent.joinpath(item.stem+".jpg")

        if rand[count] == 0: fold = input_folder+"train/"
        elif rand[count] == 1: fold = input_folder+"val/"
        elif rand[count] == 2: fold = input_folder+"test/"

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
#python3 dataset/prep_new_dataset.py "../../../../../Desktop/images/annot_finished/" 


