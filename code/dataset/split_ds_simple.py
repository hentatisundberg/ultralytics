import os 
import numpy as np 
import sys
from pathlib import Path
import yaml


# Split dataset in train/validate/test 

# Directories 
input_dir = sys.argv[1]
output_dir = sys.argv[2]


def split_dataset():
    files = list(Path(input_dir+"/labels/").glob('*.txt'))
    tot = len(files)
    split_val_test = [.1, .1]
    nval, ntest = int(round(split_val_test[0]*tot,0)), int(round(split_val_test[1]*tot,0))
    ntrain = tot - (nval+ntest)
    rand = [list(np.repeat(0, ntrain)), list(np.repeat(1, nval)), list(np.repeat(2, ntest))]
    rand = [item for sublist in rand for item in sublist]
    np.random.shuffle(rand)
    count = 0

    folds = ["train/", "validate/", "test/"]
    
    for item in files: 
        label = item
        image = Path(input_dir+"/images").joinpath(item.stem+".jpg")

        if rand[count] == 0: fold = output_dir+"train/"
        elif rand[count] == 1: fold = output_dir+"validate/"
        elif rand[count] == 2: fold = output_dir+"test/"

        label_new = Path(fold+"labels/"+item.name)
        image_new = Path(fold+"images/"+item.stem+".jpg")
        
        try: 
            os.rename(label, label_new)
            os.rename(image, image_new)
        except: pass

        count += 1


# Run 
results = split_dataset()


#Run example
#python3 -i code/dataset/split_ds_simple.py "../../../../../mnt/BSP_NAS2_work/seabirds/annotations/seabird1" "../../../../../mnt/BSP_NAS2_work/seabirds/annotations/"



