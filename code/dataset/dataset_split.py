import os
import sys
import time
import shutil
import argparse
from sklearn.model_selection import train_test_split

def doArgs(argList, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument('--image_dirs', nargs="+", help="one parent directory name with all subdir of images or all subdirectory name", default=True)
    parser.add_argument('--train_dir', type=str, help="path to train dir", required=False, default="./images/train")
    parser.add_argument('--val_dir', type=str, help="path to val dir", required=False, default="./images/val")

    return parser.parse_args(argList)

def dataset_split(image_dirs, train_dir, val_dir):
    
    startTime = float(time.time())

    # print(image_dirs)

    # if len(image_dirs) == 1:
    #     if not os.path.isdir(image_dirs[0]):
    #         print("Input doesn't exist, exiting")
    #         return
    #     image_dirs = [os.path.join(image_dirs[0], dir) for dir in os.listdir(image_dirs[0])]

    print(image_dirs)

    file_list = []
    for dir in image_dirs:
        print(dir+'.........')
        file_list = file_list + [os.path.join(dir,fullname) for fullname in os.listdir(dir)]

    print(len(file_list))
    file_idx = range(len(file_list))
    train, val, _, _ = train_test_split(file_idx, file_idx, test_size=0.2, random_state=52)
    #train
    for idx in train:
        if '.jpg' in file_list[idx] or '.png' in file_list[idx]:
            yaml_file = file_list[idx][:-4] + '.txt'
            if os.path.isfile(yaml_file.replace('images','labels')):
                shutil.copy(file_list[idx], train_dir) #copying images
                shutil.copy(yaml_file.replace('images','labels'), train_dir.replace('images','labels')) #copying labels
            else:
                print("No yaml file:",yaml_file.replace('images','labels'))
                os.mknod(yaml_file.replace('images','labels'))
                shutil.copy(file_list[idx], train_dir) #copying images
                shutil.copy(yaml_file.replace('images','labels'), train_dir.replace('images','labels')) #copying labels
    
    #test
    for idx in val:
        if '.jpg' in file_list[idx] or '.png' in file_list[idx]:
            yaml_file = file_list[idx][:-4] + '.txt'
            if os.path.isfile(yaml_file.replace('images','labels')):
                shutil.copy(file_list[idx], val_dir) #copying images
                shutil.copy(yaml_file.replace('images','labels'), val_dir.replace('images','labels')) #copying labels
            else:
                print("No yaml file:",yaml_file.replace('images','labels'))
                os.mknod(yaml_file.replace('images','labels'))
                shutil.copy(file_list[idx], val_dir) #copying images
                shutil.copy(yaml_file.replace('images','labels'), val_dir.replace('images','labels')) #copying labels

    print("Finished in %0.4f seconds" % (time.time() - startTime))
    return

if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    args = doArgs(sys.argv[1:], "dataset_split")
    image_dirs = args.image_dirs
    train_dir = args.train_dir
    val_dir = args.val_dir
    dataset_split(image_dirs, train_dir, val_dir)
