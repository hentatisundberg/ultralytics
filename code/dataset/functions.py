
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
import sys
import numpy as np
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO


def cut_vid(video_meta, vid_sourcefold, vid_outfold): 

    video_meta = pd.read_csv(video_meta)
    for ind in video_meta.index:

        viddat = video_meta.iloc[ind]

        if viddat["Done"] != "x":

            x, y, ledge_name, video_date, start_hour = viddat[0].split('_')
            year = video_date.split('-')[0]
            start_hour = int(start_hour[:2])

            startclip = pd.to_datetime(video_date+" "+viddat[1])
            endclip = pd.to_datetime(video_date+" "+viddat[2])
            startvid = startclip.floor("H")

            startsec = (startclip-startvid)/np.timedelta64(1,'s')
            endsec = (endclip-startvid)/np.timedelta64(1,'s')

            vid_rel_path = os.path.join(vid_sourcefold, 'Video'+year, ledge_name, video_date)
            filename_out = vid_outfold+viddat[0][:-4]+"_"+str(int(startsec))+"_"+str(int(endsec))+".mp4"

            ffmpeg_extract_subclip(
                os.path.join(vid_rel_path, viddat[0]),
                startsec,
                endsec,
                targetname = filename_out
            )


def save_frames(file, image_savefold, interval):
    cap = cv2.VideoCapture(str(file))

    if not cap.isOpened():
        return

    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    digit = len(str(int(nframes)))

    saveframes = list(range(1, int(nframes), int(interval)))
    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            if n in saveframes:
                cv2.imwrite('{}_{}.{}'.format(image_savefold.joinpath(file.stem), str(n).zfill(digit), "jpg"), frame)
            n += 1
        else:
            return


def save_custom_frames(file, image_savefold):
    cap = cv2.VideoCapture(str(file))

    if not cap.isOpened():
        return

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret & n == 0:
            cv2.imwrite('{}_{}.{}'.format(image_savefold.joinpath(file.stem), str(n).zfill(digit), "jpg"), frame)
            n += 1
        else:
            return

def annotate_images():

    # Load a pretrained YOLO model
    model = YOLO(yolo_model)

    # List of videos for inference 
    ims = os.listdir(im_outfold)

    # Run
    for im in ims: 

        if len(im) > 20:

            results = model(f'{im_outfold}/{im}')

            # Width and height
            imread = cv2.imread(f'{im_outfold}/{im}')
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



def write_yaml_to_file(py_obj,filename_simpl):
    with open(f'{yaml_outfold}{filename_simpl}.yaml', 'w',) as f :
        yaml.dump(py_obj,f,sort_keys=False) 


