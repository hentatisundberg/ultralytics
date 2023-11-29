import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
import sys
import numpy as np
import cv2
from autodistill_yolov8 import YOLOv8Base
from autodistill.detection import CaptionOntology


# Read arguments
video_meta_path = sys.argv[1]
datfold = sys.argv[2]
vid_outfold = sys.argv[3]
im_outfold = sys.argv[4]


video_meta = pd.read_csv(video_meta_path)

def cut_vid(): 

    for ind in video_meta.index:

        viddat = video_meta.iloc[ind]

        if viddat["Done"] != "x":

            non, ledge_name, video_date, start_hour = viddat[0].split('_')
            year = video_date.split('-')[0]
            start_hour = int(start_hour[:2])

            startclip = pd.to_datetime(video_date+" "+viddat[1])
            endclip = pd.to_datetime(video_date+" "+viddat[2])
            startvid = startclip.floor("H")

            startsec = (startclip-startvid)/np.timedelta64(1,'s')
            endsec = (endclip-startvid)/np.timedelta64(1,'s')

            filename_out = vid_outfold+viddat[0][:-4]+"_"+str(int(startsec))+"_"+str(int(endsec))+".mp4"
            #filename_out = viddat[0][:-4]+"_"+str(int(startsec))+"_"+str(int(endsec))+".mp4"

            ffmpeg_extract_subclip(
            #    os.path.join(nas_video_path, 'Video'+year, ledge_name, video_date, metadata[0]),
                os.path.join(datfold, viddat[0]),
                startsec,
                endsec,
                targetname = filename_out
            )

def save_all_frames(ext='jpg'):
    cap = cv2.VideoCapture(os.path.join(vid_outfold+file))

    if not cap.isOpened():
        return

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(os.path.join(im_outfold+file), str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return



## RUN

# Run video cutting
results = cut_vid() 

# Extract frames from all vids 
for file in os.listdir(vid_outfold)[1:]:
    save_all_frames()

# Annotate
base_model = YOLOv8Base(ontology=CaptionOntology({"fish": "fish"}), weights_path="../models/best.pt")

base_model.label(
  input_folder=im_outfold,
  output_folder="../data"
)



#python3 -i dataset/video_extraction.py "../data/fishvids.csv" "../../../../../../../../Volumes/JHS-SSD2/2023-07-03" "../vids/" "../images/"

