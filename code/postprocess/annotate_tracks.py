
import pandas as pd
from pathlib import Path 
import shutil
import cv2
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sqlite3
import numpy as np
import sys
from functions import insert_to_db, create_connection, df_from_db



# Functions

def cut_vid(row, vidpath, savepath): 

    file = row

    datefold = str(file["start"])[0:10]

    starttime = str(file["file"][:-4]).split("_")[3].replace(".", ":")
    startclip = file["start"]
    endclip = file["end"]

    if any(pd.isnull([startclip, endclip, starttime])):
        print("skip")

    else: 
        starttimestamp = pd.to_datetime(datefold+" "+starttime)

        startsec = (file["start"]-starttimestamp)/np.timedelta64(1,'s')
        endsec = (file["end"]-starttimestamp)/np.timedelta64(1,'s')

        vid_rel_path = f"{vidpath}/{datefold}/"
        full_path = vid_rel_path+file["file"]

        if os.path.isfile(full_path):

            filename_out = f"{savepath}{file['track']}.mp4"
            ffmpeg_extract_subclip(
                full_path,
                startsec,
                endsec,
                targetname = filename_out
            )
            #print(filename_out)
            return(filename_out)
  

def compress_vid(input, outputdir):
    
    file = Path(input)
    name = file.name
    output = outputdir+name
    #print(output)

    cap = cv2.VideoCapture(input)

    if not cap.isOpened():
        print("Error: Could not open the input video file.")
        exit()
    # XVID better than MJPG. DIVX = XVID
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Change this to your desired codec
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output, fourcc, frame_rate, frame_size, isColor=True)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main(ledge, minframes, maxframes):
    
    df = df_from_db("inference/Inference_stats_merge1.db", ledge, minframes, maxframes)

    for row in df.index:
        input = df.iloc[row]
        print(f'starting with {input.file}')
        vid = cut_vid(input, "../../../../../../Volumes/JHS-SSD2/full_vid/", "../../../../../../Volumes/JHS-SSD2/cut_vid/") 
        print("cut finished")
        compress_vid("../../../../../../Volumes/JHS-SSD2/cut_vid/", "../../../../../../Volumes/JHS-SSD2/annot_merge/")
        print("compression and annotation finished")


#df = df_from_db("inference/Inference_stats_nomerge.db", "FAR3", 2, 20)

# RUN 
#main("FAR3", 2, 20, 0)
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])



