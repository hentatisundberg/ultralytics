import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import shutil
import cv2
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sqlite3
import numpy as np
import sys
sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/")
from functions import cut_vid, create_connection, dr_from_db

# Functions


# Make one function of these two! 


def annotate_vid(vidpath, model):

    # Pick out relevant video information and saves video and frames
    results = model(vidpath, 
                    conf = 0.1,
                    stream=True,  
                    save = True,
                    show = False, 
                    save_frames = False)
    for result in results: 
        result.boxes

def compress_vid(inputpath, outputdir):

    file = Path(inputpath)
    name = file.name
    output = outputdir+name
    #print(output)

    cap = cv2.VideoCapture(file)

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



def cleanup(folder, vid):
    #shutil.rmtree(f"runs/detect/{folder}/{Path(vid).stem}_frames") # Remove frames
    #os.remove(f"runs/detect/{folder}/{Path(vid).stem}.avi") # Remove video from Larus 
    shutil.rmtree(f"runs/detect/{folder}") # Remove folder
    os.remove(f"../../../../../mnt/BSP_NAS2_work/fish_model/t01/{file}") # Remove video from Larus 



def main(ledge, minframes, maxframes):

    df = df_from_db("inference/Inference_stats_merge.db", ledge, minframes, maxframes)
    dfx = df[df["track"] == "FAR3_2022-06-20_04-0011"]

    folder = "predict2"

    for row in df.index:
        input = dfx.iloc[row]
        print(f'starting with {input.track}')
        vid = cut_vid(input, "../../../../../mnt/BSP_NAS2/Video/", "../../../../../mnt/BSP_NAS2_work/fish_model/t1/") 
        print("cut finished")
        annotate_vid(vid, YOLO("../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train57.pt"))
        print("annotation finished")
        if os.path.isfile(f"runs/detect/{folder}/{Path(vid).stem}.avi"):
            compress_vid(f"runs/detect/{folder}/{Path(vid).stem}.avi", "../../../../../../mnt/BSP_NAS2_work/fish_model/t2/")
            print("compression finished")
        #cleanup(folder, vid)
        print("cleanup finished")



#df = df_from_db("inference/Inference_stats_nomerge.db", "FAR3", 2, 20)

# RUN 
#main("FAR3", 2, 20, 0)
main(sys.argv[1], sys.argv[2], sys.argv[3])




vid = cut_vid(dfx, "../../../../../mnt/BSP_NAS2/Video/", "../../../../../mnt/BSP_NAS2_work/fish_model/t1/") 