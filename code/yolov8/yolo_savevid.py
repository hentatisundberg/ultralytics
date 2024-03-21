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
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/")

from functions import cut_vid, create_connection, df_from_db

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

def compress_vid(inputpath, outputdir, plotdata):

    dfx = plotdata
    file = Path(inputpath)
    name = file.name
    output = outputdir+name
    #print(output)

    cap = cv2.VideoCapture(inputpath)

    if not cap.isOpened():
        print("Error: Could not open the input video file.")
        exit()
    # XVID better than MJPG. DIVX = XVID
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change this to your desired codec
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output, fourcc, frame_rate, frame_size, isColor=True)
    font = cv2.FONT_HERSHEY_SIMPLEX 

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:

            cv2.putText(frame, f'{name}',  
                 (50, 150), font, 3, (255, 255, 255),  
                    3,  
                    cv2.LINE_4)
            startpoint = (int(dfx["x_first"]), int(dfx["y_first"]))
            endpoint = (int(dfx["x_last"]), int(dfx["y_last"]))
            cv2.circle(frame, (startpoint), 70, (255, 255, 255), 1)
            cv2.circle(frame, (endpoint), 70, (255, 0, 255), 1)
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
    os.remove(f"../../../../../mnt/BSP_NAS2_work/fish_model/t1/{vid}") # Remove video from Larus 



# Run video annotation
    
df = df_from_db("inference/Inference_stats_mergeFAR3.db", f'ledge == "FAR3"', f'ledge != "XXX"', True)
valid = pd.read_csv("inference/merged_fishFAR3.csv", sep = ";", decimal = ",")
dfvalid = df.merge(valid, on = "track")
dfvalid = dfvalid[dfvalid["multi"] > 0]

count = -1
for row in range(0, len(dfvalid)): 
    count += 1
    dfx = dfvalid.iloc[count]
    vid = cut_vid(dfx, "../../../../../mnt/BSP_NAS2/Video/", "../../../../../mnt/BSP_NAS2_work/fish_model/t1/", 2) 
    annotate_vid(vid, YOLO("../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train57.pt"))
    #if os.path.isfile(f"runs/detect/{folder}/{Path(vid).stem}.avi"):
    folder = "predict2"
    compress_vid(f"runs/detect/{folder}/{Path(vid).stem}.avi", "../../../../../../mnt/BSP_NAS2_work/fish_model/t2/", dfx)
    #print("compression finished")
    cleanup(folder, vid)
