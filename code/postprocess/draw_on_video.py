
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
from functions import create_connection, df_from_db, cut_vid



def compress_annotate_vid(file, savepath):
    
    name = file.name
    
    if name[0] != ".":
        track = file.stem
        output = savepath+name

        cap = cv2.VideoCapture(str(file))

        if not cap.isOpened():
            print("Error: Could not open the input video file")
            exit()
        # XVID better than MJPG. DIVX = XVID
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change this to your desired codec
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        font = cv2.FONT_HERSHEY_SIMPLEX 

        out = cv2.VideoWriter(output, fourcc, frame_rate, frame_size, isColor=True)
        plotdata = df_raw[df_raw["track"] == track][["track", "x", "y", "width", "height"]].reset_index()

        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                
                # Filename
                if count > (len(plotdata)-5):
                    count = 0
                #print(count)
                cv2.putText(frame, f'{name}',  
                    (50, 150),  
                    font, 3,  
                    (255, 255, 255),  
                    3,  
                    cv2.LINE_4) 
                
                #for row in range(len(plotdata)-1):
                    #print(row)
                    #print(row+1)
                x1 = int(plotdata.iloc[count]["x"]+(.5*plotdata.iloc[count]["width"]))
                y1 = int(plotdata.iloc[count]["y"]+(.5*plotdata.iloc[count]["height"]))                   
                x2 = int(plotdata.iloc[(count+1)]["x"]+(.5*plotdata.iloc[(count+1)]["width"]))
                y2 = int(plotdata.iloc[(count+1)]["y"]+(.5*plotdata.iloc[(count+1)]["height"]))                   
                
                startpoint = (x1, y1)
                endpoint = (x2, y2)
                #print(startpoint)
        #        print(endpoint)
                frame = cv2.circle(frame, (x1, y1), 100, (255, 255, 255), 1)
                #frame = cv2.line(frame, startpoint, endpoint, (255, 255, 255), 20)                   

                out.write(frame)
                count += 1
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()




# Read databases  
#df_raw = df_from_db("inference/Inference_raw_nomergeV2.db", f'ledge == "FAR3"', f'strftime("%Y-%m-%d", time2) != "XYZ"', False)
#df_stats = df_from_db("inference/Inference_stats_nomergeV2.db", f'nframes > 0', f'strftime("%Y-%m-%d", start) != "XYZ"', True)


# Test


# Cut vid
#for row in df_stats.index:
#    input = df_stats.iloc[row]
#    print(f'starting with {input.track}')
#    vid = cut_vid(input, "../../../../../../Volumes/JHS-SSD2/full_vid/", "../../../../../../Volumes/JHS-SSD2/cut_vid/", "track") 
#    print("cut finished")
    
# Compress and annotate
#allfiles = list(Path("../../../../../../Volumes/JHS-SSD2/cut_vid/").glob("*.mp4"))
#for file in allfiles[200:]:
#    print(f'processing {file} ...')
#    compress_annotate_vid(file, "../../../../../../Volumes/JHS-SSD2/annot_merge/")
#    #print("compression and annotation finished")


#track = file.stem
#pp = df_raw[df_raw["track"] == track][["track", "x", "y", "width", "height"]].reset_index()
#pp.iloc[0]["x"]

# Define input path
#path = Path("../../../../../../Volumes/JHS-SSD2/clips_unmerged/clips_annot5/")
#files = list(path.glob("*.avi"))

##for file in files: 
# #   print(f'processing file {file.name}')
# #   if file.name[0] != ".":
# #       compress_annotate_vid(file,
# #           "../../../../../../Volumes/JHS-SSD2/clips_unmerged/clips_text/")

