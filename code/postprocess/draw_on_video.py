
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


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


def df_from_db(db, scale):
    con = create_connection(db)

    cond1 = f'Ledge == "{scale}"'

    sql = (f'SELECT * '
           f'FROM Inference '
           f'WHERE {cond1};')

    df = pd.read_sql_query(
        sql,
        con, 
        parse_dates = {"start": "%Y-%m-%d %H:%M:%S.%f", "end": "%Y-%m-%d %H:%M:%S.%f"}
        )
    
    return(df)



def compress_annotate_vid(input, outputdir):
    
    file = input
    name = file.name
    output = outputdir+name

    cap = cv2.VideoCapture(str(file))

    if not cap.isOpened():
        print("Error: Could not open the input video file.")
        exit()
    # XVID better than MJPG. DIVX = XVID
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Change this to your desired codec
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    font = cv2.FONT_HERSHEY_SIMPLEX 

    out = cv2.VideoWriter(output, fourcc, frame_rate, frame_size, isColor=True)
    plotdata = df[df["track"] == file.stem]

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            
            # Filename
            cv2.putText(frame, f'{name}',  
                (50, 150),  
                font, 3,  
                (255, 255, 255),  
                3,  
                cv2.LINE_4) 
        
            # Points 
            for row in plotdata.index:
                x = (plotdata["x"][row]+(.5*plotdata["width"][row])).astype(int)
                y = (plotdata["y"][row]+(.5*plotdata["height"][row])).astype(int)
                frame = cv2.circle(frame, (x, y), 100, (255, 255, 255), 1)

            out.write(frame)

            #cv2.imshow('frame',frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Read database
df = df_from_db("inference/Inference_raw_nomerge.db", "FAR3")

# Define input path
path = Path("../../../../../../Volumes/JHS-SSD2/clips_unmerged/clips_annot5/")
files = list(path.glob("*.avi"))

for file in files: 
    print(f'processing file {file.name}')
    if file.name[0] != ".":
        compress_annotate_vid(file,
            "../../../../../../Volumes/JHS-SSD2/clips_unmerged/clips_text/")

