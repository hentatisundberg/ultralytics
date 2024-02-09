import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import shutil
import cv2
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sqlite3
import numpy as np

# Functions

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


def cut_vid(row, vidpath, savepath): 

    file = row

    #print(ind)
    datefold = str(file["start"])[0:10]
    ledge = file["Ledge"]
    yr = str(file["start"])[0:4]

    starttime = str(file["file"][:-4]).split("_")[3].replace(".", ":")
    startclip = file["start"]
    endclip = file["end"]

    if any(pd.isnull([startclip, endclip, starttime])):
        print("skip")

    else: 
        starttimestamp = pd.to_datetime(datefold+" "+starttime)

        startsec = (file["start"]-starttimestamp)/np.timedelta64(1,'s')
        endsec = (file["end"]-starttimestamp)/np.timedelta64(1,'s')

        vid_rel_path = f"{vidpath}Video{yr}/{ledge}/{datefold}/"
        #vid_rel_path = f"{vidpath}/{datefold}/"
        full_path = vid_rel_path+file["file"]

        if os.path.isfile(full_path):

            filename_out = f"{savepath}{file['track']}.mp4"
            ffmpeg_extract_subclip(
                full_path,
                startsec,
                endsec,
                targetname = filename_out
            )
            print(filename_out)
            return(filename_out)
    

def annotate_vid(vid, model):

    # Pick out relevant video information
    results = model(vid, 
                    conf = 0.1,
                    stream=True,  
                    save = True,
                    show = False, 
                    save_frames = False)
    for result in results: 
        result.boxes

def compress_vid(input, outputdir):

    file = Path(input)
    name = file.name
    output = outputdir+name
    print(output)

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

            #cv2.imshow('frame',frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def cleanup(folder):
    shutil.rmtree(f"runs/detect/{folder}/{Path(vid).stem}_frames") # Remove frames
    os.remove(f"runs/detect/{folder}/{Path(vid).stem}.avi") # Remove video from Larus 
    shutil.rmtree(f"runs/detect/{folder}") # Remove folder
    os.remove(f"../../../../../mnt/BSP_NAS2_work/fish_model/t0/{Path(vid).name}") # Remove video from Larus 


def df_from_db(db, scale, minframes, maxframes):
    # Create connection
    con = create_connection(db)

    cond1 = f'Ledge == "{scale}"'
    cond2 = f'nframes > {minframes}'
    cond3 = f'nframes < {maxframes}'

    sql = (f'SELECT * '
           f'FROM Inference '
           f'WHERE {cond1} AND {cond2} AND {cond3};')

    df = pd.read_sql_query(
        sql,
        con, 
        parse_dates = {"start": "%Y-%m-%d %H:%M:%S.%f", "end": "%Y-%m-%d %H:%M:%S.%f"}
        )
    
    return(df)


def main(ledge, minframes, maxframes):

    df = df_from_db("inference/Inference_stats_nomerge.db", ledge, minframes, maxframes)
    folder = "predict2"

    for row in df.index:
        input = df.iloc[row]
        vid = cut_vid(input, "../../../../../mnt/BSP_NAS2/Video/", "../../../../../mnt/BSP_NAS2_work/fish_model/t0/")
        annotate_vid(vid, YOLO("../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train57.pt"))
        compress_vid(f"runs/detect/{folder}/{Path(vid).stem}.avi", "../../../../../../mnt/BSP_NAS2_work/fish_model/clips_annot4/")
        cleanup(folder)
        print("Unknown error")


# RUN 
main("FAR3", 0, 20)
