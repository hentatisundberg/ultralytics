#from ydata_profiling import ProfileReport
import sqlite3
import pandas as pd
import numpy as np
import os
#from dataprep.eda import create_report
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from functions import create_connection


def cut_vid(data_frame, vidpath, savepath): 

    track_id_processed = []

    for ind in data_frame.index:

        if ind > 0:
            file = data_frame.iloc[ind]

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
                print(full_path)

                if os.path.isfile(full_path):

                    filename_out = f"{savepath}{file['track_id']}.mp4"
                    ffmpeg_extract_subclip(
                        full_path,
                        startsec,
                        endsec,
                        targetname = filename_out
                    )

                    track_id_processed.append(file["track_id"])
    return(track_id_processed)



# Create connection
con = create_connection("inference/Inference_stats.db")

# Load data 
sql = """
    SELECT * FROM Inference
    """

df = pd.read_sql_query(
    sql,
    con, 
    parse_dates = {"start": "%Y-%m-%d %H:%M:%S.%f", "end": "%Y-%m-%d %H:%M:%S.%f"}
    )


# Y-data report
#profile = ProfileReport(df, title="Fish tracks")
#profile.to_file("inference/inference_ydata.html")

# Dataprep report
#report = create_report(df, title='Fish tracks')
#report.save(filename='fish_tracks_dataprep', to='/inference')

# Cut videos
done = cut_vid(df, "../../../../../mnt/BSP_NAS2/Video/", "../../../../../mnt/BSP_NAS2_work/fish_model/clips2/")
#done = cut_vid(df, "../../../../../../Volumes/JHS-SSD2", "../../../../../../Volumes/JHS-SSD2/clips/")

