#from ydata_profiling import ProfileReport
import sqlite3
import pandas as pd
import numpy as np
import os
#from dataprep.eda import create_report
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

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


def cut_vid(input_data): 

    track_id_processed = []

    for ind in input_data.index:

        file = input_data.iloc[ind]

        datefold = str(file["start"])[0:10]
        starttime = str(file["file"]).split("_")[3].replace(".", ":")

        starttimestamp = pd.to_datetime(datefold+" "+starttime)

        startsec = (file["start"]-starttimestamp)/np.timedelta64(1,'s')
        endsec = (file["end"]-starttimestamp)/np.timedelta64(1,'s')
      
        vid_rel_path = f"/Volumes/JHS-SSD2/{datefold}/"
        full_path = vid_rel_path+"/"+file["file"]+".mp4"

        if os.path.isfile(full_path):

            filename_out = "/Volumes/JHS-SSD2/clips1/"+file["track_id"]+".mp4"

            #ffmpeg_extract_subclip(
            #    full_path,
            #    startsec,
            #    endsec,
            #    targetname = filename_out
            #)

            track_id_processed.append(file["track_id"])
    return(track_id_processed)



# Create connection
con = create_connection("inference/Inference.db")

# Load data 
sql = """
    SELECT * FROM Inference 
    """

df = pd.read_sql_query(
    sql,
    con, 
    parse_dates = {"start": "%Y-%m-%d %H:%M:%S", "end": "%Y-%m-%d %H:%M:%S"}
    )


# Y-data report
#profile = ProfileReport(df, title="Fish tracks")
#profile.to_file("inference/inference_ydata.html")

# Dataprep report
#report = create_report(df, title='Fish tracks')
#report.save(filename='fish_tracks_dataprep', to='/inference')

# Cut videos
#done = cut_vid(df)






# OLD
#done1 = pd.Series(done)
#done_df = pd.DataFrame(done1)
#done_df["done"] = 1
#done_df.columns = ["track_id", "done"]
#df2 = pd.merge(df, done_df, on = "track_id", how = "left")
#f2.to_excel("inference/tracks_annotate.xlsx")
