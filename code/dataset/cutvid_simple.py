
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
import sys
import numpy as np
import cv2
from pathlib import Path

# Input arguments (paths)
video_meta_path = "data/extract_random.csv"
vid_sourcefold = "../../../../Downloads/"
vid_outfold = "vids/"

# Read metadata on interesting videos
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

            #vid_rel_path = os.path.join(vid_sourcefold, 'Video'+year, ledge_name, video_date)
            vid_rel_path = vid_sourcefold 
            filename_out = vid_outfold+viddat[0][:-4]+"_"+str(int(startsec))+"_"+str(int(endsec))+viddat["Why"]+".mp4"

            ffmpeg_extract_subclip(
                os.path.join(vid_rel_path, viddat[0]),
                startsec,
                endsec,
                targetname = filename_out
            )


results = cut_vid() 
