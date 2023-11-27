import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
import sys
import numpy as np


video_meta = pd.read_csv("../data/fishvids.csv")


#video_metadata = [
#    ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:02:14', '04:03:29'],
#    ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:07:12', '04:10:55'],
##    ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:11:32', '04:12:02'],
#   ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:32:54', '04:33:10'],
#   ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:54:00', '04:54:15'],
#   ['Auklab1_BONDEN6_2022-07-02_05.00.00.mp4', '05:44:23', '05:44:40'],
#   ['Auklab1_ROST6_2022-07-02_04.00.00.mp4', '04:09:44', '04:10:26'],
#   ['Auklab1_ROST6_2022-07-02_05.00.00.mp4', '05:03:57', '05:04:06'],
#   ['Auklab1_ROST6_2022-07-02_05.00.00.mp4', '05:26:25', '05:28:50'],
#   ['Auklab1_FAR6_2022-07-02_04.00.00.mp4', '04:46:44', '04:58:58'],
#   ['Auklab1_FAR6_2022-07-02_03.00.00.mp4', '03:44:24', '03:46:17'],
#   ['Auklab1_ROST3_2022-07-02_04.00.00.mp4', '04:10:25', '04:10:37'],
#   ['Auklab1_ROST3_2022-07-02_04.00.00.mp4', '04:57:33', '04:58:47'],
#   ['Auklab1_ROST3_2022-07-02_03.00.00.mp4', '03:35:48', '03:36:03'],
#   ['Auklab1_TRI3_2022-07-02_03.00.00.mp4', '03:36:30', '03:36:40'],
#   ['Auklab1_TRI6_2022-07-02_03.00.00.mp4', '03:00:00', '03:06:15'],
#]

nas_video_path = "../../../../../../mnt/BSP_NAS1/Video/Video2023/TRI3/2023-07-03/"
ext_video_path = "../../../../../../../../Volumes/JHS-SSD2/2023-07-03"

#print(os.listdir(testdir))
#sys.exit()

output_folder = '../vids/'

#if not os.path.isdir(output_folder):
#    os.mkdir(output_folder)


viddat = video_meta.iloc[0]

non, ledge_name, video_date, start_hour = viddat[0].split('_')
year = video_date.split('-')[0]
start_hour = int(start_hour[:2])

startclip = pd.to_datetime(video_date+" "+viddat[1])
endclip = pd.to_datetime(video_date+" "+viddat[2])
startvid = startclip.floor("H")

startsec = (startclip-startvid)/np.timedelta64(1,'s')
endsec = (endclip-startvid)/np.timedelta64(1,'s')

filename_out = output_folder+viddat[0][:-4]+"_"+str(int(startsec))+"_"+str(int(endsec))+".mp4"
#filename_out = viddat[0][:-4]+"_"+str(int(startsec))+"_"+str(int(endsec))+".mp4"

ffmpeg_extract_subclip(
#    os.path.join(nas_video_path, 'Video'+year, ledge_name, video_date, metadata[0]),
    os.path.join(ext_video_path, viddat[0]),
    startsec,
    endsec,
    targetname = filename_out
)
