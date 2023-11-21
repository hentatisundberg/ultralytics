import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


video_metadata = [
    ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:02:14', '04:03:29'],
    ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:07:12', '04:10:55'],
    ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:11:32', '04:12:02'],
    ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:32:54', '04:33:10'],
    ['Auklab1_BONDEN6_2022-07-02_04.00.00.mp4', '04:54:00', '04:54:15'],
    ['Auklab1_BONDEN6_2022-07-02_05.00.00.mp4', '05:44:23', '05:44:40'],
    ['Auklab1_ROST6_2022-07-02_04.00.00.mp4', '04:09:44', '04:10:26'],
    ['Auklab1_ROST6_2022-07-02_05.00.00.mp4', '05:03:57', '05:04:06'],
    ['Auklab1_ROST6_2022-07-02_05.00.00.mp4', '05:26:25', '05:28:50'],
    ['Auklab1_FAR6_2022-07-02_04.00.00.mp4', '04:46:44', '04:58:58'],
    ['Auklab1_FAR6_2022-07-02_03.00.00.mp4', '03:44:24', '03:46:17'],
    ['Auklab1_ROST3_2022-07-02_04.00.00.mp4', '04:10:25', '04:10:37'],
    ['Auklab1_ROST3_2022-07-02_04.00.00.mp4', '04:57:33', '04:58:47'],
    ['Auklab1_ROST3_2022-07-02_03.00.00.mp4', '03:35:48', '03:36:03'],
    ['Auklab1_TRI3_2022-07-02_03.00.00.mp4', '03:36:30', '03:36:40'],
    ['Auklab1_TRI6_2022-07-02_03.00.00.mp4', '03:00:00', '03:06:15'],
]

nas_video_path = '/home/shreyash_kad/mnt/nas2/BSP_data/Video/'
output_folder = '/home/shreyash_kad/fish_detection/dataset/fish_video'

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)


for metadata in video_metadata:
    _, ledge_name, video_date, start_hour = metadata[0].split('_')
    year = video_date.split('-')[0]
    start_hour = int(start_hour[:2])
    clip_start_hour, clip_start_min, clip_start_sec = metadata[1].split(':')
    clip_start_hour, clip_start_min, clip_start_sec = int(clip_start_hour), int(clip_start_min), int(clip_start_sec)
    clip_end_hour, clip_end_min, clip_end_sec = metadata[2].split(':')
    clip_end_hour, clip_end_min, clip_end_sec = int(clip_end_hour), int(clip_end_min), int(clip_end_sec)

    start_time = ((clip_start_hour - start_hour) * 60 + clip_start_min) * 60 + clip_start_sec
    end_time = ((clip_end_hour - start_hour) * 60 + clip_end_min) * 60 + clip_end_sec

    ffmpeg_extract_subclip(
        os.path.join(nas_video_path, 'Video'+year, ledge_name, video_date, metadata[0]),
        start_time,
        end_time,
        targetname = os.path.join(output_folder, '_'.join([_, ledge_name, video_date, str(start_time), str(end_time)]) + '.mp4')
    )
