

from functions import save_custom_frames
from pathlib import Path


eidercams2023 = ["EJDER2", "EJDER9STRAND", "EJDER4", "EJDER5", "EJDER1", "EJDER6", "EJDER10STRAND", "EJDER7", 
                        "EJDER11STRAND",  "EJDER8"]

vid = Path("../../../../../../mnt/BSP_NAS2/Video/Video2023/")
savepath = Path("/home/jonas/Documents/python/ultralytics-1/temp")

# Save first frame in all videos
for folder in eidercams2023: 
    vid2 = vid.joinpath(folder)
    files = list(vid2.rglob("*.mp4"))
    for file in files: 
        save_custom_frames(file, savepath)

# Save 