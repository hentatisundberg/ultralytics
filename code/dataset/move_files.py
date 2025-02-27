
import pandas as pd
from pathlib import Path
import shutil

base_dir = Path("../../../../../../mnt/BSP_NAS2_vol3/Video/Video2024/")
inputfiles = pd.read_csv("data/filenames.csv")
output_dir = Path("../../../../../../mnt/BSP_NAS2_work/temp/vid")

for row in inputfiles.index:
    vid = Path(inputfiles.loc[row, "path"])
    vid_origin = base_dir.joinpath(vid)
    vid_dest = output_dir.joinpath(vid.name)
    print (vid_dest)
    print (vid_origin)
    # Copy file from origin to destination
    shutil.copy(str(vid_origin), str(vid_dest))
    
    