

from pathlib import Path
import pandas as pd

#x = pd.read_csv("../../../../../../mnt/BSP_NAS2_work/fish_model/inference_log.csv")

vid_dir = Path("../../../../../mnt/BSP_NAS2/Video/")

dates22 = list(pd.date_range(start='15/06/2022', end='15/07/2022').astype(str))
dates23 = list(pd.date_range(start='15/06/2023', end='15/07/2023').astype(str))
ledges = list(["FAR3", "FAR6", "TRI3", "TRI6", "BONDEN3", "BONDEN6", "ROST3", "ROST6"])

paths = []

for i in ledges:
    for j in dates22:
        paths.append(vid_dir.joinpath("Video2022/"+i+"/"+j+"/"))

for i in ledges:
    for j in dates23:
        paths.append(vid_dir.joinpath("Video2023/"+i+"/"+j+"/"))

inference_log = pd.DataFrame()
inference_log["paths"] = paths
inference_log["started"] = 0
inference_log["finished"] = 0
inference_log["device"] = 0

inference_log.to_csv("data/inference_log.csv")