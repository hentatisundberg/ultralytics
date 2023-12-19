import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import sys


# Select tracker and adujt tracker parameters in their yaml files
tracker = "ultralytics/cfg/trackers/bytetrack.yaml"

tracker_name = tracker.split("/")[-1].split(".")[0]

# Load a pretrained YOLO model
model = YOLO("../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train57.pt")
device = sys.argv[1]
startrow = int(sys.argv[2])

its = range(startrow, 10000)
for it in its: 
 
    # Load file which tracks progress
    filelist = pd.read_csv("../../../../../../mnt/BSP_NAS2_work/fish_model/inference_log.csv")
    
    vid_dir = Path(filelist.iloc[it]["paths"])
    vids = list(vid_dir.glob("*.mp4"))

    # SetUp output folder to save csv
    output_folder = Path("../../../../../../mnt/BSP_NAS2_work/fish_model/inference/")

    fps = 25

    for vid in vids: 

        # Pick out relevant video information
        filename = vid.name

        t1 = pd.Series(filename).str.contains('00.00.00', regex=False)[0]
        t2 = pd.Series(filename).str.contains('01.00.00', regex=False)[0]
        t3 = pd.Series(filename).str.contains('23.00.00', regex=False)[0]

        if any([t1, t2, t3]) == False:  
            name = filename.split("_")
            time = name[3].split(".")
            ledge = name[1]

            starttime = pd.to_datetime(name[2]+" "+time[0]+":"+time[1]+":"+time[2])
            starttime_u = starttime.timestamp()

            results = model.track(vid, 
                                stream=True, 
                                tracker=tracker, 
                                save = False,
                                show = False, 
                                device = device, 
                                persist=True)

            # Process results list
            time = []
            boxes = []
            track_ids = []
            confs = []
            classes = []
            framenum = []
            counter = starttime_u
            counterx = 0

            for r in results:
                if not r.boxes.conf.nelement() == 0 :
                    boxes.append(r.boxes.xyxy.tolist())
                    if r.boxes.id is not None:
                        track_ids.append(r.boxes.id.tolist())
                    else:
                        track_ids.append([-1 for _ in r.boxes])
                    classes.append(r.boxes.cls.tolist())
                    confs.append(r.boxes.conf.tolist())
                    ndetect = len(r.boxes.conf.tolist())
                    time.append([counter] * ndetect)
                    framenum.append([counterx] * ndetect)

                counter += 1/fps
                counterx += 1

            # Concatenate outputs
            conf = sum(confs, [])
            classes = sum(classes, [])
            boxesx = sum(boxes, [])
            track_ids = sum(track_ids, [])
            times = sum(time, [])
            framenums = sum(framenum, [])

            # Save as data frames
            out1 = pd.DataFrame(boxesx, columns = ["x", "y", "w", "h"])
            out2 = pd.DataFrame(list(zip(track_ids, classes, conf, times, framenums)), columns = ["track_id","class", "conf", "time", "frame"])

            out = out1.merge(out2, left_index = True, right_index = True)
            out["ledge"] = ledge
            out["filename"] = filename

            out.to_csv(output_folder.joinpath(filename[:-4]+".csv"))

