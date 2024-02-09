import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import shutil

# Video compression using os

import cv2
from pathlib import Path

def compress_vids(input, outputdir):

    file = Path(input)
    name = file.name
    output = outputdir.joinpath(name)
    print(output)

    cap = cv2.VideoCapture(input)

    if not cap.isOpened():
        print("Error: Could not open the input video file.")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Change this to your desired codec
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    print(frame_size)
    print(frame_rate)

    out = cv2.VideoWriter(output, fourcc, frame_rate, frame_size, isColor=True)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Detection models and trackers 
tracker = "ultralytics/cfg/trackers/bytetrack.yaml"
tracker_name = tracker.split("/")[-1].split(".")[0]
model = YOLO("../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train57.pt")

# Work folders
input_dir = Path("../../../../../../mnt/BSP_NAS2_work/fish_model/clips3")
#temp_save_dir = Path("../../../../../../mnt/BSP_NAS2_work/fish_model/clips_temp") 
save_dir = Path("../../../../../../mnt/BSP_NAS2_work/fish_model/clips_annot2")


vids = list(input_dir.glob("*.mp4"))


# CUT IN THE SAME SCRIPT!

def annotate_vids(vids):

    for vid in vids: 

        # Pick out relevant video information
        results = model(vid, 
                        conf = 0.1,
                        stream=True,  
                        save = True,
                        show = False, 
                        save_frames = False)
        for result in results: 
            result.boxes
        shutil.rmtree(f"runs/detect/predict13/{vid.stem}_frames") # Remove frames
        compress_vids(f"runs/detect/predict13/{vid.stem}.avi", Path("../../../../../../mnt/BSP_NAS2_work/fish_model/clips_annot2/"))



compress_vids(f"runs/detect/predict13/BONDEN322061505-3.avi", Path("runs/detect/predict/"))
