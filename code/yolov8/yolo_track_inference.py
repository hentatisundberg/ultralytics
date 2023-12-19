import os
import shutil
import datetime
import pandas as pd
import pathlib 
from ultralytics import YOLO
import seaborn as sns
import matplotlib.pyplot as plt


# Select tracker and adujt tracker parameters in their yaml files
tracker = "ultralytics/cfg/trackers/bytetrack.yaml"

tracker_name = tracker.split("/")[-1].split(".")[0]

# Load a pretrained YOLO model
model = YOLO("../../../../../../mnt/BSP_NAS2_work/fish_model/models/best_train57.pt")

# Perform object detection 
vid_dir = pathlib.Path("../../../../../../mnt/BSP_NAS2_work/fish_model/tracking_videos/")

#vid_dir = pathlib.Path("../../../../../mnt/BSP_NAS2/Video/Video2022/FAR3")


# If subfold:
# subfolds = ["2022-07-05/", "2022-07-07/"]
#vids = []
#for i in subfolds:
#    vids.append(list(vid_dir.joinpath(i).glob("*.mp4")))##

#vids2 = []
#for xs in vids:
#    for x in xs:
#        vids2.append(x)

#vids = vids2


# Else:
vids = list(vid_dir.glob("*.mp4"))


# SetUp output folder to save csv
output_folder = 'inference/tracking/'+tracker_name+"_____"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
os.mkdir(output_folder)

fps = 25

for vid in vids: 

    # Pick out relevant video information
    filename = vid.name
    name = filename.split("_")
    time = name[3].split(".")
    ledge = name[1]

    starttime = pd.to_datetime(name[2]+" "+time[0]+":"+time[1]+":"+time[2])
    starttime_u = starttime.timestamp()

    results = model.track(vid, 
                          stream=True, 
                          tracker=tracker, 
                          save = True,
                          show = False, 
                          device = 0)

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

            # Plot image with boxes
            #im_array = r.plot()  # plot a BGR numpy array of predictions
            #im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            #im.show()  # show image

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

    shutil.copy2(tracker, output_folder)
    out.to_csv(f'{output_folder}/{filename}_{tracker_name}.csv')


# Plot most recent track 
#palette = sns.color_palette("bright")
#sns.set(rc = {'axes.facecolor': 'white'})
#ax = sns.scatterplot(x= out["x"], y=out["y"], hue = out["track_id"].astype("int"), palette = palette)
#ax.invert_yaxis()
#ax.grid(False)
#plt.savefig("temp/"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
#plt.close()


