import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
import sys
import numpy as np
import cv2
import yaml
from pathlib import Path


# Read arguments
video_meta_path = sys.argv[1]
vid_sourcefold = sys.argv[2]
vid_outfold = sys.argv[3]
im_outfold = sys.argv[4]
yaml_outfold = sys.argv[5]
model = sys.argv[6]


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

            vid_rel_path = os.path.join(vid_sourcefold, 'Video'+year, ledge_name, video_date)
            filename_out = vid_outfold+viddat[0][:-4]+"_"+str(int(startsec))+"_"+str(int(endsec))+".mp4"

            ffmpeg_extract_subclip(
                os.path.join(vid_rel_path, viddat[0]),
                startsec,
                endsec,
                targetname = filename_out
            )

def save_all_frames(ext='jpg'):
    cap = cv2.VideoCapture(os.path.join(vid_outfold+file))

    if not cap.isOpened():
        return

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(os.path.join(im_outfold+file), str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return

<<<<<<< HEAD
def create_yaml():
    
    files = list(Path(annot_outfold).glob('**/*'))

    for filename in files: 

        pred = np.loadtxt(filename)

        width = 2560
        height = 1440

        # Always in file
        data_dict = {}
        data_dict["image"] = filename.name
        data_dict["size"] = {"depth": 3, "height": height, "width": width}
        data_dict["source"] = {"framenumber": 0, "path": "na", "video": "na"}
        data_dict["state"] = {"verified": False, "warnings": 0}

        if len(pred) > 0:
            data_dict["objects"] = []

            for ind in range(0, pred.ndim):
                if pred.ndim == 1: 
                    tdat = pred
                else:
                    tdat = pred[ind]
                data_dict["objects"].append(
                    {
                        "bndbox": {
                            "xmax": int(np.clip((tdat[1] + tdat[3] * 0.5) * width, 0, width - 1)),
                            "xmin": int(np.clip((tdat[1] - tdat[3] * 0.5) * width, 0, width - 1)),
                            "ymax": int(np.clip((tdat[2] + tdat[4] * 0.5) * height, 0, height - 1)),
                            "ymin": int(np.clip((tdat[2] - tdat[4] * 0.5) * height, 0, height - 1)),
                        },
                        "name": "fish",

                    }
                )

        write_yaml_to_file(data_dict, filename.stem)
=======
>>>>>>> ca93012 (update auto annotate)


def annotate_images():

    # Load a pretrained YOLO model
    model = YOLO(model)

    # List of videos for inference 
    ims = os.listdir("../images")

    # Run
    for im in ims: 

        if len(im) > 20:

            results = model(f'{im_outfold}/{im}')

            # Width and height
            imread = cv2.imread(f'{im_outfold}/{im}')
            width = imread.shape[1]
            height = imread.shape[0]

            # Process results list
            boxes = []
            boxesxyxy = []

            for r in results:
                boxes.append(r.boxes.xywh.tolist()) 
                boxesxyxy.append(r.boxes.xyxy.tolist())

            # Concatenate outputs
            boxesx = sum(boxes, [])
            boxesxyxy2 = sum(boxesxyxy, [])

            # Save as data frames
            nobj = len(boxesx)

            filename = im.replace(".jpg", ".txt")
            filename_simpl = im.replace(".jpg", "")


            # .yaml
            # Always in file
            data_dict = {}
            data_dict["image"] = filename
            data_dict["size"] = {"depth": 3, "height": height, "width": width}
            data_dict["source"] = {"framenumber": 0, "path": "na", "video": "na"}
            data_dict["state"] = {"verified": False, "warnings": 0}


            if nobj > 0:

                data_dict["objects"] = []

                for row in range(0, nobj):
                    tdat = boxesxyxy2[row]
                    
                    data_dict["objects"].append(
                        {
                            "bndbox": {
                                "xmax": tdat[2],
                                "xmin": tdat[0],
                                "ymax": tdat[3],
                                "ymin": tdat[1],
                            },
                            "name": "fish",

                        }
                    )
            
            write_yaml_to_file(data_dict, filename_simpl)


            # Plain annotation
            #if nobj > 0:

                #y = np.empty([nobj, 5], dtype = float)
                #for row in range(0, nobj):
                #    y[row, 1] = (boxesx[row][0]+(.5*boxesx[row][2]))/width # x 
                #    y[row, 2] = (boxesx[row][1]+(.5*boxesx[row][3]))/height # y 
                #    y[row, 3] = (boxesx[row][2])/width # w 
                #    y[row, 4] = (boxesx[row][3])/height # h 
                
                #np.savetxt(f'../dataset/annotations/{filename}', y, fmt="%i %1.4f %1.4f %1.4f %1.4f")
            
            #else:
             #   open(f'../dataset/annotations/{filename}', 'a').close()



def write_yaml_to_file(py_obj,filename_simpl):
    with open(f'{yaml_outfold}{filename_simpl}.yaml', 'w',) as f :
        yaml.dump(py_obj,f,sort_keys=False) 


## RUN

# Run video cutting
#results = cut_vid() 

# Extract frames from all vids 
#for file in os.listdir(vid_outfold):
#    save_all_frames()

# Annotate
<<<<<<< HEAD
base_model = YOLOv8Base(ontology=CaptionOntology({"fish": "fish"}), weights_path=model)
base_model.label(input_folder=im_outfold,output_folder=annot_outfold, device = 0)
=======
#base_model = YOLOv8Base(ontology=CaptionOntology({"fish": "fish"}), weights_path=model)
#base_model.label(input_folder=im_outfold,output_folder=annot_outfold)
>>>>>>> ca93012 (update auto annotate)

# Convert to yaml
#results = create_yaml()

results = annotate_images()





# RUN example (JHS MAC)
#python3 dataset/auto_annotate.py "../data/fishvids.csv" "../vids/" "../vids/" "../images/" "../dataset/annotations/" "../dataset/annotations_yaml/" "../models/best.pt"

# Run example (Sprattus)
<<<<<<< HEAD
#python3 dataset/auto_annotate.py "../data/fishvids.csv" "../../../../../../../../mnt/BSP_NAS2/Video/" "../vids/" "../images/" "../data/annotations/" "../data/annotations_yaml/" "runs/detect/train48/weights/best.pt"
=======
#python3 dataset/auto_annotate.py "../data/fishvids.csv" "../../../../../../../../mnt/BSP_NAS1/Video/" "../vids/" "../images/" "../dataset/annotations_yaml/" "runs/detect/train12/weights/best.pt"
>>>>>>> ca93012 (update auto annotate)

# Run example (Larus)
#python3 dataset/auto_annotate.py "../data/fishvids.csv" "../../../../../mnt/BSP_NAS2/BSP_data/Video/" "../vids/" "../images/" "../data/annotations/" "../data/annotations_yaml/" "../models/best_train41.pt"

