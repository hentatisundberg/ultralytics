
import pandas as pd
from pathlib import Path
import yaml

# Goal: 
# take all annotations and summarize them in a single file



input_yaml = "../../../../Desktop/annot_eider/annotations/"
files = list(Path(input_yaml).glob('*.yaml'))


def get_yaml_annotations():
    
    for filename in files: 

        yaml_dict = None
        print(filename)
        
        try: 
            yaml_dict = yaml.load(open(filename), Loader=yaml.FullLoader)

            height, width = yaml_dict["size"]["height"], yaml_dict["size"]["width"] 

            bounding_boxes = []
            if "objects" in yaml_dict:
                    for obj in yaml_dict["objects"]:
                        name = obj["name"]
                        xmin = obj["bndbox"]["xmin"]
                        ymin = obj["bndbox"]["ymin"]
                        xmax = obj["bndbox"]["xmax"]
                        ymax = obj["bndbox"]["ymax"]
                        objx = (xmin+xmax)/2/width 
                        objy = (ymin+ymax)/2/height
                        objwidth = (xmax-xmin)/width
                        objheight = (ymax-ymin)/height

                        bounding_boxes.append([name, objx, objy, objwidth, objheight])
            
            bb = pd.DataFrame(bounding_boxes, columns=["name", "objx", "objy", "objwidth", "objheight"])
            bb["filename"] = filename.stem

            try:    
                df_all = pd.concat([df_all, bb])
            except:
                df_all = bb 
                pass
        except: pass
    return df_all



# Create dictionary from all yaml annotations


bbs = get_yaml_annotations()
bbs.to_csv("../../../../Desktop/annot_eider/all_annotations.csv", index=False)


# Assign numbers to classes, create class file and txt annotation files
classes = bbs["name"].unique()
class_dict = dict(zip(classes, range(len(classes))))
pd.Series(classes).to_csv("../../../../Desktop/annot_eider/eider_classes.txt", index=False)


# Create txt annotation files
for file in bbs["filename"].unique(): 
    bbs_file = bbs[bbs["filename"] == file]
    bbs_file["class"] = bbs_file["name"].map(class_dict)
    bbs_file = bbs_file[["class", "objx", "objy", "objwidth", "objheight"]]
    bbs_file.to_csv(f"../../../../Desktop/annot_eider/annotations/{file}.txt", index=False, header=False, sep=" ")



# Read all annotations (csv files) and summarize them in a single file
#orig_files = list(Path("../../../../../../Volumes/JHS-SSD2/Eider_model/dataset_plain/").rglob("*.txt"))

#for file in orig_files: 
#    print(file)
#    # Read space separated file
#    try: 
#        df = pd.read_csv(file, sep=" ", header=None)
#        df["image"] = file.parent
#        try: 
#            df_all = pd.concat([df_all, df])
#        except: 
#            df_all = df
#    except: pass
#df_all.to_csv("../../../../../../Volumes/JHS-SSD2/Eider_model/dataset_plain/all_annotations.csv", index=False)

 