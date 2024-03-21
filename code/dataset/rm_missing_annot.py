
import os
from pathlib import Path

# Find all image files
# If yaml file is missing - remove image

images = list(Path("/Volumes/JHS-SSD2/eider/all/").glob("*.jpg"))
yaml = list(Path("/Volumes/JHS-SSD2/eider/all/").glob("*.yaml"))
yaml_stem = [x.stem for x in yaml]

count = 0
for im in images:
    
    imname = im.stem
    
    if imname not in yaml_stem: 
        print(f'removing {imname}')
        #os.remove(im)        
        count += 1
        print(count)

