
import os

# Find all image files
# If yaml file is missing - remove image

images = os.listdir("../images")
yaml = os.listdir("../data/annotations_yaml")
yaml = [item[:-5] for item in yaml] 

count = 0
for im in images:
    imname = im[:-5]
    
    if imname not in yaml: 
        print(f'remove {imname}')
        count += 1
        print(count)
