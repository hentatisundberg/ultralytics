


import sys
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/") # Sprattus
sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/") # Mac
from functions import remove_similar_images

# Read arguments
remove = remove_similar_images("../../../../Downloads/ims2/", 200000)

# Remove all files in the the remove list
for file in remove:
    file.unlink()



