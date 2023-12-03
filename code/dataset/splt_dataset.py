import os 
import numpy as np 

dir = "../../../../../../mnt/BSP_NAS2_work/fish_model/annotations/"
allfiles = os.listdir(dir+"train/images/")
tot = len(allfiles)
split = [.8, .1, 0]
nval, ntest = round(split[1]*tot,0), round(split[2]*tot,0)
ntrain = tot - (nval+ntest)
rand = [list(np.repeat(0, ntrain)), list(np.repeat(1, nval)), list(np.repeat(2, nval))]
rand = [item for sublist in rand for item in sublist]
np.random.shuffle(rand)
count = 0

for item in allfiles: 
    try: 
        if rand[count] == 1:
            os.rename(dir+"train/images/"+item, dir+"validate/images/"+item)
            os.rename(dir+"train/labels/"+item[:-3]+"txt", dir+"validate/labels/"+item[:-3]+"txt")
        elif rand[count] == 2:
            os.rename(dir+"train/images/"+item, dir+"test/images/"+item)
            os.rename(dir+"train/labels/"+item[:-3]+"txt", dir+"test/labels/"+item[:-3]+"txt")
        count += 1 
    except: pass