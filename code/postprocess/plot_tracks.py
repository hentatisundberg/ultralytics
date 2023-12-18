




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns



inputfile = "inference/tracking/botsort_custom2_____20231217T211805/Auklab1_FAR3_2023-07-05_07.00.00.mp4_botsort_custom2.csv"


dat = pd.read_csv(inputfile)
dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)

pdat = dat[dat["conf"] > 0.7]

# Plot most recent track 
palette = sns.color_palette("bright")
sns.set(rc = {'axes.facecolor': 'white'})
ax = sns.scatterplot(x= pdat["time2"], y=pdat["x"], hue = pdat["track_id"].astype("int"), palette = palette)
ax.invert_yaxis()
ax.grid(False)
plt.show()



# Plot most recent track 
#palette = sns.color_palette("bright")
#sns.set(rc = {'axes.facecolor': 'white'})
#ax = sns.lineplot(x= dat["time2"], y=dat["y"], hue = dat["track_id"].astype("int"), palette = palette)
#ax.invert_yaxis()
#ax.grid(False)
##plt.savefig("temp/"+"tracks_time_"+newname+".jpg")
#plt.close()
