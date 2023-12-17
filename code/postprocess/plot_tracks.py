




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns



inputfile = "inference/_-FAR3_2023-07-05_04.00.00.mp4.csv"

dat = pd.read_csv(inputfile)
dat["time2"] = pd.to_datetime(dat["time2"])

# Plot most recent track 
palette = sns.color_palette("bright")
sns.set(rc = {'axes.facecolor': 'white'})
ax = sns.scatterplot(x= dat["time2"], y=dat["x"], hue = dat["track_id"].astype("int"), palette = palette)
ax.invert_yaxis()
ax.grid(False)
plt.show()
plt.close()



# Plot most recent track 
#palette = sns.color_palette("bright")
#sns.set(rc = {'axes.facecolor': 'white'})
#ax = sns.lineplot(x= dat["time2"], y=dat["y"], hue = dat["track_id"].astype("int"), palette = palette)
#ax.invert_yaxis()
#ax.grid(False)
##plt.savefig("temp/"+"tracks_time_"+newname+".jpg")
#plt.close()
