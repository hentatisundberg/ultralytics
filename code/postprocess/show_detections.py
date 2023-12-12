

import matplotlib.pyplot as plt
import pandas as pd


dat = pd.read_csv("inference/Auklab1_ROST3_2022-07-01_23.00.00.mp4.csv")
dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)

pt = dat[dat["conf"] > .7]

fig, ax = plt.subplots()

ax.scatter(pt["time2"], pt["x"], color = "black")
ax.scatter(pt["time2"], pt["y"], color = "red")
plt.show()

#plt.savefig("temp/plot.png")
#plt.close()


