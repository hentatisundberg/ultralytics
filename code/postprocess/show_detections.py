

import matplotlib.pyplot as plt
import pandas as pd


dat = pd.read_csv("inference/Auklab1_FAR3_2022-07-08_04.00.00.mp4.csv")
dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)

fig, ax = plt.subplots()

ax.scatter(dat["time2"], dat["x"], color = "black")
ax.scatter(dat["time2"], dat["y"], color = "red")
plt.show()

#plt.savefig("temp/plot.png")
#plt.close()


