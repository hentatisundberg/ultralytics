

import matplotlib.pyplot as plt
import pandas as pd


dat = pd.read_csv("inference/Auklab1_FAR3_2022-07-09_04.00.00.mp4.csv")
dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)

fig, ax = plt.subplots()

ax.bar(x = dat["time"], height = dat["w"], color = "black", width = 0.01)
plt.savefig("temp/plot.png")
plt.close()


