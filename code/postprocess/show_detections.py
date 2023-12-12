

import matplotlib.pyplot as plt
import pandas as pd


dat = pd.read_csv("inference/Auklab1_TRI3_2023-07-03_05.00.00_570_779.mp4.csv")
dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)

fig, ax = plt.subplots()

ax.scatter(x = dat["time"], y=dat["x"], color = "black")
ax.scatter(x = dat["time"], y=dat["y"], color = "red")
plt.savefig("this.png")


