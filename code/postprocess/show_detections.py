

import matplotlib.pyplot as plt
import pandas as pd


dat = pd.read_csv("inference/Auklab1_TRI3_2023-07-03_05.00.00_570_779.mp4.csv")
dat["time2"] = pd.to_datetime(dat["time"]*1000*1000*1000)

fig, ax = plt.subplots()

ax.bar(x = dat["time"], height = dat["w"], color = "black", width = 0.01)
plt.show()


