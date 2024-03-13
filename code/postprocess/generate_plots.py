
import matplotlib.pyplot as plt



# Plot 
fish = plot_results("inference/Inference_stats_merge.db", "inference/merged_fish.csv", "conf_mean", "nframes", False, True, 4)


fish["date"] = fish["start"].dt.date
fish["H"] = fish["start"].dt.strftime("%H").astype("int")

rangex = range(2, 23)
xlist = list(rangex)
xlist2 = [x + 0.5 for x in xlist]

fig, ax = plt.subplots()
ax.hist(fish["H"], color = "darkred", width = 0.9, bins = rangex)
ax.set_xlabel("Hour")
ax.set_xticks(xlist)
ax.set_ylabel("Number of fish")
fig.suptitle("Time of fish deliveries (H), FAR3-TRI6, 2022-2023")
#plt.legend()
plt.show()


# Fish size?
fig, ax = plt.subplots()
ax.scatter(fish["maxdim_mean"], fish["mindim_mean"], c = "r", alpha = .3, s = 4)
ax.set_xlabel("fish length (pixels)")
ax.set_ylabel("fish width (pixels)")
plt.show()

# Trend fish size?
fish["size"] = fish["maxdim_mean"]*fish["mindim_mean"]
fig, ax = plt.subplots()
ax.scatter(fish["start"], fish["size"], c = "r", alpha = .5, label = "Fish", s = 12)
plt.legend()
plt.show()

# End positions of fish tracks
bg22 = cv2.imread("data/bg2022.jpg")
bg23 = cv2.imread("data/bg2023.jpg")
far3 = fish[fish["ledge"] == "FAR3"]
far3["Yr"] = far3["start"].dt.year

fig, ax = plt.subplots()
imgplot = ax.imshow(bg23)
dx = far3[far3["Yr"] == 2023]
ax.scatter(dx["x_last"], dx["y_last"], c = "y", alpha = .5, s = 12)
fig.suptitle("End position 2023")
plt.show()


# Example tracks on FAR3

date = "2023-07-04"
ledge = "FAR3"
dat = df_from_db("inference/Inference_raw_merge.db", f'ledge == "{ledge}"',  f'strftime("%Y-%m-%d", time2) == "{date}"', False)
preds = pd.read_csv("inference/merged_fish.csv", sep = ";", decimal = ",")
dat2 = pd.merge(dat, preds, on = "track", how = "left")
dat2 = dat2[dat2["multi_y"] > 3]
all_tracks = dat2["track"].unique().tolist()
random.shuffle(all_tracks)

fig, ax = plt.subplots()
imgplot = ax.imshow(bg23)

# Plot tracks in space 
for track in all_tracks[0:9]: 
    data = dat2[dat2["track"] == track]        
    col = np.random.rand(3,)
    x = data["x"]+(.5*data["width"])
    y = data["y"]+(.5*data["height"])
    ax.plot(x, y, c = col, alpha = .8)
    ax.grid(False)

plt.suptitle(f'{ledge}, {date} (10 random tracks)')
plt.show()


# Tracks over time, one day

fig, ax = plt.subplots()

# Plot tracks in space 
for track in all_tracks: 
    data = dat2[dat2["track"] == track]        
    col = np.random.rand(3,)
    x = data["x"]+(.5*data["width"])
    y = -1*(data["y"]+(.5*data["height"]))
    ax.scatter(data["time2"], y, c = col, alpha = .8)
    ax.grid(False)
    #ax.invert_yaxis()

plt.suptitle(f'{ledge}, {date} (all tracks)')
plt.show()








