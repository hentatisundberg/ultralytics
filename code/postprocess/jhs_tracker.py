

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path



# Create initial tracks based on conservative detection threshold
# Assign points around those to existing tracks 
# Merge tracks 


conf_init_thres = .75
conf_assign_thres = .5
track_assign_thres = 1000 
time_space_scale = 10
track_merge_thres = 5000


# Input file 
inputfold = Path("inference/").absolute()
files = list(inputfold.glob("*.csv"))

for file in files: 

    name = file.name
    all = pd.read_csv(file)
    
    if len(all) > 20:
        all["time2"] = pd.to_datetime(all["time"]*1000*1000*1000)
        cond1 = all["conf"] < conf_init_thres
        cond2 = all["conf"] > conf_assign_thres
        test = all[cond1 & cond2]

        # Initital track creation
        dat = all[all["conf"] > conf_init_thres]

        if len(dat) > 20:
            start = 6
            rows = range(start, len(dat))
            dat["track_temp"] = range(0, len(dat))

            track = [0]*start
            current_track = 0

            for row in rows:

                current = dat.iloc[row]
                previous = dat.iloc[(row-5):(row-1)]
                x, y, frame = previous["x"]-current["x"], previous["y"]-current["y"], previous["frame"]-current["frame"]

                dist0 = np.sqrt(x**2 + y**2)
                elapse0 = abs(frame)*time_space_scale
                score = dist0+elapse0
                minval = min(score)
                nearest = np.argwhere(score == minval)[0][0]

                if minval < track_assign_thres: 
                    current_track = track[-1] 
                else: 
                    current_track = current["track_temp"]
                track.append(current_track)

            dat["nt3"] = track


            # Aggregate data for initial tracks
            trackstats = dat.groupby(["nt3"]).aggregate({
                "frame": ["first", "last"], 
                "x": ["first", "last"],
                "y": ["first", "last"],
                "y": "count",
                "time2": "first"
                })


            # Find un-associated points within existing clusters

            # Here, use points (from test) only that is located within existing clusters
            # The existing clusters are in trackstats

            rows0 = range(0, len(trackstats))
            for row in rows0:
                cond1 = test["frame"] > trackstats.iloc[row]["frame"]["first"]
                cond2 = test["frame"] < trackstats.iloc[row]["frame"]["last"]
                temp_assign = test[cond1 & cond2]
                temp_orig = dat[dat["nt3"] == trackstats.index[row]]
                rows = range(1, len(temp_assign))
                tx = temp_orig
                for r in rows: 
                    td = temp_assign.iloc[r-1]
                    eucl = min(np.sqrt((td["x"]-temp_orig["x"])**2 + (td["y"]-temp_orig["y"])**2))
                    if eucl < 1000:
                        tx = pd.concat([tx, temp_assign.iloc[(r-1):r]])
                tx["nt4"] = temp_orig.iloc[0]["nt3"]
                if 'out' in locals():
                    out = pd.concat([out, tx])
                else:
                    out = tx

            out["nt4"] = out["nt4"].fillna(-1)


            # Assign to clusters forward and backward

            # The logic should be: 
            # Take data frame "out" 
            # For each cluster: 
            # Temporarily add all points between last point in that one in earliest point 
            # Fill in track id according to critera 



            # Here, use points (from test) only that is located within existing clusters
            trackstats2 = out.groupby(["nt4"]).aggregate({
                "frame": ["first", "last"], 
                "x": ["first", "last"],
                "y": ["first", "last"],
                "y": "count",
                "time2": "first"
                })

            trackstats2 = trackstats2[trackstats2.index != -1.0]


            # Forward assign 

            #out2 = test.merge(out, left_index = True, right_index = True, how = "outer")
            #out2["nt4"] = out2["nt4"].fillna(-1)


            test2 = test.join(out["nt4"])
            test2["nt4"] = test2["nt4"].fillna(-1)

            rows0 = range(0, len(trackstats2))
            for row in rows0:    # Track by track
                if row == len(trackstats2)-1:
                    firstnext = 1000000
                else: 
                    firstnext = trackstats2.iloc[(row+1)]["frame"]["first"]
                cond1 = test2["nt4"] == -1.0
                cond2 = test2["frame"] > trackstats2.iloc[row]["frame"]["last"]
                cond3 = test2["frame"] < firstnext 
                temp_assign = test2[cond1 & cond2 & cond3]
                temp_orig = out[out["nt4"] == trackstats2.index[row]]
                rows = range(1, len(temp_assign))
                tx = temp_orig
                for r in rows: 
                    td = temp_assign.iloc[(r-1):r]
                    eucl = min(np.sqrt((td["x"]-tx["x"].iloc[-1])**2 + (td["y"]-tx["y"].iloc[-1])**2))
                    time = ((td["frame"]-tx["frame"].iloc[-1])*time_space_scale).iloc[0]
                    if time+eucl < 10000:
                        tx = pd.concat([tx, td])
                tx["nt5"] = trackstats2.index[row]
                if 'out2' in locals():
                    out2 = pd.concat([out2, tx])
                else:
                    out2 = tx


            # Backward assign
            test3 = test2.iloc[::-1]
            rows0 = range(0, len(trackstats2)-1)
            for row in rows0:    # Track by track
                if row == len(trackstats2):
                    firstnext = 0
                else: 
                    firstnext = trackstats2.iloc[(row)]["frame"]["first"]
                cond1 = test3["nt4"] == -1.0
                cond2 = test3["frame"] > trackstats2.iloc[row+1]["frame"]["last"]
                cond3 = test3["frame"] < firstnext 
                if len(temp_assign) > 0:
                    temp_assign = test2[cond1 & cond2 & cond3]
                    temp_orig = out2[out2["nt5"] == trackstats2.index[row]]
                    rows = range(1, len(temp_assign))
                    tx = temp_orig
                    for r in rows: 
                        td = temp_assign.iloc[(r-1):r]
                        eucl = min(np.sqrt((td["x"]-tx["x"].iloc[-1])**2 + (td["y"]-tx["y"].iloc[-1])**2))
                        time = abs(((td["frame"]-tx["frame"].iloc[-1])*time_space_scale).iloc[0])
                        if time+eucl < 100000:
                            tx = pd.concat([tx, td])
                    tx["nt6"] = trackstats2.index[row]
                if 'out3' in locals():
                    out3 = pd.concat([out3, tx])
                else:
                    out3 = tx

            # Merge final clusters 

            # Here, use points (from test) only that is located within existing clusters
            trackstats3 = out3.groupby(["nt6"]).aggregate({
                "frame": ["first", "last"], 
                "x": ["first", "last"],
                "y": ["first", "last"],
                "conf": "count",
                "time2": "first"
                })


            update_track = [trackstats3.index[0]]
            rows = range(0, len(trackstats3)-1)
            for row in rows:
                previous_track = trackstats3.index[0]
                d1 = trackstats3.iloc[row]
                d2 = trackstats3.iloc[row+1]    
                x1, y1, t1 = d1["x"]["last"], d1["y"]["last"], d1["frame"]["last"]
                x2, y2, t2 = d2["x"]["first"], d2["y"]["first"], d2["frame"]["first"]
                eucl = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                elapse = time_space_scale*abs(t1-t2)
                dist = eucl + elapse
                print(dist)
                if dist < track_merge_thres:
                    update_track.append(1)
                else:
                    update_track.append(0)


            # Update track in trackstats3
            new_track = [0]
            orig_track = list(trackstats3.index)
            ind = range(1, len(orig_track))
            for i in ind:
                if update_track[i] == 1:
                    new_track.append(new_track[i-1])
                else: 
                    new_track.append(orig_track[i])
            trackstats3["final"] = new_track


            # Update track id in data frame 
            out3x = out3.merge(trackstats3["final"], left_on = "nt6", right_index = True, how = "left")


            # Update summary statistics
            trackstats4 = out3x.groupby(["final"]).aggregate({
                "frame": ["first", "last"], 
                "x": ["first", "last"],
                "y": ["first", "last"],
                "conf": "count",
                "time2": "first"
                })


            # Plot most recent track 
            palette = sns.color_palette("bright")
            sns.set(rc = {'axes.facecolor': 'white'})
            ax = sns.scatterplot(x= out3x["time2"], y=out3x["y"], hue = out3x["final"].astype("int"), palette = palette)
            ax = sns.scatterplot(x= all["time2"], y=all["y"], s = 10, marker = ".", c = "black")
            ax.invert_yaxis()
            ax.grid(False)
            plt.savefig(f'temp/{name}.png')
            plt.close()

            trackstats4.to_csv(f'temp/{name}.csv')
del out, out2, out3






# PLOT RESULTS


#fig, ax = plt.subplots()
#ax = plt.scatter(x= dat["time2"], y=dat["x"], s = 100, marker = "h", c = "blue")
#ax.scatter(x= out["time2"], y=out["x"], s = 30, marker = "h", c = "red", label = "within")
#ax.scatter(x= all["time2"], y=all["x"], s = 3, marker = "h", c = "black", label = "all")
#ax.scatter(x= out2["time2"], y=out2["x"], marker = "o", s=200, facecolors='none', edgecolors='g', label = "forward")
#ax.scatter(x= out3["time2"], y=out3["x"], marker = "o", s=200, facecolors='none', edgecolors='r', label = "forward")
#ax.legend()
#plt.show()




