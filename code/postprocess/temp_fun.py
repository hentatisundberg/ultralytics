
def associate_points_within(track_data, all_data):
    tracks = track_data["track_id"].unique().astype("int")
    unassoc = all_data
    unassoc = unassoc[unassoc["track_id"] == -1]
    outdata = pd.DataFrame()
    
    for track in tracks: 

        track_temp = track_data[track_data["track_id"] == track]
        minf, maxf = np.min(track_temp["frame"]), np.max(track_temp["frame"])
        
        # Points within

        cand_within = unassoc.loc[(unassoc["frame"] > minf) & (unassoc["frame"] < maxf+framedist)]

        ids = cand_within.index
        n_it = int(np.ceil(len(ids)/chunksize))
        res = []
    
        for i in list(range(0, n_it)):
            for ele in range(chunksize):
                res.append(i)
        
        res = res[0:len(ids)] # How to split dataset
        df = pd.DataFrame(list(ids), columns = ["ids"])  # Index of unassociated point that will be checked
        df["res"] = res
        
        for j in range(0, n_it): 

            current = df[df["res"] == j]["ids"]
            d2 = cand_within[cand_within.index.isin(current)][["frame", "x", "y"]]
            d1 = track_temp[["frame", "x", "y"]]

            iterate = 1
            while iterate == 1: 

                if len(track_temp) < size: 
                    ss1 = len(d1)
                else: 
                    ss1 = size
                
                    d1first = d1.iloc[0:1]
                    d1last = d1.iloc[-1:]
                    d1sample = d1.sample(ss1)
                    d1s = pd.concat([d1first, d1sample, d1last])

                    d1s["frame"] = d1s["frame"]*time_scaling_assign
                    d2["frame"] = d2["frame"]*time_scaling_assign

                    # Min distance per point to track
                    dist = []

                    # Loop through each candidate point, recover its min distance 
                    points = range(0, len(d2))
                    for point in points: 
                        p = np.array(d2.iloc[point].tolist())
                        d = np.linalg.norm(p - np.array(d1s.values.tolist()), axis=1)
                        dist.append(np.min(d))

                    nearest = np.min(dist)

                    if nearest < track_assign_thresh:
                        minpos = cand_bef.loc[dist == nearest]
                        minpos["track_id"] = track
                        track_temp = pd.concat([track_temp, minpos]) # Update track data
                        cand_bef.drop(minpos.index, inplace = True) # Delete from candidates
                        nrow = len(track_temp) 
                        if len(cand_bef) == 0:
                            iterate = 0
                            outdata = pd.concat([outdata, track_temp])
                            print(f'merged all in chunk {j} in track {track}')
                    else:
                        iterate = 0
                        outdata = pd.concat([outdata, track_temp])
                        print(f'all the rest of points outside specified range')
    return outdata



