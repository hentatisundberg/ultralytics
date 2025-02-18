
import pandas as pd

input = pd.read_csv("data/vid_request.csv", sep = ";", parse_dates = ["start_date", "end_date"])


filenames = []
datetime = []
ledge = []
for row in input.index: 
    r = input.iloc[row] 
    start = r["start_date"]
    end = r["end_date"]
    # Sequence of dates, from start to end, with 1 hour interval
    dates = pd.date_range(start, end, freq='h')
    ftemp = []
    for date in dates:
        # Create filename
        dx = date.strftime("%Y-%m-%d")
        datetime.append(dx)
        path = f'{r["nvr_prefix"]}_{r["camera"]}_{date.strftime("%Y-%m-%d_%H.00.00")}.mp4'
        filenames.append(f'{r["camera"]}/{dx}/{path}')
        ledge.append(r["camera"])
        
# Save list as csv
d = {"camera": ledge, "datetime": datetime, "path": filenames}
filenames = pd.DataFrame(d)
filenames.to_csv("data/filenames.csv", index = False)