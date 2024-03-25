
import sys
sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/") # Mac
sys.path.append("/home/jonas/Documents/vscode/ultralytics/code/generic_functions/") # Sprattus
sys.path.append("/home/jonas/Documents/python/ultralytics-1/code/generic_functions/") # Larus
from functions import init_dir, df_from_db, insert_to_db

# Add new metrics to existings trackstats

def add_stats(db, metric_name, db_outname):
    df = df_from_db(db, 'ledge != "X"', 'ledge != "X"', True)
    out = [] 
    for row in df.index:
        out.append(init_dir(df.iloc[row]))
    df[metric_name] = out
    insert_to_db(df.iloc[:,2:], db_outname)

add_stats("../../../../../../mnt/BSP_NAS2_work/fish_model/inference/Inference_stats_nomergeALL.db", 
          "init_dir", 
          "../../../../../../mnt/BSP_NAS2_work/fish_model/inference/Inference_stats_nomergeC.db")