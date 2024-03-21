

from functions import init_dir, df_from_db, insert_to_db

# Add new metrics to existings trackstats

def add_stats(db, metric_name, db_outname):
    df = df_from_db(db, 'ledge != "X"', 'ledge != "X"', True)
    out = [] 
    for row in df.index:
        out.append(init_dir(df.iloc[row]))
    df[metric_name] = out
    insert_to_db(df.iloc[:,2:], db_outname)

add_stats("inference/Inference_stats_nomerge.db", "QUIDUIFCHIH", "inference/Inference_stats_nomerge2.db")