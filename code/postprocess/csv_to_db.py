
import pandas as pd
import sqlite3

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


def insert_to_db(file):
    file = pd.read_csv(file)
    file = file.reset_index()
    con_local = create_connection("inference/Inference_raw.db")
    file.to_sql("Inference", con_local, if_exists='append')



