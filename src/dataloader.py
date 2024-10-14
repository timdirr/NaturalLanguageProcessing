import glob
import pandas as pd

DATA_PATH="data/"


def __load_csv(filename):
    return pd.read_csv(filename)


def load_raw_data():
    path = DATA_PATH
    if __debug__:
        path = "../" + path

    df = None
    for filename in glob.glob(path + 'raw/*.csv'):
        if df is None:
            df = __load_csv(filename)
        else:
            df = pd.concat([df, __load_csv(filename)], axis=0, ignore_index=True)

    return df