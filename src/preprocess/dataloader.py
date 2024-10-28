import glob
import pandas as pd
import os
import logging as log

DATA_PATH = "data"


def __load_csv(filename):
    return pd.read_csv(filename)


def load_raw_data():
    path = os.path.join(DATA_PATH, 'raw', "*.csv")
    log.info('Loading raw data from %s', path)

    df = None
    for filename in glob.glob(path):
        if df is None:
            df = __load_csv(filename)
        else:
            df = pd.concat([df, __load_csv(filename)],
                           axis=0, ignore_index=True)
    log.info('Raw data loaded: %s', df.shape)
    return df
