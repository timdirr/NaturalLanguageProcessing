import glob
import pandas as pd
import os
import logging as log
import numpy as np
from globals import DATA_PATH, SPLIT_FOLDER, TEST_FILE, TRAIN_FILE, DEV_FILE

from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

import re


def __load_csv(filename):
    return pd.read_csv(filename)

def __save_csv(filename, data):
    data.to_csv(filename, index=False)


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


def load_stratified_data():
    log.info(f"Loading stratified data splits...")
    if os.path.exists(os.path.join(DATA_PATH, SPLIT_FOLDER)):
        if os.path.isdir(os.path.join(DATA_PATH, SPLIT_FOLDER, TEST_FILE)):
            test = __load_csv(os.path.join(DATA_PATH, SPLIT_FOLDER, TEST_FILE))
            train = __load_csv(os.path.join(DATA_PATH, SPLIT_FOLDER, TRAIN_FILE))
            dev = __load_csv(os.path.join(DATA_PATH, SPLIT_FOLDER, DEV_FILE))

            return train, test, dev
        else:
            raise FileNotFoundError(f"{os.path.join(DATA_PATH, SPLIT_FOLDER)} seems to be empty or files are missing")
    else:
        log.info(f"No splits found, creating stratified splits...")
        os.makedirs(os.path.join(DATA_PATH, SPLIT_FOLDER))
        df = conllu2df('conllu_data.conllu')
        y = np.vstack(df["genre"].to_numpy()).astype(int)
        X = df.drop(["genre"], axis=1)

        log.info(f"Creating train-test split with ration 0.1")
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, 0.1)

        log.info(f"Creating stratified dev set as subset with ration 0.1")
        _, _, X_dev, y_dev = iterative_train_test_split(X_train, y_train, 0.1)

        test = X_test
        train = X_train
        dev = X_dev

        test["genre"] = y_test
        dev["genre"] = y_dev
        train["genre"] = y_train

        __save_csv(os.path.join(DATA_PATH, SPLIT_FOLDER, TEST_FILE), test)
        __save_csv(os.path.join(DATA_PATH, SPLIT_FOLDER, TRAIN_FILE), train)
        __save_csv(os.path.join(DATA_PATH, SPLIT_FOLDER, TEST_FILE), dev)

        return train, test, dev


def conllu2df(filename):
    def movie_entry_generator():
        current_id = None
        current_genre = None
        current_description = []

        filepath = os.path.join(DATA_PATH, filename)
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                if line.startswith("# movie_id"):
                    if current_id is not None:
                        yield {
                            "movie_id": current_id,
                            "genre": current_genre,
                            "description": " ".join(current_description)
                        }

                    current_id = line.split('=')[1].strip()
                    current_genre = None
                    current_description = []

                elif line.startswith("# genre"):
                    current_genre = line.split('=')[1].strip()
                    current_genre = np.array(re.sub(r"[\[\]']", '', current_genre).split(' '))

                elif line.startswith("# text"):
                    current_description.append(line.split('=', 1)[1].strip())

            if current_id is not None:
                yield {
                    "movie_id": current_id,
                    "genre": current_genre,
                    "description": " ".join(current_description)
                }

    log.info(f"Converting conllu file {filename} to a dataframe...")
    return pd.DataFrame(movie_entry_generator())
