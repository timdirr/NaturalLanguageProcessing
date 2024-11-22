import glob
import itertools
import pandas as pd
import os
import logging as log
import numpy as np
from globals import DATA_PATH, SPLIT_FOLDER, TEST_FILE, TRAIN_FILE, DEV_FILE
from download import download
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
    log.info(f"Checking if all initial data is loaded...")
    download()
    log.info(f"Loading stratified data splits...")
    SPLIT_FOLDER_PATH = os.path.join(DATA_PATH, SPLIT_FOLDER)
    TEST_FILE_PATH = os.path.join(SPLIT_FOLDER_PATH, TEST_FILE)
    TRAIN_FILE_PATH = os.path.join(SPLIT_FOLDER_PATH, TRAIN_FILE)
    DEV_FILE_PATH = os.path.join(SPLIT_FOLDER_PATH, DEV_FILE)

    if os.path.exists(SPLIT_FOLDER_PATH) and os.path.isfile(TEST_FILE_PATH) and os.path.isfile(TRAIN_FILE_PATH) and os.path.isfile(DEV_FILE_PATH):
        test = __load_csv(TEST_FILE_PATH)
        train = __load_csv(os.path.join(DATA_PATH, SPLIT_FOLDER, TRAIN_FILE))
        dev = __load_csv(DEV_FILE_PATH)
        log.info(f"Splits found and loaded.")
        return train, test, dev

    log.info(f"No splits found, creating stratified splits...")
    os.makedirs(SPLIT_FOLDER_PATH, exist_ok=True)
    df = conllu2df('conllu_data.conllu')
    y = np.vstack(df["genre"].to_numpy()).astype(int)
    X = df.drop(["genre"], axis=1).values

    log.info(f"Creating train-test split with ration 0.1")
    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X, y, 0.1)

    log.info(f"Creating stratified dev set as subset with ration 0.1")
    _, _, X_dev, y_dev = iterative_train_test_split(X_train, y_train, 0.1)

    test = pd.DataFrame(X_test, columns=df.drop(["genre"], axis=1).columns)
    train = pd.DataFrame(X_train, columns=df.drop(["genre"], axis=1).columns)
    dev = pd.DataFrame(X_dev, columns=df.drop(["genre"], axis=1).columns)

    test["genre"] = y_test.tolist()
    dev["genre"] = y_dev.tolist()
    train["genre"] = y_train.tolist()

    __save_csv(TEST_FILE_PATH, test)
    __save_csv(TRAIN_FILE_PATH, train)
    __save_csv(DEV_FILE_PATH, dev)

    log.info(f"Stratified splits created and saved.")
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
                    current_genre = np.array(
                        re.sub(r"[\[\]']", '', current_genre).split(' '))

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
