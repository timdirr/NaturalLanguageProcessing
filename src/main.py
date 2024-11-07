from preprocess.dataloader import load_raw_data
from preprocess.dataclean import clean_data
from preprocess.tokenizer import tokenize
import exploritory_analysis.raw_data_exploration as raw_data_exploration
import exploritory_analysis.clean_data_exploration as clean_data_exploration
import logging as log
import pandas as pd
import os
import argparse
import re
import json

from globals import DATA_PATH, EXPORT_PATH, LOGGING, DATA_EXPLORATION, CONFIG_PATH

OUTPUT_PATH = os.path.join(DATA_PATH, "clean_data.csv")
RAW_PATH = os.path.join(DATA_PATH, "raw_data.csv")


def check_file_exists(file_path, description):
    if os.path.exists(file_path):
        return True
    else:
        print(f"{description} is not available.")
        return False


def main():

    # check if path exeists

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)

        tokenize = config['tokenize']
        assert tokenize in [True, False], "Invalid value for tokenize"

        explore = config['explore']
        assert explore in ["raw", "clean", "full", False,
                           "None", None], "Invalid value for explore"

        preprocess = config['preprocess']
        assert preprocess in [True, False], "Invalid value for preprocess"

        store_intermediate = config['store_intermediate']
        assert store_intermediate in [
            True, False], "Invalid value for store_intermediate"

        verbose = config['verbose']
        assert verbose in [True, False], "Invalid value for verbose"
    else:
        raise FileNotFoundError("Config file not found")

    if verbose:
        log.basicConfig(level=log.INFO,
                        format='%(asctime)s: %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    else:
        log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    if preprocess:
        df_raw = load_raw_data()
        df_raw.to_csv(RAW_PATH, index=False, quoting=1)
        if store_intermediate:
            df_clean = clean_data(df_raw, save_intermediate=True)
        else:
            df_clean = clean_data(df_raw)
        df_clean.to_csv(OUTPUT_PATH, index=False, quoting=1)

        log.info(f"Cleaned data saved to {OUTPUT_PATH}")

    if explore or explore != "None":
        if (explore == "raw" or explore == "full") and check_file_exists(RAW_PATH, "Raw data"):
            df_raw = pd.read_csv(RAW_PATH)
            raw_data_exploration.analyse_data(df_raw)
        if (explore == "clean" or explore == "full") and check_file_exists(OUTPUT_PATH, "Cleaned data"):
            df_clean = pd.read_csv(OUTPUT_PATH, converters={
                                   "genre": lambda x: re.sub(r"[\[\]']", '', x).split(' ')})
            clean_data_exploration.analyse_data(df_clean)

    if tokenize:
        if check_file_exists(OUTPUT_PATH, "Clean data"):
            tokenize()


if __name__ == '__main__':
    main()
