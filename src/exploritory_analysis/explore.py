import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


def check_valid_duplicates(df_og: pd.DataFrame, group_by: str, target: str):
    logging.info(f'Checking duplicates for {
                 group_by} with different {target}...')

    df = df_og.copy()
    df['is_duplicate'] = df.duplicated(subset=group_by, keep=False)
    duplicates = df[df['is_duplicate'] == True].copy()
    duplicates.groupby(group_by)
    duplicates.loc[:, 'valid'] = duplicates.groupby(
        [group_by])[target].nunique().eq(1)
    error_duplicates = duplicates[duplicates['valid'] == False]

    if len(error_duplicates) > 0:
        logging.info(f'Duplicate {group_by} with different {target}:')
        logging.info(error_duplicates)
    else:
        logging.info(f'No duplicate {group_by} with different {target} found.')


def analyse_data(raw_data: pd.DataFrame, clean_data: pd.DataFrame):
    check_valid_duplicates(raw_data, 'movie_id', 'genre')
    check_valid_duplicates(raw_data, 'movie_id', 'description')
    check_valid_duplicates(raw_data, 'movie_name', 'genre')
    check_valid_duplicates(raw_data, 'movie_name', 'description')
    check_valid_duplicates(raw_data, 'movie_name', 'movie_id')
