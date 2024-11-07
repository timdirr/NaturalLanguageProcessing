import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging as log
import os
from globals import DATA_PATH, EXPORT_PATH

from exploritory_analysis.description_exploration import plot_description_length


def check_valid_duplicates(df_og: pd.DataFrame, group_by: str, target: str):
    log.info(f"""Checking duplicates for {
        group_by} with different {target}...""")

    df = df_og.copy()
    df.set_index(group_by, inplace=True)
    df = df[[target]]

    grouped = df.groupby([group_by])[target].apply(
        lambda x: [i for i in x]).to_frame(name=target)
    grouped.loc[:, 'number_different_values'] = grouped[target].apply(
        lambda x: len(set(x)))
    grouped.loc[:, target] = grouped[target].apply(lambda x: [[i] for i in x])

    error_duplicates = grouped[grouped['number_different_values'].ne(1)].copy()
    error_duplicates.sort_values(
        by='number_different_values', ascending=False, inplace=True)

    if len(error_duplicates) > 0:
        # export dataframe to csv
        error_duplicates.to_csv(
            os.path.join(EXPORT_PATH, f'error_duplicates_{group_by}_{target}.csv'), index=True)
        log.info(f"""Error duplicates found and saved in export folder (Length: {
                 len(error_duplicates)}).""")
    else:
        log.info(f'No duplicate {group_by} with different {target} found.')


def analyse_data(raw_data: pd.DataFrame):
    log.info('------------------------------------')


    check_valid_duplicates(raw_data, 'movie_id', 'genre')
    check_valid_duplicates(raw_data, 'movie_id', 'description')
    check_valid_duplicates(raw_data, 'movie_name', 'genre')
    check_valid_duplicates(raw_data, 'movie_name', 'description')
    check_valid_duplicates(raw_data, 'movie_name', 'movie_id')

    log.info(
        f"""Number of rows with "See full summary" in description: {raw_data[raw_data['description'].str.contains(
            'See full summary')].shape[0]}""")

    plot_description_length(raw_data)
