import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os


def check_valid_duplicates(df_og: pd.DataFrame, group_by: str, target: str):
    logging.info(f'Checking duplicates for {
                 group_by} with different {target}...')

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
        logging.info(f'Duplicate {group_by} with different {target}:')
        logging.info(f'\n {error_duplicates}')
        logging.info(f'Total: {len(error_duplicates)}')
        # export dataframe to csv
        error_duplicates.to_csv(
            os.path.join('export', f'error_duplicates_{group_by}_{target}.csv'), index=True)

    else:
        logging.info(f'No duplicate {group_by} with different {target} found.')


def analyse_data(raw_data: pd.DataFrame, clean_data: pd.DataFrame):
    logging.info('------------------------------------')
    check_valid_duplicates(raw_data, 'movie_id', 'genre')
    check_valid_duplicates(raw_data, 'movie_id', 'description')
    check_valid_duplicates(raw_data, 'movie_name', 'genre')
    check_valid_duplicates(raw_data, 'movie_name', 'description')
    check_valid_duplicates(raw_data, 'movie_name', 'movie_id')

    df = raw_data[raw_data['description'].str.contains(
        'See full summary')].copy()
    print(df.head())
    print(df.shape)

    raw_data['description_length'] = raw_data['description'].apply(
        lambda x: len(x.split()))
    plt.figure(figsize=(12, 6))
    plt.hist(raw_data['description_length'], bins=50, color='blue')
    plt.xlabel('Description Length (in words)')
    plt.ylabel('Frequency')
    plt.title('Description Length Distribution')
    plt.savefig(os.path.join('export', 'description_length_distribution.png'))

    raw_data['description_length_char'] = raw_data['description'].apply(
        lambda x: len(x))
    plt.figure(figsize=(12, 6))
    plt.hist(raw_data['description_length_char'], bins=100, color='blue')
    plt.xlabel('Description Length (in characters)')
    plt.ylabel('Frequency')
    plt.title('Description Length Distribution')
    plt.savefig(os.path.join(
        'export', 'description_length_char_distribution.png'))
