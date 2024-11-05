import itertools
import numpy as np
import logging as log
import re
import os
import pandas as pd

from tqdm import tqdm

# from src.preprocess.imdb_crawler import IMDBCrawler

DATA_PATH = "data"
CRAWL_DATA_PATH = os.path.join(DATA_PATH, 'crawl_data.csv')
# Genres that have very little cardinality
EXCLUDED_GENRES = ["Reality-TV", "News",
                   "Adult", "Talk-Show", "Game-Show", "Short"]
USED_COLS = ["movie_id", "movie_name", "description", "genre"]
TEXT_LEN_CUTOFF = 25  # Remove short descriptions?
PATTERNS = [
    r'Add a Plot',  # Some missing data crawled through IMDb API
    r'\bunder wrap',
    r'plot.*unknown',
    r'plot.*undisclosed',
    r'in development',
    r'^no plot',
    r'\bTBA\b',
    r'^coming soon',
    r'^(?:not available)|(?:(?:plot|outline|details|story).*not available)',
    r'^(?:not disclosed)|(?:(?:plot|outline|details|story).*not disclosed)',
    r'(?:plot|story|synopsis).*unavailable'
]


def merge_with_crawl_data(df_merged):
    df_crawl = pd.read_csv(os.path.join("data", "crawl_data.csv")).dropna()

    # Merging df1 and df2 on 'movie_id' to bring in descriptions from df2
    merged_df = df_merged.merge(
        df_crawl, on='movie_id', how='left', suffixes=('', '_new'))

    # Replacing 'description' in df1 with 'description_new' from df2 where it exists
    merged_df['description'] = merged_df['description_new'].combine_first(
        merged_df['description'])

    # Dropping the temporary 'description_new' column
    merged_df = merged_df.drop(columns=['description_new'])

    return merged_df


def description_separator(descs):
    # Remove duplicates, filter out 'Add a Plot', and select the longest description
    descs = [desc for desc in set(descs) if desc != 'Add a Plot']
    if not descs:
        return 'Add a Plot'

    # Select the longest description
    longest_desc = max(descs, key=len)
    return longest_desc


def description_cleaner(df, pattern):
    # Count matches
    count_matches = df['description'].str.contains(
        pattern, na=False, case=False).sum()

    # Filter out matching rows
    df_filtered = df[~df['description'].str.contains(
        pattern, na=False, case=False)]

    # Log the count of matches
    log.info('Matched descriptions for pattern "%s": %d',
             pattern, count_matches)

    return df_filtered, count_matches


def clean_data(df, save_intermediate=False):
    df_clean = df.copy()

    df_clean = df_clean[USED_COLS]
    df_clean.dropna(inplace=True)
    df_clean = df_clean[~df_clean['genre'].str.contains(
        '|'.join(EXCLUDED_GENRES), case=False, na=False)]

    tqdm.pandas()

    df_merged = df_clean.groupby(["movie_id", "movie_name"], as_index=False).agg({
        "genre": lambda x: ', '.join(x.unique()),
        "description": lambda x: description_separator(x)
    })

    df_merged['genre'] = df_merged['genre'].progress_apply(
        lambda x: np.sort(list(set([s.strip() for s in x.split(", ")]))))

    if os.path.isfile(CRAWL_DATA_PATH):
        df_merged = merge_with_crawl_data(df_merged)

    total_missing_count = 0
    # Get rid of incomplete or unwanted descriptions
    for pattern in PATTERNS:
        df_merged, count = description_cleaner(df_merged, pattern)
        total_missing_count += count

    if save_intermediate:
        # TODO: implement save of merged file which is needed for crawler
        pass

    # TODO: add method that merged crawled data with df_merged
    # call_method_that_does_this_inplace()

    log.info('Missing description rows: %s', total_missing_count)

    # Clean up the end of descriptions in one go
    # Some incomplete data crawled through IMDb API
    patt = r'\.{3,} *See full (?:summary|synopsis) »$'
    count_matches = df_merged['description'].str.contains(
        patt, regex=True).sum()

    df_merged['description'] = df_merged['description'].str.replace(
        patt, '', regex=True).str.strip()

    # Remove desc author signature
    df_merged['description'] = df_merged['description'].str.replace(
        r'—.*$', '', regex=True).str.strip()

    log.info('Incomplete description rows: %s', count_matches)

    genres = np.sort(df_merged['genre'].explode().unique())

    def __encode_genres(genre):
        return np.isin(genres, genre).astype(int)

    df_merged["encoded_genre"] = df_merged.apply(
        lambda x: __encode_genres(x["genre"]), axis=1)

    df_merged = df_merged.drop_duplicates(subset='description', keep='first')
    df_merged = df_merged.drop_duplicates(subset='movie_id', keep='first')
    # df_merged[df_merged['description'] >= TEXT_LEN_CUTOFF]

    log.info('Genres: \n%s', genres)
    log.info('Shape: %s', df_clean.shape)
    log.info('Cleaned data: \n%s', df_clean.head())

    return df_merged
