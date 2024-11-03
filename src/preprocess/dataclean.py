import itertools
import numpy as np
import logging as log
import re

USED_COLS = ["movie_id", "movie_name", "description", "genre"]

counter = 0


def description_cleaner(descs):
    # Remove duplicates, filter out 'Add a Plot', and select the longest description
    descs = [desc for desc in set(descs) if desc != 'Add a Plot']
    if not descs:
        return 'Add a Plot'
    # Select the longest description
    longest_desc = max(descs, key=len)    
    return longest_desc


def clean_data(df):
    df_clean = df.copy()

    df_clean = df_clean[USED_COLS]
    df_clean.dropna(inplace=True)

    df_merged = df_clean.groupby(["movie_id", "movie_name"], as_index=False).agg({
        "genre": lambda x: ', '.join(x.unique()),
        "description": lambda x: description_cleaner(x)
    })

    df_merged['genre'] = df_merged['genre'].apply(
        lambda x: np.sort(list(set([s.strip() for s in x.split(", ")]))))

    log.info('Number of movies with more than 1 description: %s',
             df_merged[df_merged["description"].str.contains(';;')].shape[0])

    empty_description_rows = df_merged[df_merged["description"]
                                       == 'Add a Plot']
    log.info('Empty description rows: \n%s', empty_description_rows)
    df_merged.drop(empty_description_rows.index, inplace=True)

    # Clean up descriptions in one go
    df_merged['description'] = df_merged['description'].str.replace(
        r'\.{3,} *See full (summary|synopsis) »$', '', regex=True
    ).str.strip()
    
    genres = np.sort(df_merged['genre'].explode().unique())

    def __encode_genres(genre):
        return np.isin(genres, genre).astype(int)

    df_merged["encoded_genre"] = df_merged.apply(
        lambda x: __encode_genres(x["genre"]), axis=1)

    log.info('Genres: \n%s', genres)
    log.info('Shape: %s', df_clean.shape)
    log.info('Cleaned data: \n%s', df_clean.head())

    return df_merged
