import itertools
import numpy as np
import logging as log

from tqdm import tqdm

from src.preprocess.imdb_crawler import IMDBCrawler


USED_COLS = ["movie_id", "movie_name", "description", "genre"]

counter = 0


def description_cleaner(x):
    x = set(x)
    z = [d for d in x if d != 'Add a Plot']
    if len(z) == 0:
        return 'Add a Plot'
    return ';;'.join(z)


def clean_data(df, save_intermediate=False):
    df_clean = df.copy()

    df_clean = df_clean[USED_COLS]
    df_clean.dropna(inplace=True)

    tqdm.pandas()

    df_merged = df_clean.groupby(["movie_id", "movie_name"], as_index=False).agg({
        "genre": lambda x: ', '.join(x.unique()),
        "description": lambda x: description_cleaner(x)
    })

    df_merged['genre'] = df_merged['genre'].progress_apply(
        lambda x: np.sort(list(set([s.strip() for s in x.split(", ")]))))

    log.info('Number of movies with more than 1 description: %s',
             df_merged[df_merged["description"].str.contains(';;')].shape[0])

    if save_intermediate:
        # TODO: implement save of merged file which is needed for crawler
        pass

    # TODO: add method that merged crawled data with df_merged
    # call_method_that_does_this_inplace()

    empty_description_rows = df_merged[df_merged["description"]
                                       == 'Add a Plot']
    log.info('Empty description rows: \n%s', empty_description_rows)
    df_merged.drop(empty_description_rows.index, inplace=True)

    genres = np.sort(df_merged['genre'].explode().unique())

    def __encode_genres(genre):
        return np.isin(genres, genre).astype(int)

    df_merged["encoded_genre"] = df_merged.apply(
        lambda x: __encode_genres(x["genre"]), axis=1)

    log.info('Genres: \n%s', genres)
    log.info('Shape: %s', df_clean.shape)
    log.info('Cleaned data: \n%s', df_clean.head())

    return df_merged
