import itertools
import numpy as np
import logging as log

USED_COLS = ["movie_id", "movie_name", "description", "genre"]

counter = 0


def description_cleaner(x):
    x = set(x)
    z = [d for d in x if d != 'Add a Plot']
    if len(z) == 0:
        return 'Add a Plot'
    return ';;'.join(z)


def clean_data(df):
    df_clean = df.copy()

    df_clean = df_clean[USED_COLS]
    df_clean.dropna(inplace=True)

    df_merged = df_clean.groupby(["movie_id", "movie_name"], as_index=False).agg({
        "genre": lambda x: ', '.join(x.unique()),
        "description": lambda x: description_cleaner(x)
    })

    log.info('Number of movies with more than 1 description: %s',
             df_merged[df_merged["description"].str.contains(';;')].shape[0])

    empty_description_rows = df_merged[df_merged["description"]
                                       == 'Add a Plot']
    log.info('Empty description rows: \n%s', empty_description_rows)
    df_merged.drop(empty_description_rows.index, inplace=True)

    genres = np.array(list(itertools.chain(
        *[genre.split(',') for genre in df_clean["genre"].unique()])))
    strip = np.vectorize(lambda x: x.strip(' '))

    genres = np.unique(strip(genres))

    def __encode_genres(genre):
        return np.isin(genres, genre).astype(int)

    df_merged["encoded_genre"] = df_merged.apply(
        lambda x: __encode_genres(x["genre"]), axis=1)

    log.info('Genres: \n%s', genres)
    log.info('Shape: %s', df_clean.shape)
    log.info('Cleaned data: \n%s', df_clean.head())

    return df_merged
