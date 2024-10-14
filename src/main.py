from src.dataloader import load_raw_data
import itertools
import numpy as np

USED_COLS = ["movie_id", "movie_name", "description", "genre"]


def clean_data(df):
    df_clean = df.copy()

    df_clean = df_clean[USED_COLS]
    df_clean.dropna(inplace=True)

    df_merged = df_clean.groupby(["movie_id", "movie_name"], as_index=False).agg({
        "genre": lambda x: ', '.join(x.unique()),
        "description": "first"
    })

    df_merged.drop(df[df["description"] == ' Add a Plot'].index, inplace=True)

    genres = np.array(list(itertools.chain(*[genre.split(',') for genre in df_clean["genre"].unique()])))
    strip = np.vectorize(lambda x: x.strip(' '))

    genres = np.unique(strip(genres))

    def __encode_genres(genre):
        return np.isin(genres, genre).astype(int)

    df_merged["encoded_genre"] = df_merged.apply(lambda x: __encode_genres(x["genre"]), axis=1)

    return df_merged


def main():
    df = load_raw_data()
    df_clean = clean_data(df)

    del df


if __name__ == '__main__':
    main()