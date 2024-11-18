import glob
import pandas as pd
import os
import logging as log
from globals import DATA_PATH, EXPORT_PATH


def __load_csv(filename):
    return pd.read_csv(filename)


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
