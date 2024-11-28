import re
import pandas as pd
import numpy as np
import json
import os
from globals import DATA_PATH


def numpy_to_pandas(X, columns):
    # TODO: might not work (but should)
    '''
    Converts a numpy array to a pandas DataFrame.

    Parameters:
    ----------
    X : numpy.ndarray
        The numpy array to convert.
    columns : pandas.Columns
        The columns of the DataFrame.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame.
    '''
    return pd.DataFrame(X, columns=columns)


def pandas_ndarray_series_to_numpy(series):
    '''
    Converts a pandas Series of ndarrays to a numpy array.

    Parameters:
    ----------
    series : pandas.Series
        The Series to convert.

    Returns:
    -------
    numpy.ndarray
        The numpy array.
    '''
    return np.array(series.tolist()).astype(int)


def get_genre_converter():
    '''
    Returns a dictionary that can be used to convert genres when loading from a CSV file.
    Returns:
    -------
    dict
        The dictionary that can be used to convert genres when loading from a CSV file.
    '''
    return {"genre": lambda x: re.sub(r"[\[\]']", '', x).split(' ')}


def load_genres():
    with open(os.path.join(DATA_PATH, 'genres.json'), 'r') as f:
        genres = json.load(f)
    return genres


def encode_genres(genre):
    genres = load_genres()
    return np.isin(genres, genre).astype(int)


def decode_genres(encoded_genre):
    '''
    Takes an encoded genre and return the correspondind genres as a list of strings.
    '''
    genres = load_genres()
    indices = np.where(encoded_genre == 1)[0]
    if len(indices) == 0:
        return []
    return [genres[i] for i in indices]
