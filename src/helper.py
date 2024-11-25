import re
import pandas as pd
import numpy as np


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
