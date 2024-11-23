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


def pandas_ndarray_series_to_numpy(series: pd.Series):
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
    if isinstance(series.iloc[0], list):
        return np.vstack(series)

    assert isinstance(series.iloc[0], np.ndarray), "Series must contain ndarrays."
    return np.vstack([x.tolist() for x in series.to_numpy()])
