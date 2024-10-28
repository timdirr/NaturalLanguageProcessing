import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging as log
import os


def plot_genre_distribution(df: pd.DataFrame):
    log.info("Plotting genre distribution...")
    genre_count = df['genre'].apply(lambda x: len(x))


def analyse_data(df: pd.DataFrame):
    print(df.head())
    print(df['encoded_genre'].iloc[1])
    print(df['genre'].iloc[1])
    # plot_genre_distribution(df)
