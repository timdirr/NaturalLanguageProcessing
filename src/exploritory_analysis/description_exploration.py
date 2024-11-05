import logging as log
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_description_length(df: pd.DataFrame, cleaned: bool = False):
    if cleaned:
        log.info('Plotting cleaned description length distribution...')
    else:
        log.info('Plotting raw description length distribution...')

    df['description_length'] = df['description'].apply(
        lambda x: len(x.split()))
    plt.figure(figsize=(12, 6))
    plt.hist(df['description_length'], bins=80, color='blue', range=(0, 400))
    plt.xlabel('Description Length (in words)')
    plt.ylabel('Frequency')

    if cleaned:
        plt.title('Description Length Distribution (Cleaned)')
        plt.savefig(os.path.join(
            'export', 'description_length_distribution_cleaned.png'))
    else:
        plt.title('Description Length Distribution (Raw)')
        plt.savefig(os.path.join(
            'export', 'description_length_distribution_raw.png'))

    df['description_length_char'] = df['description'].apply(
        lambda x: len(x))
    plt.figure(figsize=(12, 6))
    plt.hist(df['description_length_char'],
             bins=200, color='blue', range=(0, 2000))
    plt.xlabel('Description Length (in characters)')
    plt.ylabel('Frequency')

    if cleaned:
        plt.title('Description Length Distribution (Cleaned)')
        plt.savefig(os.path.join(
            'export', 'description_length_char_distribution_cleaned.png'))
        log.info(
            'Cleaned description length distribution plotted and saved in export folder.')
    else:
        plt.savefig(os.path.join(
            'export', 'description_length_char_distribution_raw.png'))
        plt.title('Description Length Distribution (Raw)')
        log.info(
            'Raw description length distribution plotted and saved in export folder.')
