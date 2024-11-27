import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union
from wordcloud import WordCloud
import textwrap

import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
import textwrap


def save_table_as_image(df, filename="table_image.png"):
    # Wrap text in the 'Description' column
    wrap_width = 40  # Adjust this value as needed
    if 'Description' in df.columns:
        df = df.copy()  # Avoid modifying the original DataFrame
        df['Description'] = df['Description'].apply(
            lambda x: '\n'.join(textwrap.wrap(str(x), width=wrap_width))
        )

    # Calculate row heights based on the number of line breaks
    cell_heights = []
    max_lines = 0
    for i in range(len(df)):
        row_text = df.iloc[i].astype(str).values
        line_counts = [cell_text.count('\n') + 1 for cell_text in row_text]
        max_line_count = max(line_counts)
        cell_heights.append(max_line_count)
        if max_line_count > max_lines:
            max_lines = max_line_count

    # Adjust figure size based on total number of lines
    total_lines = sum(cell_heights)
    fig_height = total_lines * 0.3  # Adjust scaling factor as needed
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")  # Turn off the axis

    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Manually set column widths
    ncols = len(df.columns)
    widths = [1.0 / ncols] * ncols  # Start with equal widths

    # Adjust the width for the 'Description' column
    try:
        desc_col_idx = list(df.columns).index('Description')
        # Set the 'Description' column to take up 50% of the table width
        widths[desc_col_idx] = 0.5  # 50% of the table width

        # Adjust widths of other columns accordingly
        remaining_width = 1.0 - widths[desc_col_idx]
        num_other_cols = ncols - 1
        if num_other_cols > 0:
            other_col_width = remaining_width / num_other_cols
            for idx in range(ncols):
                if idx != desc_col_idx:
                    widths[idx] = other_col_width
    except ValueError:
        # 'Description' column not found; keep default widths
        pass

    # Set the column widths and row heights for each cell
    cells = table.get_celld()
    for (row, col), cell in cells.items():
        cell.set_width(widths[col])
        if row == 0:
            # Header row
            cell.set_text_props(weight='bold')
        else:
            # Adjust cell height based on the number of lines in the cell
            cell_lines = df.iloc[row - 1, col].count('\n') + 1
            cell.set_height(cell_lines * 0.015)  # Adjust scaling factor as needed

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_wordcloud(feat_names: list, importances: Union[list, np.array], genre: str, path: str):
    '''
    Plots wordcloud for given feature importances and names. Saved under path/wordclouds/wordcloud_genre.png
    Parameters:
    feat_names: list of str
        List of feature names
    importances: list of float
        List of feature importances
    genre: str
        Genre for which wordcloud is to be plotted
    path: str
        Path that should contain wouldcloud folder
    '''
    path = os.path.join(path, 'wordclouds')
    if not os.path.exists(path):
        os.makedirs(path)

    importance_dict = dict(zip(feat_names, importances))

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(importance_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {genre} Movies")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"wordcloud_{genre}.png"))
    plt.close()


def plot_feature_importances(feat_names: list, importances: Union[list, np.array], ymax: int, genre: str, path: str):
    '''
    Plots feature importances as bar plots for given feature importances and names. Saved under path/feature_importances/feature_importance_genre.png
    Parameters:
    feat_names: list of str
        List of feature names
    importances: list of float
        List of feature importances
    ymax: int
        Maximum value for y-axis based on maximum feature importance accross all genres
    genre: str
        Genre for which wordcloud is to be plotted
    path: str
        Path that should contain wouldcloud folder
    '''
    path = os.path.join(path, 'feature_importances')
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(figsize=(10, 5))
    plt.bar(feat_names, importances)
    plt.ylim(0, ymax + 0.5)
    plt.title(f"Feature importance for {genre} Movies")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"feature_importance_{genre}.png"))
    plt.close()
