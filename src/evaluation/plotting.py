import os
import pandas as pd
from typing import Union
import logging as log
from sklearn.metrics import recall_score, jaccard_score
from sklearn import tree
from wordcloud import WordCloud

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import textwrap
from evaluation.metrics import compute_metrics, confusion_matrix, score_per_sample
from evaluation.colored_text_plot import save_colored_descriptions
from helper import decode_genres, load_genres
from text_modelling.modelling import BagOfWords, WordEmbeddingModel
from classifier.base import MultiLabelClassifier
import sklearn


log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


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


def plot_metrics_per_genre(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           metrics_names: list[str],
                           path: str):
    '''
    Plots metrics per genre as bar plots. Saved under path/metrics_per_genre.png

    Args:
        y_true (np.ndarray): Ground truth (binary matrix, shape [n_samples, n_classes]).
        y_pred (np.ndarray): Predictions (binary matrix, shape [n_samples, n_classes]).
        model (MultiLabelClassifier): Model for which feature importances are to be computed.
        metrics_names (list[str]):  List of metrics to compute. Default: ['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall'].
        path (str): Path to directory to save results in.
    '''
    path = os.path.join(path, 'metrics_per_genre')
    if not os.path.exists(path):
        os.makedirs(path)

    if 'at_least_one' in metrics_names:
        metrics_names.remove('at_least_one')
    if 'at_least_two' in metrics_names:
        metrics_names.remove('at_least_two')

    genres = load_genres()

    metrics = defaultdict(list)
    genre_occurences = []

    for i, genre in enumerate(genres):
        y_true_genre = y_true[:, i]
        y_pred_genre = y_pred[:, i]
        metrics_genre = compute_metrics(y_true_genre, y_pred_genre, metrics_names)
        metrics[genre] = metrics_genre
        genre_occurences.append(np.sum(y_true_genre))

    genre_occurences = np.array(genre_occurences) / np.sum(genre_occurences)
    # turn into dataframe containing columns: genre, [metrics_names]
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Genre'})
    metrics_df = metrics_df.melt(id_vars=['Genre'], var_name='Metric', value_name='Value')
    metrics_df = metrics_df.pivot(index='Genre', columns='Metric', values='Value').reset_index()
    metrics_df['genre_occurences'] = genre_occurences
    metrics_df = metrics_df.sort_values(by='balanced_accuracy', ascending=False)

    fig, axs = plt.subplots(1, 1, figsize=(40, 15))
    metrics_df.plot(
        x='Genre',
        kind='bar',
        stacked=False,
        title='Metrics per Genre',
        ax=axs,
    )

    plt.xticks(fontsize=20, rotation=45, ha='right')  # Increased fontsize and rotated labels
    plt.xlabel('Genre', fontsize=25)
    plt.title('Metrics per Genre (Sorted by balanced Accuracy)', fontsize=30)
    plt.legend(title='Metrics', fontsize=20, title_fontsize=25)

    plt.tight_layout()

    plt.savefig(os.path.join(path, 'metrics_per_genre.png'))
    plt.close()


def plot_good_qualitative_results(X,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  clf: MultiLabelClassifier,
                                  text_model: Union[BagOfWords, WordEmbeddingModel],
                                  n_samples: int = 30,
                                  path: str = None):

    path = os.path.join(path, "qualitative_results")
    if not os.path.exists(path):
        os.makedirs(path)
    # extract predictions with very good performance
    metrics = np.array([jaccard_score(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
    good_indices = np.argsort(metrics)[-n_samples:]
    # get descriptions
    descriptions = X[good_indices, ]

    if not isinstance(descriptions[0], str):
        descriptions = np.array(list(map(' '.join, descriptions)))

    true_genres = [decode_genres(y_true[i]) for i in good_indices]
    predicted_genres = [decode_genres(y_pred[i]) for i in good_indices]
    results = pd.DataFrame({
        "Description": descriptions,
        "True Labels": true_genres,
        "Predicted Labels": predicted_genres,
    })
    try:
        save_table_as_image(results, os.path.join(path, "good_qualitative_results.png"))
    except:
        log.error("Error saving table as image")
    results.to_csv(os.path.join(path, "good_qualitative_results.csv"), index=False)
    try:
        save_colored_descriptions(clf, text_model, descriptions, predicted_genres, path)
    except:
        log.error("Error saving colored descriptions")


def plot_bad_qualitative_results(X,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 clf: MultiLabelClassifier,
                                 text_model: Union[BagOfWords, WordEmbeddingModel],
                                 n_samples: int = 30,
                                 path: str = None):

    path = os.path.join(path, "qualitative_results")
    if not os.path.exists(path):
        os.makedirs(path)
    # extract predictions with very bad performance
    metrics = np.array([jaccard_score(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
    bad_indices = np.argsort(metrics)[:n_samples]
    # get descriptions
    descriptions = X[bad_indices, ]

    if not isinstance(descriptions[0], str):
        descriptions = np.array(list(map(' '.join, descriptions)))

    true_genres = [decode_genres(y_true[i]) for i in bad_indices]
    predicted_genres = [decode_genres(y_pred[i]) for i in bad_indices]
    results = pd.DataFrame({
        "Description": descriptions,
        "True Labels": true_genres,
        "Predicted Labels": predicted_genres,
    })

    try:
        save_table_as_image(results, os.path.join(path, "bad_qualitative_results.png"))
    except:
        log.error("Error saving table as image")

    results.to_csv(os.path.join(path, "bad_qualitative_results.csv"), index=False)
    try:
        save_colored_descriptions(clf, text_model, descriptions, predicted_genres, path, good_example=False)
    except:
        log.error("Error saving colored descriptions")


def plot_cfm(y_true: np.ndarray, y_pred: np.ndarray, path: str = None):
    cfm = confusion_matrix(y_true, y_pred)
    num_labels = cfm.shape[0]
    genres = load_genres()
    path = os.path.join(path, "confusion_matrices")
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(num_labels):
        cm = cfm[i]
        genre = genres[i]
        plt.figure(figsize=(6, 6))
        plt.matshow(cm, cmap='Blues', fignum=1)

        for (x, y), value in np.ndenumerate(cm):
            plt.text(y, x, f"{value}", va='center', ha='center', fontsize=42, color='black')

        plt.title(f"Confusion Matrix for genre {genre}", fontsize=18, weight='bold', pad=20)
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Actual", fontsize=14)
        plt.xticks(range(2), labels=["0", "1"], fontsize=12)
        plt.yticks(range(2), labels=["0", "1"], fontsize=12)
        plt.grid(False)
        plt.savefig(os.path.join(path, f'confusion_matrix_{genre}.png'))
        plt.close()


def plot_decision_tree(clf: MultiLabelClassifier, text_model: Union[BagOfWords, WordEmbeddingModel], path: str = None):
    path = os.path.join(path, "decision_tree")
    genres = load_genres()
    if not os.path.exists(path):
        os.makedirs(path)
    feature_names = text_model.get_feature_names_out()
    estimators = clf.multi_output_clf_.estimators_
    # acces axis as iterative
    for genre, estimator in zip(genres, estimators):
        fig, axs = plt.subplots(1, 1, figsize=(30, 30))

        tree.plot_tree(estimator, max_depth=5, ax=axs, feature_names=feature_names, class_names=[genre, f"No {genre}"], filled=True, fontsize=30, impurity=False)

        axs.set_title(f"Decision Tree for {genre}", fontsize=30)

        plt.savefig(os.path.join(path, f"decision_tree_{genre}.png"))
        plt.close()


def plot_metrics_per_length(X: np.ndarray,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            path: str,
                            metric: str = 'jaccard'):
    '''
    Plots metrics per length of the input text. Saved under path/metrics_per_length.png
    '''

    log.info("Plotting metrics per length...")
    path = os.path.join(path, 'metrics_per_length')
    if not os.path.exists(path):
        os.makedirs(path)
    # get lengths of the input text
    lengths = np.array([len(text.split()) for text in X])
    metrics = score_per_sample(y_true, y_pred, metric=metric)

    bins = np.linspace(0, lengths.max(), 10)
    bin_indices = np.digitize(lengths, bins) - 1

    metrics_per_bin = [np.mean(metrics[bin_indices == i]) for i in range(len(bins) - 1)]

    plt.figure(figsize=(10, 5))
    plt.plot(bins[:-1], metrics_per_bin, marker='o')
    plt.xlabel('Length of input text')
    plt.ylabel('Average Jaccard Score')
    plt.title('Metrics per length of input text')
    plt.xticks(bins, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'metrics_per_length.png'))
    plt.close()


def plot_metrics_per_genre_distribution(y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        path: str,
                                        metric: str = 'jaccard'):
    '''
    Plots the distribution of metrics per genre. 
    '''
    log.info("Plotting metrics per genre distribution...")
    path = os.path.join(path, 'metrics_per_genre_distribution')
    if not os.path.exists(path):
        os.makedirs(path)
    genre_counts = np.sum(y_true, axis=0)
    metrics = score_per_sample(y_true, y_pred, metric=metric)
    metrics_per_genre = []
    for i in range(y_true.shape[1]):
        indices = y_true[:, i] == 1
        metrics_per_genre.append(np.mean(metrics[indices]))

    plt.figure(figsize=(10, 5))
    plt.scatter(genre_counts, metrics_per_genre)
    for i, txt in enumerate(load_genres()):
        plt.annotate(txt, (genre_counts[i], metrics_per_genre[i]))
    plt.xlabel('Number of samples per genre')
    plt.ylabel('Average Jaccard Score per genre')
    plt.title('Metrics per genre distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'metrics_per_genre_distribution.png'))
    plt.close()
