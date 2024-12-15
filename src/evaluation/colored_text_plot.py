# ADAPTED FROM: https://matplotlib.org/3.1.0/api/transformations.html#matplotlib.transforms.offset_copy
import json
import logging as log
import os
from typing import Union
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np

from classifier.base import MultiLabelClassifier
from evaluation.utils import get_feature_importances
from globals import DATA_PATH, EXPORT_PATH
from helper import pandas_ndarray_series_to_numpy, load_genres
from preprocess.dataloader import load_stratified_data
from text_modelling.modelling import BagOfWords, WordEmbeddingModel

log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def plot_colored_description(description_words, predicted_genres, vector, text_model: Union[BagOfWords, WordEmbeddingModel],
                             size=18, ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """

    if ax is None:
        ax = plt.gca()
    t = None
    canvas = ax.figure.canvas
    for word in description_words:
        color = get_color_by_score(get_score_by_word(word, vector=vector, text_model=text_model))
        text = ax.text(0, canvas.figure.bbox.bounds[3] - size, word + " ", color=color, size=size, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()

        if (ex.transformed(ax.transData.inverted()).xmax) > 1:
            t = transforms.offset_copy(
                text.get_transform(), x=-lastx, y=-ex.height, units='dots')
            ...
        else:
            t = transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')

        lastx = ex.xmax

    t = None
    for genre in predicted_genres:
        text = ax.text(0, size, genre + " ", color='black', size=size, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')


def get_score_by_word(word: np.ndarray, vector,
                      text_model: Union[BagOfWords, WordEmbeddingModel]):
    feature_names = text_model.get_feature_names_out()
    # get the index of the word in the ndarray feature names
    word_index = np.argwhere(feature_names == word)
    if len(word_index) == 0:
        return 0
    if vector[word_index[0]] < 0:
        return 0
    return (vector[word_index[0]] / np.max(vector))[0]


def get_score_vector_by_genres(feat_impts, all_genres, predicted_genres):
    score_vector = np.zeros(len(feat_impts[0]))

    for genre, feat_impt in zip(all_genres, feat_impts):
        if genre in predicted_genres:
            score_vector += feat_impt
    return score_vector


def get_color_by_score(score):
    '''
    Returns a color based on the score (score between 0 and 1)

    Inputs:
    -------
    score: float
        The score of the word (between 0 and 1)
    Returns:
    --------
    color: str
        The color of the word
    '''
    if score < 0.1:
        return plt.cm.Reds(0.1)
    return plt.cm.Reds(score)


def save_colored_descriptions(clf, text_model, descriptions, predicted_genres_list, path, good_example=True):
    feat_impts = get_feature_importances(clf)
    all_genres = load_genres()

    if len(feat_impts) == 0:
        log.warning("Model does not have attribute for feature importance. Cannot plot colored descriptions.")
        return

    for i, (description, predicted_genres) in enumerate(zip(descriptions, predicted_genres_list)):
        vector = get_score_vector_by_genres(feat_impts, all_genres, predicted_genres)
        words = description.split()
        plt.rcParams['figure.subplot.left'] = 0
        plt.rcParams['figure.subplot.bottom'] = 0
        plt.rcParams['figure.subplot.right'] = 1
        plt.rcParams['figure.subplot.top'] = 1
        plt.figure(figsize=(10, 10))
        plot_colored_description(words, predicted_genres, size=30, vector=vector, text_model=text_model)

        plt.axis('off')
        if good_example:
            plt.savefig(os.path.join(path, f"good_description_{i}.png"), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(path, f"bad_description_{i}.png"), bbox_inches='tight')
        plt.close()
