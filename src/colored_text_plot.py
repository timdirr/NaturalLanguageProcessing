# ADAPTED FROM: https://matplotlib.org/3.1.0/api/transformations.html#matplotlib.transforms.offset_copy
import json
import os
from typing import Union
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np

from classifier.base import MultiLabelClassifier
from evaluation.utils import get_feature_importances
from globals import DATA_PATH, EXPORT_PATH
from helper import pandas_ndarray_series_to_numpy
from preprocess.dataloader import load_stratified_data
from text_modelling.modelling import BagOfWords, WordEmbeddingModel


def plot_colored_description(strings, vector, text_model: Union[BagOfWords, WordEmbeddingModel],
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
    last = 0
    first = True
    for s in strings:
        color = get_color_by_score(get_score_by_word(s, vector=vector, text_model=text_model))
        text = ax.text(0, canvas.figure.bbox.bounds[3] - size, s + " ", color=color, size=size, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()

        if (ex.transformed(ax.transData.inverted()).xmax) > 1:
            t = transforms.offset_copy(
                text.get_transform(), x=-lastx, y=-ex.height, units='dots')
            ...
        else:
            t = transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')
        if first:
            firsty = ex.ymax
            firstx = ex.xmax
            first = False
        lastx = ex.xmax
        lasty = ex.ymax


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


def get_score_vector_by_genres(predicted_genres, model: MultiLabelClassifier,
                               text_model: Union[BagOfWords, WordEmbeddingModel]):
    feat_impts = get_feature_importances(model, text_model)
    score_vector = np.zeros(len(feat_impts[0]))
    with open(os.path.join(DATA_PATH, "genres.json"), 'r') as f:
        genres = json.load(f)

    for genre, feat_impt in zip(genres, feat_impts):
        if genre in predicted_genres:
            score_vector += feat_impt

    print(score_vector)
    print(len(score_vector))
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


if __name__ == '__main__':

    BOW = BagOfWords("tf-idf", ngram_range=(1, 1))
    clf = MultiLabelClassifier("lreg")
    _, _, dev = load_stratified_data()
    X_dev, y_dev = dev["description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])

    X_transformed = BOW.fit_transform(X_dev)
    clf.fit(X_transformed, y_dev)
    y_pred = clf.predict(X_transformed)

    vector = get_score_vector_by_genres(['Horror'], clf, BOW)

    words = "A year after saving her friend's life and destroying a killer, Mary Sotherland and her friends rent a lake house in upstate New York for the 4th of July. However, Mary tries hard not to let her paranoia get the best of her. When her friends start being discovered dead and mutilated, Mary has to track down the deranged killer once and for all.".split()

    plt.rcParams['figure.subplot.left'] = 0
    plt.rcParams['figure.subplot.bottom'] = 0
    plt.rcParams['figure.subplot.right'] = 1
    plt.rcParams['figure.subplot.top'] = 1
    plt.figure(figsize=(10, 5))
    plot_colored_description(words, size=30, vector=vector, text_model=BOW)

    plt.axis('off')
    plt.savefig(os.path.join(EXPORT_PATH, 'testpic.png'), bbox_inches='tight', pad_inches=0)
