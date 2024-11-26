import os
import json
import numpy as np
import pandas as pd
import logging as log
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from typing import Union
from wordcloud import WordCloud
from globals import DATA_PATH, EXPORT_PATH
from helper import pandas_ndarray_series_to_numpy
from preprocess.dataloader import load_stratified_data
from classifier.base import MultiLabelClassifier
from text_modelling.modelling import BagOfWords, WordEmbeddingModel
from preprocess.dataclean import decode_genres

from evaluation.metrics import compute_metrics, single_signed_overlap
from evaluation.plotting import plot_feature_importances, plot_wordcloud, save_table_as_image
from evaluation.utils import get_feature_importances

from sklearn.metrics import recall_score


log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def plot_metric_per_genre(y, y_pred, metrics: list[str]):
    pass


def analyse_features(model: MultiLabelClassifier,
                     text_model: Union[BagOfWords, WordEmbeddingModel],
                     top_k: int = 10,
                     path: str = None):
    '''
    Analyse features for given model and text model. Saves wordclouds and feature importances for each genre under path.

    Args:
        model (MultiLabelClassifier): Model for which feature importances are to be computed.

        text_model (Union[BagOfWords, WordEmbeddingModel]): Text model used to vectorize text data.

        path (str): Path to directory to save results in.

        top_k (int): Number of top features to consider for each genre. Default: 10.
    '''

    feature_names = text_model.get_feature_names_out()
    feat_impts = get_feature_importances(model, text_model)
    with open(os.path.join(DATA_PATH, "genres.json"), 'r') as f:
        classes = json.load(f)

    for i, (genre, feat_impt) in enumerate(zip(classes, feat_impts)):
        indices = np.argsort(feat_impt)[::-1][:top_k]
        feat_names = feature_names[indices]
        importances = np.sort(feat_impt)[::-1][:top_k]

        plot_feature_importances(feat_names, importances, np.max(feat_impts), genre, path)
        plot_wordcloud(feat_names, importances, genre, path)


# Color words in descriptions according to importance

def plot_bad_qualitative_results(X: np.ndarray,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 n_samples: int = 10,
                                 path: str = None):

    path = os.path.join(path, "qualitative_results")
    if not os.path.exists(path):
        os.makedirs(path)
    # extract predictions with very bad performance
    metrics = np.array([recall_score(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
    bad_indices = np.argsort(metrics)[:n_samples]
    print("Bad indices: ", bad_indices)
    # get descriptions
    descriptions = X[bad_indices]

    true_genres = [decode_genres(y_true[i]) for i in bad_indices]
    predicted_genres = [decode_genres(y_pred[i]) for i in bad_indices]
    results = pd.DataFrame({
        "Description": descriptions,
        "True Labels": true_genres,
        "Predicted Labels": predicted_genres,
    })
    save_table_as_image(results, os.path.join(path, "bad_qualitative_results.png"))
    # save to csv
    results.to_csv(os.path.join(path, "bad_qualitative_results.csv"), index=False)


def plot_good_qualitative_results(X: np.ndarray,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,

                                  n_samples: int = 10,
                                  path: str = None,
                                  top_k: int = 10):

    path = os.path.join(path, "qualitative_results")
    if not os.path.exists(path):
        os.makedirs(path)
    # extract predictions with very good performance
    metrics = np.array([recall_score(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
    good_indices = np.argsort(metrics)[-n_samples:]
    print("Good indices: ", good_indices)
    # get descriptions
    descriptions = X[good_indices]

    true_genres = [decode_genres(y_true[i]) for i in good_indices]
    predicted_genres = [decode_genres(y_pred[i]) for i in good_indices]
    results = pd.DataFrame({
        "Description": descriptions,
        "True Labels": true_genres,
        "Predicted Labels": predicted_genres,
    })
    save_table_as_image(results, os.path.join(path, "good_qualitative_results.png"))
    # save to csv
    results.to_csv(os.path.join(path, "good_qualitative_results.csv"), index=False)


def evaluate(model: MultiLabelClassifier,
             text_model: Union[BagOfWords, WordEmbeddingModel],
             X,
             y_true,
             y_pred):

    clf_name = type(model.multi_output_clf_.estimators_[0]).__name__
    text_model_name = type(text_model.model).__name__
    dir_path = os.path.join(EXPORT_PATH, f"evluation_{clf_name}_{text_model_name}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log.info(f"Evaluating model {type(model.multi_output_clf_.estimators_[0]).__name__}")

    metrics = compute_metrics(y_true, y_pred)
    log.info(f"Metrics: {metrics}")

    analyse_features(model, text_model, path=dir_path)
    plot_bad_qualitative_results(X, y_true, y_pred, path=dir_path)
    plot_good_qualitative_results(X, y_true, y_pred, path=dir_path)


def main():
    BOW = BagOfWords("tf-idf", ngram_range=(1, 1))
    clf = MultiLabelClassifier("lreg")
    _, _, dev = load_stratified_data()
    X_dev, y_dev = dev["description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])

    X_transformed = BOW.fit_transform(X_dev)
    clf.fit(X_transformed, y_dev)
    y_pred = clf.predict(X_transformed)

    # y_pred = np.zeros((len(y_dev), 21), dtype=int)
    # for i in range(3):
    #    random_indices = np.random.choice(21, size=3, replace=False)
    #    y_pred[i, random_indices] = 1

    evaluate(clf, BOW, X_dev, y_dev, y_pred)


if __name__ == "__main__":
    main()
