import os
import json
import numpy as np
import logging as log
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from typing import Union
from globals import DATA_PATH, EXPORT_PATH, SEED, MODEL_PATH,  UNIQUE_GENRES
from helper import pandas_ndarray_series_to_numpy, load_genres
from preprocess.dataloader import load_stratified_data
from preprocess.data_manager import DataManager
from classifier.base import MultiLabelClassifier
from classifier.dl import MovieGenreClassifier
from text_modelling.modelling import BagOfWords, WordEmbeddingModel, Word2VecModel

from evaluation.metrics import compute_metrics, score_per_sample
from evaluation.plotting import plot_bad_qualitative_results_binary, plot_bad_qualitative_results_binary, plot_decision_tree, plot_feature_importances, plot_good_qualitative_results_binary, plot_metrics_per_length_binary, plot_wordcloud, plot_bad_qualitative_results, plot_good_qualitative_results, plot_cfm, plot_metrics_per_genre, plot_metrics_per_length, plot_metrics_per_genre_distribution
from evaluation.utils import get_feature_importances, prepare_evaluate

from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def analyse_features(clf: MultiLabelClassifier,
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
    feat_impts = get_feature_importances(clf)
    classes = load_genres()

    for i, (genre, feat_impt) in enumerate(zip(classes, feat_impts)):
        indices = np.argsort(feat_impt)[::-1][:top_k]
        feat_names = feature_names[indices]
        importances = np.sort(feat_impt)[::-1][:top_k]

        plot_feature_importances(feat_names, importances, np.max(feat_impts), genre, path)
        plot_wordcloud(feat_names, importances, genre, path)


def evaluate(X: np.ndarray,
             ypred: np.ndarray,
             ytrue: np.ndarray,
             classifier: MultiLabelClassifier,
             text_model: Union[BagOfWords, WordEmbeddingModel],
             model: DataManager,
             features: bool):
    '''
    Wrapper function to analyse a trained model. Computes a set of matrices and plots
    Args:
        :param X: the set of input features used for the prediction
        :param ypred: predicted labels
        :param ytrue: ground truth labels
        :param classifier: the fitted classifier object
        :param text_model: the fitted text model
        :param lemmatized: (boolean) if lemmatized text was used
        :param features: (boolean) if we want to analyse the feature importance. Only applicable for certain models
        :return:
    '''

    classifier_name = type(classifier).__name__
    model_name = type(text_model.model).__name__

    dir_path = prepare_evaluate(classifier_name, model_name, model, binary=True)
    metrics = compute_metrics(ytrue, ypred, metrics_names=['precision', 'recall', 'f1', 'accuracy'])
    metrics["lemmatized"] = model.lemmatized
    log.info(f"Metrics:\n {metrics}")

    plot_metrics_per_length_binary(X, ytrue, ypred, path=dir_path, metric='f1')

    plot_bad_qualitative_results_binary(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_good_qualitative_results_binary(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_cfm(ytrue, ypred,  path=dir_path)

    with open(os.path.join(dir_path, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=4)


def fit_predict_binary_for_genre(classifier, text_model, manager: DataManager, genre="Comedy", fine_tune=False):
    # load stratified data
    _, test, dev = load_stratified_data()

    genres = load_genres()

    genre_idx = genres.index(genre)

    manager.test = test
    manager.dev = dev
    X_dev, y_dev = manager.dev
    X_test, y_test = manager.test
    y_test = y_test[:, genre_idx]
    y_dev = y_dev[:, genre_idx]

    # Balance the dataset
    y_dev_pos = np.argwhere(y_dev == 1)
    y_dev_neg = np.argwhere(y_dev == 0)
    y_dev_neg = y_dev_neg[:len(y_dev_pos)]

    y_test_pos = np.argwhere(y_test == 1)
    y_test_neg = np.argwhere(y_test == 0)
    y_test_neg = y_test_neg[:len(y_test_pos)]

    y_dev = y_dev[np.concatenate([y_dev_pos, y_dev_neg]).flatten()]
    X_dev = X_dev[np.concatenate([y_dev_pos, y_dev_neg]).flatten()]
    y_test = y_test[np.concatenate([y_test_pos, y_test_neg]).flatten()]
    X_test = X_test[np.concatenate([y_test_pos, y_test_neg]).flatten()]

    log.info(f"Genre: {genre}")
    log.info(f"Length balanced train set: {len(y_dev)}")
    log.info(f"Length balanced test set: {len(y_test)}")

    transformed_data = text_model.fit_transform(X_dev)
    classifier = classifier.fit(transformed_data, y_dev)
    y_pred = classifier.predict(text_model.transform(X_test))

    return X_test, y_pred, y_test, classifier, text_model, manager


def run_eval(predict=True, eval=True):

    # model = Word2VecModel(min_count=1)
    # model.load_pretrained('word2vec-google-news-300')

    X, y_pred, y_true, classifier, text_model, manager = fit_predict_binary_for_genre(LogisticRegression(),
                                                                                      BagOfWords("count", ngram_range=(1, 1)),
                                                                                      DataManager(lemmatized=True, prune=False))
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    X, y_pred, y_true, classifier, text_model, manager = fit_predict_binary_for_genre(LogisticRegression(),
                                                                                      BagOfWords("count", ngram_range=(1, 1)),
                                                                                      DataManager(lemmatized=True, prune=True))
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    """
    X, y_pred, y_true, classifier, text_model = fit_predict(
        MovieGenreClassifier(
            model_name="distilbert-base-uncased", unique_genres=UNIQUE_GENRES, num_labels=len(UNIQUE_GENRES),
            seed=SEED),
        BagOfWords("count", ngram_range=(1, 1)),
        lemmatized=False,
        fine_tune=False)
    """
    # evaluate(X, y_pred, y_true, classifier, text_model, lemmatized=False,  features=False)

    # comparative_evaluation(BagOfWords("tf-idf", ngram_range=(1, 1)))


if __name__ == "__main__":
    run_eval()
