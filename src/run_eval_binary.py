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


def evaluate(X: np.ndarray,
             ypred: np.ndarray,
             ytrue: np.ndarray,
             classifier: MultiLabelClassifier,
             text_model: Union[BagOfWords, WordEmbeddingModel],
             model: DataManager,
             genre: str = None,
             balanced: bool = True):
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

    dir_path = prepare_evaluate(classifier_name, model_name, model, genre=genre, balanced_train=balanced)
    metrics = compute_metrics(ytrue, ypred, metrics_names=['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy'])
    metrics["lemmatized"] = model.lemmatized
    log.info(f"Metrics:\n {metrics}")

    plot_metrics_per_length_binary(X, ytrue, ypred, path=dir_path, metric='f1')

    plot_bad_qualitative_results_binary(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_good_qualitative_results_binary(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_cfm(ytrue, ypred,  path=dir_path)

    with open(os.path.join(dir_path, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=4)


def fit_predict_binary_for_genre(classifier, text_model, manager: DataManager, genre="Comedy", balanced=True, dev=True):
    # load stratified data
    if dev:
        _, test, train = load_stratified_data()
    else:
        train, test, _ = load_stratified_data()
        manager.train_set = 'full'

    genres = load_genres()

    genre_idx = genres.index(genre)

    manager.test = test
    manager.train = train
    X_train, y_train = manager.train
    X_test, y_test = manager.test
    y_test = y_test[:, genre_idx]
    y_train = y_train[:, genre_idx]

    # Balance the train set
    if balanced:
        y_train_pos = np.argwhere(y_train == 1)
        y_train_neg = np.argwhere(y_train == 0)
        y_train_neg = y_train_neg[:len(y_train_pos)]

        y_train = y_train[np.concatenate([y_train_pos, y_train_neg]).flatten()]
        X_train = X_train[np.concatenate([y_train_pos, y_train_neg]).flatten()]

    log.info(f"Genre: {genre}")
    log.info(f"Length train set: {len(y_train)}")
    log.info(f"Length test set: {len(y_test)}")

    transformed_data = text_model.fit_transform(X_train)
    classifier = classifier.fit(transformed_data, y_train)
    y_pred = classifier.predict(text_model.transform(X_test))

    return X_test, y_pred, y_test, classifier, text_model, manager


def run_eval(predict=True, eval=True, genre=None):

    # model = Word2VecModel(min_count=1)
    # model.load_pretrained('word2vec-google-news-300')

    # testing once with balanced data
    X, y_pred, y_true, classifier, text_model, manager = fit_predict_binary_for_genre(LogisticRegression(),
                                                                                      BagOfWords("count", ngram_range=(1, 1)),
                                                                                      DataManager(lemmatized=True, prune=False),
                                                                                      genre=genre,
                                                                                      balanced=True)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, genre=genre)

    # testing once with unbalanced data
    X, y_pred, y_true, classifier, text_model, manager = fit_predict_binary_for_genre(LogisticRegression(),
                                                                                      BagOfWords("count", ngram_range=(1, 1)),
                                                                                      DataManager(lemmatized=True, prune=False),
                                                                                      genre=genre,
                                                                                      balanced=False)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, genre=genre, balanced=False)


if __name__ == "__main__":
    # get genre from command line args
    run_eval(genre="War")
