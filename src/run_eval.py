import os
import json
import numpy as np
import logging as log
import pandas as pd
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
from evaluation.plotting import plot_decision_tree, plot_feature_importances, plot_wordcloud, plot_bad_qualitative_results, plot_good_qualitative_results, plot_cfm, plot_metrics_per_genre, plot_metrics_per_length, plot_metrics_per_genre_distribution
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
    if type(classifier).__name__ == "MovieGenreClassifier":
        classifier_name = type(classifier).__name__
        model_name = "raw_text"
    else:
        classifier_name = type(classifier.multi_output_clf_.estimators_[0]).__name__
        model_name = type(text_model.model).__name__

    dir_path = prepare_evaluate(classifier_name, model_name, model, balanced_train=classifier.balanced_fitting)
    metrics = compute_metrics(ytrue, ypred, metrics_names=['jaccard', 'hamming', 'precision', 'recall', 'at_least_one', 'at_least_two'])
    metrics["lemmatized"] = model.lemmatized
    log.info(f"Metrics:\n {metrics}")

    if features and classifier_name != "MovieGenreClassifier":
        analyse_features(classifier, text_model, path=dir_path)

    plot_metrics_per_genre(ytrue, ypred, metrics_names=['balanced_accuracy', 'precision', 'recall'], path=dir_path)
    plot_metrics_per_length(X, ytrue, ypred, path=dir_path, metric='jaccard')
    plot_metrics_per_genre_distribution(ytrue, ypred, path=dir_path, metric='jaccard')

    plot_bad_qualitative_results(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_good_qualitative_results(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_cfm(ytrue, ypred,  path=dir_path)

    if classifier_name != "MovieGenreClassifier" and classifier.multi_output_clf_.estimators_[0].__class__.__name__ == "DecisionTreeClassifier":
        plot_decision_tree(classifier, text_model, path=dir_path)

    with open(os.path.join(dir_path, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=4)


def fit_predict(classifier, text_model, manager: DataManager, fine_tune=False, dev=True):
    if dev:
        _, test, train = load_stratified_data()
    else:
        train, test, _ = load_stratified_data()
        manager.train_set = 'full'

    manager.test = test
    manager.train = train
    X_train, y_train = manager.train
    X_test, y_test = manager.test

    if type(classifier).__name__ == "MovieGenreClassifier":
        output_dir = os.path.join(MODEL_PATH, "distilbert_movie_genres")
        X_train = X_train.reshape(-1, 1)
        _X_train, _y_train, X_val, y_val = iterative_train_test_split(X_train, y_train, test_size=0.2)
        train_data = pd.DataFrame({'description': _X_train.reshape(-1), 'genre': pd.Series(list(_y_train))})
        val_data = pd.DataFrame({'description': X_val.reshape(-1), 'genre': pd.Series(list(y_val))})
        test_data = pd.DataFrame({'description': X_test, 'genre': pd.Series(list(y_test))})

        if fine_tune:
            log.info(f'Fine-tuning model (Path: {output_dir})')
            classifier.fine_tune(output_dir=output_dir,
                                 train_data=train_data, eval_data=val_data)
            log.info('Fine-tuning finished')
        classifier.load_model(model_path=os.path.join(output_dir, 'best'))
        log.info(f"Loaded best model (Path: {os.path.join(output_dir, 'best')})")
        y_pred = classifier.predict(test_data)
    else:
        transformed_data = text_model.fit_transform(X_train)
        classifier = classifier.fit(transformed_data, y_train)
        y_pred = classifier.predict_at_least_1(text_model.transform(X_test))

    return X_test, y_pred, y_test, classifier, text_model, manager


def run_eval(predict=True, eval=True, dev=True):
    # X, y_pred, y_true, classifier, text_model, manager = fit_predict(MultiLabelClassifier("lreg", n_jobs=-1, balanced_fitting=False),
    #                                                                  BagOfWords("count", ngram_range=(1, 1)),
    #                                                                  DataManager(lemmatized=True, prune=False), dev=dev)
    # evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    X, y_pred, y_true, classifier, text_model, manager = fit_predict(MultiLabelClassifier("lreg", n_jobs=-1, balanced_fitting=True),
                                                                     BagOfWords("count", ngram_range=(1, 1)),
                                                                     DataManager(lemmatized=True, prune=False), dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)


if __name__ == "__main__":
    run_eval(dev=False)
