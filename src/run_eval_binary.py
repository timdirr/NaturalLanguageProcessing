import os
import json
import numpy as np
import logging as log
from sklearn.linear_model import LogisticRegression

from typing import Union
from helper import load_genres
from preprocess.dataloader import load_stratified_data
from preprocess.data_manager import DataManager
from classifier.base import MultiLabelClassifier
from text_modelling.modelling import BagOfWords, WordEmbeddingModel

from evaluation.metrics import compute_metrics
from evaluation.plotting import plot_bad_qualitative_results_binary, plot_bad_qualitative_results_binary, plot_good_qualitative_results_binary, plot_metrics_per_length_binary, plot_cfm
from evaluation.utils import prepare_evaluate
from imblearn.over_sampling import SMOTE

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
             balancing_ratio=None):
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

    dir_path = prepare_evaluate(classifier_name, model_name, model, genre=genre, balancing_ratio=balancing_ratio)
    metrics = compute_metrics(ytrue, ypred, metrics_names=['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy'])
    metrics["lemmatized"] = model.lemmatized
    log.info(f"Metrics:\n {metrics}")

    plot_metrics_per_length_binary(X, ytrue, ypred, path=dir_path, metric='f1')

    plot_bad_qualitative_results_binary(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_good_qualitative_results_binary(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_cfm(ytrue, ypred,  path=dir_path)

    with open(os.path.join(dir_path, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=4)


def fit_predict_binary_for_genre(classifier, text_model, manager: DataManager, genre="Comedy", balancing_ratio=None, dev=True):
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
    transformed_data = text_model.fit_transform(X_train)

    if balancing_ratio:
        smote = SMOTE(random_state=42, sampling_strategy=balancing_ratio)
        genres = load_genres()
        log.info(f"Resampling")
        log.info(f"Num positive samples: {np.sum(y_train)}")
        log.info(f"Num total sumples: {len(y_train)}")
        if np.sum(y_train) > (balancing_ratio/(1 + balancing_ratio)) * len(y_train):
            log.info("Skipping SMOTE")
        else:
            transformed_data, y_train = smote.fit_resample(transformed_data, y_train)
        log.info(f"Num positive samples after resampling: {np.sum(y_train)}")
        log.info(f"Num total sumples after resampling: {len(y_train)}")

    log.info(f"Genre: {genre}")
    log.info(f"Length train set: {len(y_train)}")
    log.info(f"Length test set: {len(y_test)}")

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
                                                                                      balancing_ratio=0.5)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, genre=genre, balancing_ratio=0.5)

    # testing once with unbalanced data
    X, y_pred, y_true, classifier, text_model, manager = fit_predict_binary_for_genre(LogisticRegression(),
                                                                                      BagOfWords("count", ngram_range=(1, 1)),
                                                                                      DataManager(lemmatized=True, prune=False),
                                                                                      genre=genre,
                                                                                      balancing_ratio=None)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, genre=genre, balancing_ratio=None)


if __name__ == "__main__":
    # get genre from command line args
    run_eval(genre="War")
