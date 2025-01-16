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

from globals import EXPORT_PATH

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
             balancing_ratio=None,
             plot=False):
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

    if plot:
        plot_metrics_per_length_binary(X, ytrue, ypred, path=dir_path, metric='f1')

        plot_bad_qualitative_results_binary(X, ytrue, ypred, classifier, text_model, path=dir_path)
        plot_good_qualitative_results_binary(X, ytrue, ypred, classifier, text_model, path=dir_path)
        plot_cfm(ytrue, ypred,  path=dir_path)

    with open(os.path.join(dir_path, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=4)


def fit_predict_binary_for_genre(classifier, text_model, manager: DataManager, genre="Comedy", balancing_ratio=None, dev=False, custom_clf=False):
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
    log.info(y_pred[:100])

    if custom_clf:
        # Assign 1 if "police", "murder" or "crime" in the description
        # load top words for genre from file
        with open(os.path.join(EXPORT_PATH, f"top_words_{genre}.json"), 'r') as file:
            keywords = json.load(file)

        y_frac_keywords = np.array([])
        for description in X_test:
            num_keywords = sum(1 for keyword in keywords if keyword in description.lower())
            y_frac_keywords = np.append(y_frac_keywords, num_keywords / len(keywords) if keywords else 0)

        y_prob = classifier.predict_proba(text_model.transform(X_test))
        y_prob = y_prob[:, 1] + y_frac_keywords
        y_pred = np.where(y_prob > 0.5, 1, 0)

        # y_custom_pred = np.array([1 if any(keyword in description.lower() for keyword in keywords) else 0 for description in X_test])
        # set prediction to 1 when one of both is one
        log.info(y_pred)

    return X_test, y_pred, y_test, classifier, text_model, manager


def run_eval(predict=True, eval=True, genre=None):

    # model = Word2VecModel(min_count=1)
    # model.load_pretrained('word2vec-google-news-300')

    # testing once with balanced data
    X, y_pred, y_true, classifier, text_model, manager = fit_predict_binary_for_genre(LogisticRegression(),
                                                                                      BagOfWords("count", ngram_range=(1, 1)),
                                                                                      DataManager(lemmatized=True, prune=False),
                                                                                      genre=genre,
                                                                                      balancing_ratio=None,
                                                                                      custom_clf=False)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, genre=genre, balancing_ratio=None, plot=False)

    X, y_pred, y_true, classifier, text_model, manager = fit_predict_binary_for_genre(LogisticRegression(),
                                                                                      BagOfWords("count", ngram_range=(1, 1)),
                                                                                      DataManager(lemmatized=True, prune=False),
                                                                                      genre=genre,
                                                                                      balancing_ratio=None,
                                                                                      custom_clf=True)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, genre=genre, balancing_ratio=None, plot=False)


if __name__ == "__main__":
    # get genre from command line args
    run_eval(genre="Romance")


'''
CRIME
 {'accuracy': 0.8540227441811032, 'balanced_accuracy': 0.6794911798594101, 'f1': 0.4768615501809179, 'precision': 0.5422260718925942, 'recall': 0.42556084296397007, 'lemmatized': True}
 {'accuracy': 0.8415347008183653, 'balanced_accuracy': 0.7108584048469452, 'f1': 0.5067813430367185, 'precision': 0.49355670103092786, 'recall': 0.5207341944255608, 'lemmatized': True}'

WAR:
 {'accuracy': 0.9710915081305134, 'balanced_accuracy': 0.7124116794117752, 'f1': 0.4981549815498155, 'precision': 0.5818965517241379, 'recall': 0.43548387096774194, 'lemmatized': True}
 {'accuracy': 0.9701349771495377, 'balanced_accuracy': 0.743855138281768, 'f1': 0.5253378378378378, 'precision': 0.5514184397163121, 'recall': 0.5016129032258064, 'lemmatized': True}

ROMANCE:
  {'accuracy': 0.8386119672653842, 'balanced_accuracy': 0.7010645390568548, 'f1': 0.5372543044339478, 'precision': 0.6209933075026418, 'recall': 0.4734156820622986, 'lemmatized': True}
  {'accuracy': 0.8295780635561696, 'balanced_accuracy': 0.7287077244683763, 'f1': 0.5660938979840346, 'precision': 0.5704935914916826, 'recall': 0.5617615467239527, 'lemmatized': True}
'''
