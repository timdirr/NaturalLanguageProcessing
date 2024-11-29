import os
import json
import time
import numpy as np
import pandas as pd
import logging as log
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Union
from wordcloud import WordCloud
from globals import DATA_PATH, EXPORT_PATH, SEED
from helper import pandas_ndarray_series_to_numpy
from preprocess.dataloader import load_stratified_data
from classifier.base import MultiLabelClassifier
from text_modelling.modelling import BagOfWords, WordEmbeddingModel, Word2VecModel

from evaluation.metrics import compute_metrics, score_per_sample
from evaluation.plotting import plot_decision_tree, plot_feature_importances, plot_wordcloud, plot_bad_qualitative_results, plot_good_qualitative_results, plot_cfm, plot_metrics_per_genre
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
    with open(os.path.join(DATA_PATH, "genres.json"), 'r') as f:
        classes = json.load(f)

    for i, (genre, feat_impt) in enumerate(zip(classes, feat_impts)):
        indices = np.argsort(feat_impt)[::-1][:top_k]
        feat_names = feature_names[indices]
        importances = np.sort(feat_impt)[::-1][:top_k]

        plot_feature_importances(feat_names, importances, np.max(feat_impts), genre, path)
        plot_wordcloud(feat_names, importances, genre, path)


def evaluate(clf: MultiLabelClassifier,
             text_model: Union[BagOfWords, WordEmbeddingModel],
             lemmatized=False,
             features=True,
             ):

    _, _, dev = load_stratified_data()
    if lemmatized:
        X_dev, y_dev = dev["lemmatized_description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])
    else:
        X_dev, y_dev = dev["description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])

    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X_dev[..., np.newaxis], y_dev, 0.1)

    X_train = X_train.squeeze(-1)
    X_test = X_test.squeeze(-1)

    log.info(f"Fitting model")
    X_transformed = text_model.fit_transform(X_train)
    clf.fit(X_transformed, y_train)
    log.info(f"Model fitted.")

    log.info(f"Predicting on test set")
    start_time = time.time()
    y_pred = clf.predict(text_model.transform(X_test))
    log.info(f"Predictions done in: {time.time() - start_time}")

    log.info(f"Predicting on test set (at least 1)")
    start_time = time.time()
    y_pred_at_least_1 = clf.predict_at_least_1(text_model.transform(X_test))
    log.info(f"Predictions (at least 1) done in: {time.time() - start_time}")

    dir_path = prepare_evaluate(clf, text_model)
    metrics = compute_metrics(y_test, y_pred, metrics_names=['jaccard', 'hamming', 'precision', 'recall', 'at_least_one', 'at_least_two'])
    log.info(f"Metrics:\n {metrics}")
    metrics = compute_metrics(y_test, y_pred_at_least_1, metrics_names=['jaccard', 'hamming', 'precision', 'recall', 'at_least_one', 'at_least_two'])
    log.info(f"Metrics (at least 1):\n {metrics}")

    if features:
        analyse_features(clf, text_model, path=dir_path)
    plot_metrics_per_genre(y_test, y_pred, clf, metrics_names=['balanced_accuracy', 'precision', 'recall'], path=dir_path)

    plot_bad_qualitative_results(X_test, y_test, y_pred, clf, text_model, path=dir_path)
    plot_good_qualitative_results(X_test, y_test, y_pred, clf, text_model, path=dir_path)
    plot_cfm(y_test, y_pred,  path=dir_path)
    if clf.multi_output_clf_.estimators_[0].__class__.__name__ == "DecisionTreeClassifier":
        plot_decision_tree(clf, text_model, path=dir_path)


def comparative_evaluation(model, lemmatized=False):
    _, _, dev = load_stratified_data()
    if lemmatized:
        X_dev, y_dev = dev["lemmatized_description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])
    else:
        X_dev, y_dev = dev["description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])

    dir_path = os.path.join(EXPORT_PATH, "comparative")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    def test_model(model, classifier, X_train, X_test, y_train, y_test):
        transformed_data = model.fit_transform(X_train)
        classifier = classifier.fit(transformed_data, y_train)

        jaccard = score_per_sample(y_test, classifier.predict_at_least_1(
            model.transform(X_test)))
        return jaccard

    k_fold = MultilabelStratifiedKFold(
        n_splits=2, random_state=SEED, shuffle=True)

    scores = {"lreg": [], "knn": []}
    for train_idx, test_idx in tqdm(k_fold.split(X_dev, y_dev)):
        X_dev_train, X_dev_test, y_dev_train, y_dev_test = X_dev[train_idx], X_dev[test_idx], y_dev[train_idx], y_dev[test_idx]
        results = test_model(BagOfWords("tf-idf", ngram_range=(1, 1)), MultiLabelClassifier("lreg"), X_dev_train, X_dev_test, y_dev_train, y_dev_test)
        scores["lreg"] = scores["lreg"] + results
        results = test_model(BagOfWords("tf-idf", ngram_range=(1, 1)), MultiLabelClassifier("knn", n_neighbors=15, weights='distance',
                             algorithm='auto', metric='euclidean', n_jobs=-1), X_dev_train, X_dev_test, y_dev_train, y_dev_test)
        scores["knn"] = scores["knn"] + results
        # results = test_model(BagOfWords("tf-idf", ngram_range=(1, 1)), MultiLabelClassifier("svm"), X_dev_train, X_dev_test, y_dev_train, y_dev_test)
        # scores["svm"] = results

    sample_size = 100
    samples = np.arange(1, sample_size + 1)
    random_indices = np.random.choice(len(y_dev), sample_size, replace=False)

    random_sampled_scores = {
        classifier: [score[i] for i in random_indices] for classifier, score in scores.items()
    }

    sort_by = "lreg"
    scores_to_sort = np.array(random_sampled_scores[sort_by])
    sorted_indices = np.argsort(scores_to_sort)

    sampled_scores = {
        classifier: [score[i] for i in sorted_indices] for classifier, score in random_sampled_scores.items()
    }

    plt.figure(figsize=(10, 6))
    for classifier, score in sampled_scores.items():
        plt.plot(samples, score, label=classifier, marker='')

    plt.xlabel('samples')
    plt.ylabel('Jaccard score')
    plt.title('Jaccard score per Sample for Each Classifier')
    plt.legend()
    plt.grid(False)
    plt.savefig(os.path.join(dir_path, f'comparative.png'))


def main():
    evaluate(MultiLabelClassifier("lreg"), BagOfWords("tf-idf", ngram_range=(1, 1)), lemmatized=False, features=True)
    evaluate(MultiLabelClassifier("bayes"), BagOfWords("tf-idf", ngram_range=(1, 1)), lemmatized=False, features=True)
    evaluate(MultiLabelClassifier("dt", max_depth=3), BagOfWords("tf-idf", ngram_range=(1, 1)), lemmatized=False, features=True)
    evaluate(MultiLabelClassifier("knn"), BagOfWords("tf-idf", ngram_range=(1, 1)))
    # evaluate(MultiLabelClassifier("svm"), BagOfWords("tf-idf", ngram_range=(1, 1)))
    # evaluate(MultiLabelClassifier("mlp"), BagOfWords("tf-idf", ngram_range=(1, 1)))
    comparative_evaluation(BagOfWords("tf-idf", ngram_range=(1, 1)))


if __name__ == "__main__":
    main()
