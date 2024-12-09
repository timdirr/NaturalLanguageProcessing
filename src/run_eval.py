import os
import json
import numpy as np
import logging as log
from tqdm import tqdm

from typing import Union
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


def evaluate(X, ypred, ytrue, classifier, text_model, lemmatized, features):
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
    classifier_name = type(classifier.multi_output_clf_.estimators_[0]).__name__
    model_name = type(text_model.model).__name__

    dir_path = prepare_evaluate(classifier_name, model_name)
    metrics = compute_metrics(ytrue, ypred, metrics_names=['jaccard', 'hamming', 'precision', 'recall', 'at_least_one', 'at_least_two'])
    metrics["lemmatized"] = lemmatized
    log.info(f"Metrics:\n {metrics}")

    if features:
        analyse_features(classifier, text_model, path=dir_path)

    plot_metrics_per_genre(ytrue, ypred, classifier, metrics_names=['balanced_accuracy', 'precision', 'recall'], path=dir_path)

    print(type(X[0]))
    plot_bad_qualitative_results(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_good_qualitative_results(X, ytrue, ypred, classifier, text_model, path=dir_path)
    plot_cfm(ytrue, ypred,  path=dir_path)

    if classifier.multi_output_clf_.estimators_[0].__class__.__name__ == "DecisionTreeClassifier":
        plot_decision_tree(classifier, text_model, path=dir_path)

    with open(os.path.join(dir_path, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=4)

"""
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
    
"""


def fit_predict(classifier, text_model, lemmatized=False):

    # load stratified data, but keep only test set
    _, _, dev = load_stratified_data()

    # depending on model and classifier, lemmatized input data may be beneficial
    if lemmatized:
        X_dev, y_dev = dev["lemmatized_description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])
    else:
        X_dev, y_dev = dev["description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])

    k_fold = MultilabelStratifiedKFold(
            n_splits=5, random_state=SEED, shuffle=True)

    predictions = None
    ground_truth = None
    X = None
    for train_idx, test_idx in tqdm(k_fold.split(X_dev, y_dev)):
        X_dev_train, X_dev_test, y_dev_train, y_dev_test = X_dev[train_idx], X_dev[test_idx], y_dev[train_idx], y_dev[test_idx]
        transformed_data = text_model.fit_transform(X_dev_train)
        classifier = classifier.fit(transformed_data, y_dev_train)

        y_pred = classifier.predict_at_least_1(text_model.transform(X_dev_test))

        if predictions is None:
            predictions = y_pred
        else:
            predictions = np.vstack((predictions, y_pred))

        if ground_truth is None:
            ground_truth = y_dev_test
        else:
            ground_truth = np.vstack((ground_truth, y_dev_test))

        if X is None:
            X = X_dev_test
        else:
            X = np.concatenate((X, X_dev_test),  axis=0)

    return X, predictions, ground_truth, classifier, text_model


def run_eval(predict=True, eval=True):
    X, y_pred, y_true, classifier, text_model = fit_predict(MultiLabelClassifier("lreg", n_jobs=-1), BagOfWords("count", ngram_range=(1, 1)), lemmatized=True)
    evaluate(X, y_pred, y_true, classifier, text_model, lemmatized=False,  features=True)
    
    X, y_pred, y_true, classifier, text_model = fit_predict(MultiLabelClassifier("knn", n_jobs=-1), BagOfWords("count", ngram_range=(1, 1)), lemmatized=True)
    evaluate(X, y_pred, y_true, classifier, text_model, lemmatized=True,  features=False)
    X, y_pred, y_true, classifier, text_model = fit_predict(MultiLabelClassifier("svm"), BagOfWords("count", ngram_range=(1, 1)), lemmatized=True)
    evaluate(X, y_pred, y_true, classifier, text_model, lemmatized=True,  features=False)

    X, y_pred, y_true, classifier, text_model = fit_predict(MultiLabelClassifier("lreg", n_jobs=-1), BagOfWords("tf-idf", ngram_range=(1, 1)), lemmatized=True)
    evaluate(X, y_pred, y_true, classifier, text_model, lemmatized=True,  features=True)
    X, y_pred, y_true, classifier, text_model = fit_predict(MultiLabelClassifier("knn", n_jobs=-1), BagOfWords("tf-idf", ngram_range=(1, 1)), lemmatized=True)
    evaluate(X, y_pred, y_true, classifier, text_model, lemmatized=True,  features=False)
    X, y_pred, y_true, classifier, text_model = fit_predict(MultiLabelClassifier("svm"), BagOfWords("tf-idf", ngram_range=(1, 1)), lemmatized=True)
    evaluate(X, y_pred, y_true, classifier, text_model, lemmatized=True,  features=False)


    #comparative_evaluation(BagOfWords("tf-idf", ngram_range=(1, 1)))


if __name__ == "__main__":
    run_eval()
