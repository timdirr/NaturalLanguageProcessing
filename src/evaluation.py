import os
import numpy as np
from sklearn.metrics import jaccard_score, hamming_loss, accuracy_score, f1_score, precision_score, recall_score
import logging as log
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from globals import DATA_PATH, EXPORT_PATH
from helper import pandas_ndarray_series_to_numpy
from preprocess.dataloader import load_stratified_data
from classifier.base import MultiLabelClassifier
from text_modelling.modelling import BagOfWords, WordEmbeddingModel
import json
from typing import Union

log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def get_metrics(y, y_pred, metrics_names: list[str] = ['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall', 'at_least_one']):
    '''
    Get metrics for multilabel classification
    Parameters:
    y: ndarray of shape (n_samples, n_classes)
        True labels
    y_pred: ndarray of shape (n_samples, n_classes)
        Predicted labels
    metrics: list of str
        List of metrics to compute. Default: ['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall']
    '''

    metrics = {}
    if 'jaccard' in metrics_names:
        metrics['jaccard'] = jaccard_score(y, y_pred, average='samples')
    if 'hamming' in metrics_names:
        metrics['hamming'] = hamming_loss(y, y_pred)
    if 'accuracy' in metrics_names:
        metrics['accuracy'] = accuracy_score(y, y_pred)
    if 'f1' in metrics_names:
        metrics['f1'] = f1_score(y, y_pred, average='samples')
    if 'precision' in metrics_names:
        metrics['precision'] = precision_score(y, y_pred, average='samples')
    if 'recall' in metrics_names:
        metrics['recall'] = recall_score(y, y_pred, average='samples')
    if 'at_least_one' in metrics_names:
        metrics['at_least_one'] = at_lest_k(y, y_pred, 1)

    return metrics


def at_lest_k(y, y_pred, k: int = 1):
    '''
    Compute the fraction of samples for which at least one label is predicted correctly
    Parameters:
    y: ndarray of shape (n_samples, n_classes)
        True labels
    y_pred: ndarray of shape (n_samples, n_classes)
        Predicted labels
    '''
    return np.mean(np.any(y & y_pred, axis=1))


def plot_metric_per_genre(y, y_pred, metrics: list[str]):
    pass


def plot_feature_importance(model: MultiLabelClassifier, text_model: Union[BagOfWords, WordEmbeddingModel], top_k: int = 10):
    feat_impts = []  # contains feature importances for each classifier contained in MultiOutputClassifier
    estimators = model.multi_output_clf_.estimators_
    feature_names = text_model.get_feature_names_out()
    log.info(f"Feature names shape: {feature_names.shape}")

    log.info(f"Number of Estimators: {len(estimators)}")

    with open(os.path.join(DATA_PATH, 'genres.json'), 'r') as f:
        classes = json.load(f)

    log.info(f"Classes: {classes}")
    log.info(f"Classes: {type(classes)}")
    log.info(f"Classes length: {len(classes)}")

    if hasattr(estimators[0], 'coef_'):
        for clf in estimators:
            # ndarray of shape (1, n_features)
            feat_impts.append(clf.coef_)
        log.info(f"Feature importance Shape: {feat_impts[0].shape}")

    elif hasattr(estimators[0], 'feature_importances_'):
        for clf in estimators:
            # ndarray of shape (n_features,)
            feat_impts.append(clf.feature_importances_)
    else:
        raise Warning(
            "Model does not have attribute for feature importance. Returning None")

    fig, axs = plt.subplots(7, 3, figsize=(70, 30))
    for i, (cls, feat_impt) in enumerate(zip(classes, feat_impts)):
        # cls is a string
        # feat_impt is an ndarray of shape (1, n_features)
        # get feature names with highets improtance
        feat_impt = feat_impt[0]

        indices = np.argsort(feat_impt)[::-1][:top_k]
        feat_names = feature_names[indices]
        importances = np.sort(feat_impt)[::-1][:top_k]

        axs[i//3, i % 3].bar(feat_names, importances)
        axs[i//3, i % 3].set_xlabel('Feature importance', fontsize=20)
        axs[i//3, i % 3].set_title(f'Feature importance for classifier {i}')
        axs[i//3, i % 3].set_title(cls)
        axs[i//3, i % 3].set_ylim(0, np.max(importances) + 0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_PATH, 'feature_importance_test.png'))


# Wordclouds for each class


# Color words in descriptions according to importance


def plot_qualitative_results(model, text_model, X, y, y_pred, n_samples: int = 5):
    # Get descriptions for which predictions are very bad/ very good

    pass


def evaluate(model: MultiLabelClassifier, text_model, y, y_pred):

    log.info(f"Evaluating model {type(model.multi_output_clf_.estimators_[0]).__name__}")

    metrics = get_metrics(y, y_pred)
    log.info(f"Metrics: {metrics}")

    plot_feature_importance(model, text_model)


def main():
    BOW = BagOfWords("tf-idf", ngram_range=(1, 1))
    clf = MultiLabelClassifier("lreg")
    _, _, dev = load_stratified_data()
    X_dev, y_dev = dev["description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])
    print(y_dev)

    X_transformed = BOW.fit_transform(X_dev)
    clf.fit(X_transformed, y_dev)
    # y_pred = clf.predict(X_transformed)

    y_pred = np.zeros((len(y_dev), 21), dtype=int)
    for i in range(3):
        random_indices = np.random.choice(21, size=3, replace=False)
        y_pred[i, random_indices] = 1

    print(y_dev.shape)
    print(y_pred.shape)

    evaluate(clf, BOW, y_dev, y_pred)


if __name__ == "__main__":
    main()
