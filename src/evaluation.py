import os
import json
import numpy as np
import logging as log
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from typing import Union
from wordcloud import WordCloud
from globals import DATA_PATH, EXPORT_PATH
from helper import pandas_ndarray_series_to_numpy
from preprocess.dataloader import load_stratified_data
from classifier.base import MultiLabelClassifier
from text_modelling.modelling import BagOfWords, WordEmbeddingModel


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


def plot_wordcloud(feat_names: list[str], importances: Union[list[float], np.array], genre: str, path: str):
    '''
    Plots wordcloud for given feature importances and names. Saved under path/wordclouds/wordcloud_genre.png
    Parameters:
    feat_names: list of str
        List of feature names
    importances: list of float
        List of feature importances
    genre: str
        Genre for which wordcloud is to be plotted
    path: str
        Path that should contain wouldcloud folder
    '''
    path = os.path.join(path, 'wordclouds')
    if not os.path.exists(path):
        os.makedirs(path)

    importance_dict = dict(zip(feat_names, importances))

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(importance_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {genre} Movies")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"wordcloud_{genre}.png"))
    plt.close()


def plot_feature_importances(feat_names: list[str], importances: Union[list[float], np.array], ymax: int, genre: str, path: str):
    '''
    Plots feature importances as bar plots for given feature importances and names. Saved under path/feature_importances/feature_importance_genre.png
    Parameters:
    feat_names: list of str
        List of feature names
    importances: list of float
        List of feature importances
    ymax: int
        Maximum value for y-axis based on maximum feature importance accross all genres
    genre: str
        Genre for which wordcloud is to be plotted
    path: str
        Path that should contain wouldcloud folder
    '''
    path = os.path.join(path, 'feature_importances')
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(figsize=(10, 5))
    plt.bar(feat_names, importances)
    plt.ylim(0, ymax + 0.5)
    plt.title(f"Feature importance for {genre} Movies")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"feature_importance_{genre}.png"))
    plt.close()


def analyse_features(model: MultiLabelClassifier, text_model: Union[BagOfWords, WordEmbeddingModel], path: str, top_k: int = 10):

    feat_impts = []  # contains feature importances for each classifier contained in MultiOutputClassifier
    estimators = model.multi_output_clf_.estimators_
    feature_names = text_model.get_feature_names_out()

    with open(os.path.join(DATA_PATH, 'genres.json'), 'r') as f:
        classes = json.load(f)

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

    for i, (genre, feat_impt) in enumerate(zip(classes, feat_impts)):
        feat_impt = feat_impt[0]

        indices = np.argsort(feat_impt)[::-1][:top_k]
        feat_names = feature_names[indices]
        importances = np.sort(feat_impt)[::-1][:top_k]

        plot_feature_importances(feat_names, importances, np.max(feat_impts), genre, path)
        plot_wordcloud(feat_names, importances, genre, path)


# Wordclouds for each class


# Color words in descriptions according to importance


def plot_qualitative_results(model, text_model, X, y, y_pred, n_samples: int = 5):
    # Get descriptions for which predictions are very bad/ very good

    pass


def evaluate(model: MultiLabelClassifier, text_model: Union[BagOfWords, WordEmbeddingModel], y, y_pred):

    clf_name = type(model.multi_output_clf_.estimators_[0]).__name__
    text_model_name = type(text_model.model).__name__
    dir_path = os.path.join(EXPORT_PATH, f"evluation_{clf_name}_{text_model_name}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log.info(f"Evaluating model {type(model.multi_output_clf_.estimators_[0]).__name__}")

    metrics = get_metrics(y, y_pred)
    log.info(f"Metrics: {metrics}")

    analyse_features(model, text_model, dir_path)


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

    evaluate(clf, BOW, y_dev, y_pred)


if __name__ == "__main__":
    main()
