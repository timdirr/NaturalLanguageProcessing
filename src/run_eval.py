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
from sklearn.preprocessing import MaxAbsScaler

from evaluation.metrics import compute_metrics, score_per_sample
from evaluation.plotting import plot_decision_tree, plot_feature_importances, plot_wordcloud, plot_bad_qualitative_results, plot_good_qualitative_results, plot_cfm, plot_metrics_per_genre, plot_metrics_per_length, plot_metrics_per_genre_distribution
from evaluation.utils import get_feature_importances, prepare_evaluate

from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def plot_pred_hist(y_pred, path=None):
    log.info("Plotting prediction histogram...")
    import matplotlib.pyplot as plt
    genres = load_genres()
    genre_counts = np.sum(y_pred, axis=0)
    # sort by genre count
    genre_counts, genres = zip(*sorted(zip(genre_counts, genres), reverse=True))
    plt.figure(figsize=(10, 6))
    plt.bar(genres, genre_counts)
    plt.xlabel('Genres')
    plt.ylabel('Number of Predictions')
    plt.title('Number of Predictions for Each Genre')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'pred_hist.png'))
    plt.close()


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
             features: bool,
             plots: bool = False):
    '''
    Wrapper function to analyse a trained model. Computes a set of matrices and plots
    Args:
        :param X: the set of input features used for the prediction
        :param ypred: predicted labels
        :param ytrue: ground truth labels
        :param classifier: the fitted classifier object, can
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

    dir_path = prepare_evaluate(classifier_name, model_name, model, balancing_ratio=classifier.balancing_ratio)
    log.info(f"Saved evaluation results in: {dir_path}")
    log.info("Computing metrics for complete predictions...")
    log.info("Averaging should be samples")
    metrics = compute_metrics(ytrue, ypred, metrics_names=['jaccard', 'hamming', 'precision', 'recall', 'at_least_one', 'at_least_two'])
    metrics["lemmatized"] = model.lemmatized
    log.info(f"Metrics:\n {metrics}")
    plot_pred_hist(ypred, path=dir_path)
    if plots:
        log.info(f"Plots will be saved in {dir_path}")
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


def fit_predict(classifier, text_model, manager: DataManager, fine_tune=False, dev=True, at_least_one=True, custom_pred=False):
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
        scaler = MaxAbsScaler()
        transformed_data = scaler.fit_transform(transformed_data)
        classifier = classifier.fit(transformed_data, y_train)

        if at_least_one:
            y_pred = classifier.predict_at_least_1(scaler.transform(text_model.transform(X_test)), custom_pred, X_test)
        else:
            y_pred = classifier.predict(scaler.transform(text_model.transform(X_test)), custom_pred, X_test)

    return X_test, y_pred, y_test, classifier, text_model, manager


def run_eval(predict=True, eval=True, dev=True):
    # baseline lreg tf-idf w/o lemmatized
    print("Baseline lreg tf-idf w/o lemmatized")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=None, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        BagOfWords("tf-idf", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=True),
        dev=dev, at_least_one=True)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True, plots=True)

    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=None, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        BagOfWords("tf-idf", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=False),
        dev=dev, at_least_one=True)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True, plots=False)

    # baseline lreg count w/o lemmatized
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=None, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        Word2VecModel(),
        DataManager(lemmatized=True, prune=False),
        dev=dev, at_least_one=True)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True, plots=False)

    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=None, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        Word2VecModel(),
        DataManager(lemmatized=True, prune=False),
        dev=dev, at_least_one=True)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True, plots=False)

    # baseline lreg count
    print("Baseline lreg count")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=None, solver='lbfgs', max_iter=1000),
        BagOfWords("count", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=False),
        dev=dev, at_least_one=False)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    #################################################################################################################################
    # lreg only
    #################################################################################################################################

    # atleast1 tf-idf
    print("atleast1 tf-idf")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=None, solver='lbfgs', max_iter=1000),
        BagOfWords("tf-idf", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=False),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    # atleast1 count
    print("atleast1 count")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=None, solver='lbfgs', max_iter=1000),
        BagOfWords("count", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=False),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    ##################################################################################################################################

    # atleast1, prune tf-idf
    print("atleast1, prune tf-idf")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=None, solver='lbfgs', max_iter=1000),
        BagOfWords("tf-idf", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=True),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    # atleast1, prune count
    print("atleast1, prune count")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=None, solver='lbfgs', max_iter=1000),
        BagOfWords("count", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=True),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    ##################################################################################################################################

    # atleast1, balancing tf-idf 0.5
    print("atleast1, balancing tf-idf 0.5")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=0.5, solver='lbfgs', max_iter=1000),
        BagOfWords("tf-idf", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=False),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    # atleast1, balancing count 0.5
    print("atleast1, balancing count 0.5")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=0.5, solver='lbfgs', max_iter=1000),
        BagOfWords("count", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=False),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    #################################################################################################################################

    # atleast1, balancing tf-idf 0.75
    print("atleast1, balancing tf-idf 0.75")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=0.75, solver='lbfgs', max_iter=1000),
        BagOfWords("tf-idf", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=False),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    # atleast1, balancing count 0.75
    print("atleast1, balancing count 0.75")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=0.75, solver='lbfgs', max_iter=1000),
        BagOfWords("count", ngram_range=(1, 1)),
        DataManager(lemmatized=True, prune=False),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    ##################################################################################################################################

    # full tf-idf, 0.5 balancing
    print("full tf-idf 0.5")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=0.5, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        BagOfWords("tf-idf", ngram_range=(1, 1), stop_words='english', sublinear_tf=True),
        DataManager(lemmatized=True, prune=True),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    # full count, 0.5 balancing
    print("full count 0.5")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=0.5, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        BagOfWords("count", ngram_range=(1, 1), stop_words='english'),
        DataManager(lemmatized=True, prune=True),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    # full tf-idf, 0.75 balancing
    print("full tf-idf 0.75")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=0.75, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        BagOfWords("tf-idf", ngram_range=(1, 1), stop_words='english', sublinear_tf=True),
        DataManager(lemmatized=True, prune=True),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)

    # full count, 0.75 balancing
    print("full count 0.75")
    X, y_pred, y_true, classifier, text_model, manager = fit_predict(
        MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=0.75, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        BagOfWords("count", ngram_range=(1, 1), stop_words='english'),
        DataManager(lemmatized=True, prune=True),
        dev=dev)
    evaluate(X, y_pred, y_true, classifier, text_model, manager, features=True)


if __name__ == "__main__":
    run_eval(dev=False)


'''
No balacing + drama
{'jaccard': 0.3322474935345591, 'hamming': 0.08811472298559131, 'precision': 0.5174823750309987, 'recall': 0.3854793283026889, 'at_least_one': 0.6229142310553725, 'at_least_two': 0.16101604846423637, 'lemmatized': False}
{'jaccard': 0.37536987802292304, 'hamming': 0.09376787169326228, 'precision': 0.5341656790322961, 'recall': 0.4900635915966982, 'at_least_one': 0.7310553725156764, 'at_least_two': 0.2594855988946753, 'lemmatized': True}

X, y_pred, y_true, classifier, text_model, manager = fit_predict(
    MultiLabelClassifier("lreg", n_jobs=-1, balancing_ratio=0.5, solver='lbfgs', max_iter=1000, class_weight='balanced'),
    BagOfWords("tf-idf", ngram_range=(1, 1)),
    DataManager(lemmatized=True, prune=False),
    dev=dev, at_least_one=False, custom_pred=False)

 {'jaccard': 0.36769923933012466, 'hamming': 0.13541239643907302, 'precision': 0.42865702291119445, 'recall': 0.6797746838133701, 'at_least_one': 0.8896800935274737, 'at_least_two': 0.4476033584865554, 'lemmatized': True}
 {'jaccard': 0.3421701398000295, 'hamming': 0.16740557419694416, 'precision': 0.3792213139951616, 'recall': 0.7581809969178447, 'at_least_one': 0.9302263789988309, 'at_least_two': 0.5382080986289722, 'lemmatized': True}
 
 '''
