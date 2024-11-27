import os
import json
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
from colored_text_plot import save_colored_descriptions
from text_modelling.modelling import BagOfWords, WordEmbeddingModel, Word2VecModel
from helper import load_genres, decode_genres

from evaluation.metrics import compute_metrics, single_signed_overlap, confusion_matrix, score_per_sample
from evaluation.plotting import plot_feature_importances, plot_wordcloud, save_table_as_image
from evaluation.utils import get_feature_importances

from sklearn.pipeline import make_pipeline
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from sklearn.metrics import recall_score


log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def plot_metric_per_genre(y, y_pred, metrics: list):
    pass


def analyse_features(model: MultiLabelClassifier,
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
    feat_impts = get_feature_importances(model, text_model)
    with open(os.path.join(DATA_PATH, "genres.json"), 'r') as f:
        classes = json.load(f)

    for i, (genre, feat_impt) in enumerate(zip(classes, feat_impts)):
        indices = np.argsort(feat_impt)[::-1][:top_k]
        feat_names = feature_names[indices]
        importances = np.sort(feat_impt)[::-1][:top_k]

        plot_feature_importances(feat_names, importances, np.max(feat_impts), genre, path)
        plot_wordcloud(feat_names, importances, genre, path)


# Color words in descriptions according to importance

def plot_bad_qualitative_results(X: np.ndarray,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model: MultiLabelClassifier,
                                 text_model: Union[BagOfWords, WordEmbeddingModel],
                                 n_samples: int = 10,
                                 path: str = None):

    path = os.path.join(path, "qualitative_results")
    if not os.path.exists(path):
        os.makedirs(path)
    # extract predictions with very bad performance
    metrics = np.array([recall_score(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
    bad_indices = np.argsort(metrics)[:n_samples]
    print("Bad indices: ", bad_indices)
    # get descriptions
    descriptions = X[bad_indices]

    true_genres = [decode_genres(y_true[i]) for i in bad_indices]
    predicted_genres = [decode_genres(y_pred[i]) for i in bad_indices]
    results = pd.DataFrame({
        "Description": descriptions,
        "True Labels": true_genres,
        "Predicted Labels": predicted_genres,
    })
    save_table_as_image(results, os.path.join(path, "bad_qualitative_results.png"))
    # save to csv
    results.to_csv(os.path.join(path, "bad_qualitative_results.csv"), index=False)
    save_colored_descriptions(model, text_model, descriptions, predicted_genres, path, good_example=False)


def plot_good_qualitative_results(X: np.ndarray,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  model: MultiLabelClassifier,
                                  text_model: Union[BagOfWords, WordEmbeddingModel],
                                  n_samples: int = 10,
                                  path: str = None,
                                  top_k: int = 10):

    path = os.path.join(path, "qualitative_results")
    if not os.path.exists(path):
        os.makedirs(path)
    # extract predictions with very good performance
    metrics = np.array([recall_score(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
    good_indices = np.argsort(metrics)[-n_samples:]
    print("Good indices: ", good_indices)
    # get descriptions
    descriptions = X[good_indices]

    true_genres = [decode_genres(y_true[i]) for i in good_indices]
    predicted_genres = [decode_genres(y_pred[i]) for i in good_indices]
    results = pd.DataFrame({
        "Description": descriptions,
        "True Labels": true_genres,
        "Predicted Labels": predicted_genres,
    })
    save_table_as_image(results, os.path.join(path, "good_qualitative_results.png"))
    # save to csv
    results.to_csv(os.path.join(path, "good_qualitative_results.csv"), index=False)
    # plot colored descriptions
    save_colored_descriptions(model, text_model, descriptions, predicted_genres, path)


def plot_cfm(y_true: np.ndarray, y_pred: np.ndarray, path: str = None):
    cfm = confusion_matrix(y_true, y_pred)
    num_labels = cfm.shape[0]
    genres = load_genres()
    path = os.path.join(path, "conufsion_matrices")
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(num_labels):
        cm = cfm[i]
        genre = genres[i]
        plt.figure(figsize=(6, 6))
        plt.matshow(cm, cmap='Blues', fignum=1)

        for (x, y), value in np.ndenumerate(cm):
            plt.text(y, x, f"{value}", va='center', ha='center', fontsize=42, color='black')

        plt.title(f"Confusion Matrix for genre {genre}", fontsize=18, weight='bold', pad=20)
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Actual", fontsize=14)
        plt.xticks(range(2), labels=["0", "1"], fontsize=12)
        plt.yticks(range(2), labels=["0", "1"], fontsize=12)
        plt.grid(False)
        plt.savefig(os.path.join(path, f'confusion_matrix_{genre}.png'))
        plt.close()


def prepare_evaluate(model: MultiLabelClassifier,
                     text_model: Union[BagOfWords, WordEmbeddingModel]):

    clf_name = type(model.multi_output_clf_.estimators_[0]).__name__
    text_model_name = type(text_model.model).__name__
    dir_path = os.path.join(EXPORT_PATH, f"evluation_{clf_name}_{text_model_name}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log.info(f"Evaluating model {type(model.multi_output_clf_.estimators_[0]).__name__}")

    return dir_path


def evaluate(clf,
             model,
             lemmatized=False,
             features=False,
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

    X_transformed = model.fit_transform(X_train)
    clf.fit(X_transformed, y_train)
    y_pred = clf.predict(model.transform(X_test))

    dir_path = prepare_evaluate(clf, model)
    metrics = compute_metrics(y_test, y_pred, metrics_names=['jaccard', 'hamming', 'precision', 'recall', 'at_least_one', 'at_least_two'])
    log.info(f"Metrics: {metrics}")

    if features:
        analyse_features(clf, model, path=dir_path)
    plot_bad_qualitative_results(X_test, y_test, y_pred, model, clf, path=dir_path)
    plot_good_qualitative_results(X_test, y_test, y_pred, model, clf, path=dir_path)
    plot_cfm(y_test, y_pred,  path=dir_path)


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

        jaccard = score_per_sample(y_test, classifier.predict(
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
    evaluate(MultiLabelClassifier("lreg"), BagOfWords("tf-idf", ngram_range=(1, 1)))
    evaluate(MultiLabelClassifier("svm"), BagOfWords("tf-idf", ngram_range=(1, 1)))
    evaluate(MultiLabelClassifier("knn"), BagOfWords("tf-idf", ngram_range=(1, 1)))
    evaluate(MultiLabelClassifier("mlp"), BagOfWords("tf-idf", ngram_range=(1, 1)))

    # comparative_evaluation(BagOfWords("tf-idf", ngram_range=(1, 1)))


if __name__ == "__main__":
    main()
