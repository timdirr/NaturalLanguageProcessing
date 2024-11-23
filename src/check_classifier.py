from classifier.base import MultiLabelClassifier
from preprocess.dataloader import load_stratified_data
import pandas as pd
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from helper import pandas_ndarray_series_to_numpy

from globals import SEED
from text_modelling.modelling import BagOfWords
from sklearn.metrics import jaccard_score
import numpy as np
from tqdm import tqdm
import logging as log

log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def test_model(model, classifier, X_train, X_test, y_train, y_test):
    transformed_data = model.fit_transform(X_train)
    classifier = classifier.fit(transformed_data, y_train)

    jaccard = jaccard_score(classifier.compute_target_matrix(y_test),
                            classifier.predict(model.transform(X_test)), average='samples')
    return jaccard


def main():
    train, test, dev = load_stratified_data()
    X_train, y_train = train["description"].to_numpy(), pandas_ndarray_series_to_numpy(train["genre"])
    X_test, y_test = test["description"].to_numpy(), pandas_ndarray_series_to_numpy(test["genre"])
    X_dev, y_dev = dev["description"].to_numpy(), pandas_ndarray_series_to_numpy(dev["genre"])

    k_fold = MultilabelStratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    jaccard_scores = []
    log.info(f"Performing 5-fold cross-validation for count with KNN")

    for train_idx, test_idx in tqdm(k_fold.split(X_dev, y_dev)):
        X_dev_train, X_dev_test, y_dev_train, y_dev_test = X_dev[train_idx], X_dev[test_idx], y_dev[train_idx], y_dev[test_idx]
        jaccard_scores.append(
            test_model(
                BagOfWords("tf-idf", ngram_range=(1, 1)),
                MultiLabelClassifier("knn"),
                X_dev_train, X_dev_test, y_dev_train, y_dev_test))

    log.info(f"Mean-Jaccard score: {np.mean(jaccard_scores)}")


if __name__ == "__main__":
    main()
