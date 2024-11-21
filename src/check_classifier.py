from classifier.base import MultiLabelClassifier
from preprocess.dataloader import load_stratified_data
import pandas as pd
import os

from globals import SEED
from text_modelling.modelling import BagOfWords
from sklearn.metrics import jaccard_score
import numpy as np

import logging as log

from skmultilearn.model_selection import IterativeStratification

log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def test_model(model, classifier, train, test):
    X_train, y_train = train.drop(["genre"], axis=1), train["genre"]
    X_test, y_test = test.drop(["genre"], axis=1), test["genre"]

    transformed_data = model.fit_transform(X_train)
    classifier = classifier.fit(transformed_data, y_train)

    jaccard = jaccard_score(classifier.compute_target_matrix(y_test),
                            classifier.predict(model.transform(X_test)), average='samples')
    return jaccard


def main():
    train, test, dev = load_stratified_data()
    X_train, y_train = train.drop(columns='genre'), train['genre']
    X_test, y_test = test.drop(columns='genre'), test['genre']
    X_dev, y_dev = dev.drop(columns='genre'), dev['genre']

    k_fold = IterativeStratification(n_splits=5, order=1, random_state=SEED)
    jaccard_scores = []
    log.info(f"Performing 5-fold cross-validation for count with KNN")
    for train, test in k_fold.split(X_dev, y_dev):
        jaccard_scores.append(test_model(BagOfWords("count", ngram_range=(1,1)), MultiLabelClassifier("knn"), train, test))

    log.info(f"Mean-Jaccard score: {np.mean(jaccard_scores)}")

    k_fold = IterativeStratification(n_splits=5, order=1, random_state=SEED)
    jaccard_scores = []
    log.info(f"Performing 5-fold cross-validation for tf-idf with KNN")
    for train, test in k_fold.split(X_dev, y_dev):
        jaccard_scores.append(test_model(BagOfWords("tf-idf", ngram_range=(1,1)), MultiLabelClassifier("knn"), train, test))

    log.info(f"Mean-Jaccard score: {np.mean(jaccard_scores)}")

    k_fold = IterativeStratification(n_splits=5, order=1, random_state=SEED)
    jaccard_scores = []
    log.info(f"Performing 5-fold cross-validation for tf-idf with KNN")
    for train, test in k_fold.split(X_dev, y_dev):
        jaccard_scores.append(test_model(BagOfWords("tf-idf", ngram_range=(1,1)), MultiLabelClassifier("knn"), train, test))

    log.info(f"Mean-Jaccard score: {np.mean(jaccard_scores)}")


if __name__ == "__main__":
    main()
#%%
