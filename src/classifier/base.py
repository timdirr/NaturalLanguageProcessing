from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import logging as log
import pandas as pd
import helper
from globals import SEED


class MultiLabelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator_name, verbose=True, **kwargs):
        if verbose:
            log.info(f"Creating multilabel classifier {estimator_name}")

        if estimator_name == "lreg":
            self.base_estimator = LogisticRegression(**kwargs)
        elif estimator_name == "knn":
            self.base_estimator = KNeighborsClassifier(**kwargs)
        elif estimator_name == "svm":
            self.base_estimator = SVC(**kwargs)
        elif estimator_name == "bayes":
            self.base_estimator = MultinomialNB(**kwargs)
        elif estimator_name == "rf":
            self.base_estimator = RandomForestClassifier(**kwargs)
        elif estimator_name == "mlp":
            self.base_estimator = MLPClassifier(**kwargs)
        else:
            raise ValueError(
                "Base estimator not found. Choose from: lreg, knn, svm, bayes, rf, mlp")

        self.multi_output_clf_ = MultiOutputClassifier(self.base_estimator)

    def fit(self, X, y):
        if isinstance(y, pd.Series):
            y = helper.pandas_ndarray_series_to_numpy(y)
        self.multi_output_clf_.fit(X, y)
        return self

    def predict(self, X):
        return self.multi_output_clf_.predict(X)

    def predict_proba(self, X):
        return self.multi_output_clf_.predict_proba(X)
