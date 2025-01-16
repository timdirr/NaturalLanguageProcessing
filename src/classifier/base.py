import functools
from joblib import Parallel, delayed
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import _MultiOutputEstimator, MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import logging as log
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import helper
from globals import SEED, EXPORT_PATH
import os
import json

from classifier.balanced_fit import balanced_fit
from helper import load_genres


class MultiLabelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator_name, verbose=True, balancing_ratio=None, **kwargs):
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
        elif estimator_name == "dt":
            self.base_estimator = DecisionTreeClassifier(**kwargs, random_state=SEED)
        elif estimator_name == "rf":
            self.base_estimator = RandomForestClassifier(**kwargs)
        elif estimator_name == "mlp":
            self.base_estimator = MLPClassifier(**kwargs)
        else:
            raise ValueError(
                "Base estimator not found. Choose from: lreg, knn, svm, bayes, rf, mlp")

        if estimator_name == "svm":
            self.multi_output_clf_ = MultiOutputClassifier(BaggingClassifier(self.base_estimator, n_jobs=-1, n_estimators=10))
        else:
            self.multi_output_clf_ = MultiOutputClassifier(self.base_estimator)

        try:
            # raise an AttributeError if `predict_proba` does not exist for the base estimator
            self.multi_output_clf_._check_predict_proba()
            self._has_predict_proba = True
        except AttributeError:
            self._has_predict_proba = False

        self.balancing_ratio = balancing_ratio
        # monkey patch the fit method to use the balanced_fit method
        _MultiOutputEstimator.fit = balanced_fit

    def fit(self, X, y):
        if isinstance(y, pd.Series):
            y = helper.pandas_ndarray_series_to_numpy(y)

        # hack to set the balancing_ratio of the balanced_fit method
        _MultiOutputEstimator.fit.__defaults__ = (0.5, self.balancing_ratio,)
        # fit the model
        self.multi_output_clf_.fit(X, y)
        return self

    def predict(self, X, custom_pred=False, X_raw=None):
        if self._has_predict_proba:
            return (self.predict_proba(X, custom_pred, X_raw) >= 0.5).astype(int)
        return self.multi_output_clf_.predict(X)

    def predict_proba(self, X, custom_pred=False, X_raw=None):
        y_prob = np.array(self.multi_output_clf_.predict_proba(X)).transpose()[1]
        if custom_pred:
            for i, genre in enumerate(load_genres()):
                with open(os.path.join(EXPORT_PATH, f"top_words_{genre}.json"), 'r') as file:
                    keywords = json.load(file)
                y_frac_keywords = np.array([])
                for description in X_raw:
                    num_keywords = sum(1 for keyword in keywords if keyword in description.lower())
                    y_frac_keywords = np.append(y_frac_keywords, num_keywords / len(keywords) if keywords else 0)
                y_prob[:, i] = y_prob[:, i] + y_frac_keywords
        return y_prob

    def predict_at_least_1(self, X, custom_pred=False, X_raw=None):
        predictions = self.predict(X, custom_pred, X_raw)
        if self._has_predict_proba:
            predictions_proba = self.predict_proba(X, custom_pred, X_raw)
            idx = predictions.sum(axis=1) == 0
            predictions[idx] = (predictions_proba[idx] >= predictions_proba.max(axis=1)[idx][:, None]).astype(int)
        return predictions
