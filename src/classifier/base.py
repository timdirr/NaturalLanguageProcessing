from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
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
        if estimator_name == "knn":
            if not kwargs:
                if verbose:
                    log.info(f"no kwargs found for {estimator_name}, using default kwargs n_neighbors=15, weights='distance', algorithm='auto', metric='minkowski', p=3, n_jobs=-1")
                self.base_estimator = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='auto', metric='euclidean', n_jobs=-1)
            else:
                self.base_estimator = KNeighborsClassifier(**kwargs)
        elif estimator_name == "svm":
            if not kwargs:
                if verbose:
                    log.info(f"no kwargs found for {estimator_name}, C=1, kernel='rbf', gamma='scale', class_weight='balanced', max_iter=-1, random_state={SEED}")
                self.base_estimator = SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced', max_iter=-1, random_state=SEED)
            else:
                self.base_estimator = SVC(**kwargs)
        elif estimator_name == "bayes":
            if not kwargs:
                if verbose:
                    log.info(f"no kwargs found for {estimator_name}, alpha=1.0, fit_prior=True")
                self.base_estimator = MultinomialNB(alpha=1.0, fit_prior=True)
            else:
                self.base_estimator = MultinomialNB(**kwargs)
        elif estimator_name == "rf":
            self.base_estimator = RandomForestClassifier(**kwargs)
        elif estimator_name == "mlp":
            self.base_estimator = MLPClassifier(**kwargs)
        else:
            raise ValueError("base_estimator must be knn. Nothing else supported yet")

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
