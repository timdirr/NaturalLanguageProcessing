from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from scipy.sparse import csr_matrix

import numpy as np
from sklearn.svm import SVC


class MultiLabelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator_name, **kwargs):
        if estimator_name == "knn":
            self.base_estimator = KNeighborsClassifier(**kwargs)
        elif estimator_name == "svm":
            self.base_estimator = SVC(**kwargs)
        elif estimator_name == "bayes":
            self.base_estimator = MultinomialNB(**kwargs)
        else:
            raise ValueError("base_estimator must be knn. Nothing else supported yet")
        self.multi_output_clf_ = MultiOutputClassifier(self.base_estimator)

    def fit(self, X, y):
        y_matrix = self.compute_target_matrix(y)
        self.multi_output_clf_.fit(X, y_matrix)
        return self

    def compute_target_matrix(self, y):
        y = np.array(y.tolist()).astype(int)
        return y

    def predict(self, X):
        return self.multi_output_clf_.predict(X)

    def predict_proba(self, X):
        return self.multi_output_clf_.predict_proba(X)