from joblib import Parallel, delayed
import numpy as np
from sklearn.base import is_classifier
from sklearn.calibration import check_classification_targets
from sklearn.utils._metadata_requests import _routing_enabled, process_routing
from sklearn.utils.validation import has_fit_parameter, _check_method_params
from sklearn.utils._bunch import Bunch
from sklearn.multioutput import _fit_estimator
from imblearn.over_sampling import SMOTE


def balanced_fit(self, X, y, sample_weight=None, **fit_params):
    """Fit the model to data, separately for each output variable.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input data.

    y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
        Multi-output targets. An indicator matrix turns on multilabel
        estimation.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, then samples are equally weighted.
        Only supported if the underlying regressor supports sample
        weights.

    **fit_params : dict of string -> object
        Parameters passed to the ``estimator.fit`` method of each step.

        .. versionadded:: 0.23

    Returns
    -------
    self : object
        Returns a fitted instance.
    """
    if not hasattr(self.estimator, "fit"):
        raise ValueError("The base estimator should implement a fit method")

    y = self._validate_data(X="no_validation", y=y, multi_output=True)

    if is_classifier(self):
        check_classification_targets(y)

    if y.ndim == 1:
        raise ValueError(
            "y must have at least two dimensions for "
            "multi-output regression but has only one."
        )

    if _routing_enabled():
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )
    else:
        if sample_weight is not None and not has_fit_parameter(
            self.estimator, "sample_weight"
        ):
            raise ValueError(
                "Underlying estimator does not support sample weights."
            )

        fit_params_validated = _check_method_params(X, params=fit_params)
        routed_params = Bunch(estimator=Bunch(fit=fit_params_validated))
        if sample_weight is not None:
            routed_params.estimator.fit["sample_weight"] = sample_weight

    ys = [y[:, i] for i in range(y.shape[1])]

    Xs_new = []
    ys_new = []

    for y in ys:
        y_pos = np.argwhere(y == 1)
        y_neg = np.argwhere(y == 0)
        y_neg = y_neg[:len(y_pos)*2]

        y_new = y[np.concatenate([y_pos, y_neg]).flatten()]
        X_new = X[np.concatenate([y_pos, y_neg]).flatten()]

        idx = np.random.permutation(len(y_new))
        y_new = y_new[idx]
        X_new = X_new[idx]

        ys_new.append(y_new)
        Xs_new.append(X_new)

    self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        delayed(_fit_estimator)(
            self.estimator, Xs_new[i], ys_new[i], **routed_params.estimator.fit
        )
        for i in range(len(ys_new))
    )

    if hasattr(self.estimators_[0], "n_features_in_"):
        self.n_features_in_ = self.estimators_[0].n_features_in_
    if hasattr(self.estimators_[0], "feature_names_in_"):
        self.feature_names_in_ = self.estimators_[0].feature_names_in_

    return self
