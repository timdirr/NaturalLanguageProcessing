import logging as log
from joblib import Parallel, delayed
import numpy as np
from sklearn.base import is_classifier
from sklearn.calibration import check_classification_targets
from sklearn.utils._metadata_requests import _routing_enabled, process_routing
from sklearn.utils.validation import has_fit_parameter, _check_method_params
from sklearn.utils._bunch import Bunch
from sklearn.multioutput import _fit_estimator
from imblearn.over_sampling import SMOTE

from helper import load_genres


def balanced_fit(self, X, y, sample_weight=0.5, balancing_ratio=None, **fit_params):
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
    Xs = [X for _ in range(y.shape[1])]

    if balancing_ratio:
        Xs_resampled = []
        ys_resampled = []

        log.info(f"Balancing ratio: {balancing_ratio}")
        smote = SMOTE(random_state=42, sampling_strategy=balancing_ratio)
        genres = load_genres()
        for i, y in enumerate(ys):
            log.info(f"Resampling for Genre: {genres[i]}")
            log.info(f"Num positive samples: {np.sum(y)}")
            log.info(f"Num total sumples: {len(y)}")
            if np.sum(y) > (balancing_ratio/(1 + balancing_ratio)) * len(y):
                log.info("Skipping SMOTE")
                ys_resampled.append(y)
                Xs_resampled.append(X)
                continue

            X_smote, y_smote = smote.fit_resample(X, y)

            log.info(f"Num positive samples after resampling: {np.sum(y_smote)}")
            log.info(f"Num total sumples after resampling: {len(y_smote)}")

            ys_resampled.append(y_smote)
            Xs_resampled.append(X_smote)

        log.info("Resampling finished")

        ys = ys_resampled
        Xs = Xs_resampled
    else:
        log.info("No resampling")

    self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        delayed(_fit_estimator)(
            self.estimator, Xs[i], ys[i], **routed_params.estimator.fit
        )
        for i in range(len(ys))
    )

    if hasattr(self.estimators_[0], "n_features_in_"):
        self.n_features_in_ = self.estimators_[0].n_features_in_
    if hasattr(self.estimators_[0], "feature_names_in_"):
        self.feature_names_in_ = self.estimators_[0].feature_names_in_

    return self
