import os

import logging as log
from globals import DATA_PATH, EXPORT_PATH
from classifier.base import MultiLabelClassifier


def prepare_evaluate(classifier_name, model_name, model, binary=False):
    if classifier_name == "MovieGenreClassifier":
        model_name = "raw_text"
    if binary:
        dir_path = os.path.join(EXPORT_PATH, f"evaluation_{classifier_name}_{model_name}_binary_{str(model)}")
    else:
        dir_path = os.path.join(EXPORT_PATH, f"evaluation_{classifier_name}_{model_name}_{str(model)}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if binary:
        log.info(f"Evaluating binary classifier {classifier_name} with {model_name}")
    else:
        log.info(f"Evaluating classifier {classifier_name} with {model_name}")

    return dir_path


def get_feature_importances(clf: MultiLabelClassifier):
    '''
    Get feature importances for given model and text model.

    Args:
        clf (MultiLabelClassifier): Model for which feature importances are to be computed.

    Returns:
        feat_impts (list of np.ndarray): List of feature importances for each classifier in MultiOutputClassifier
    '''
    feat_impts = []  # contains feature importances for each classifier contained in MultiOutputClassifier
    estimators = clf.multi_output_clf_.estimators_

    if hasattr(estimators[0], 'coef_'):
        log.info("Extracting feature importances from coef_ attribute.")
        for est in estimators:
            # ndarray of shape (1, n_features)
            feat_impts.append(est.coef_[0])

    elif hasattr(estimators[0], 'feature_importances_'):
        log.info("Extracting feature importances from feature_importances_ attribute.")
        for est in estimators:
            # ndarray of shape (n_features,)
            feat_impts.append(est.feature_importances_)

    elif hasattr(estimators[0], 'feature_log_prob_'):
        log.info("Extracting feature importances from feature_log_prob_ attribute.")
        for est in estimators:
            # ndarray of shape (n_classes, n_features)
            feat_impts.append(est.feature_log_prob_[1, :])
    else:
        log.warning("Model does not have attribute for feature importance. Returning empty list.")
    return feat_impts
