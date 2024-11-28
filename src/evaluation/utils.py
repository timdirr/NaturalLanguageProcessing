import os
import json
import numpy as np
import pandas as pd
import logging as log
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from typing import Union
from wordcloud import WordCloud
from globals import DATA_PATH, EXPORT_PATH
from helper import pandas_ndarray_series_to_numpy
from preprocess.dataloader import load_stratified_data
from classifier.base import MultiLabelClassifier
from text_modelling.modelling import BagOfWords, WordEmbeddingModel


def prepare_evaluate(model: MultiLabelClassifier,
                     text_model: Union[BagOfWords, WordEmbeddingModel]):

    clf_name = type(model.multi_output_clf_.estimators_[0]).__name__
    text_model_name = type(text_model.model).__name__
    dir_path = os.path.join(EXPORT_PATH, f"evluation_{clf_name}_{text_model_name}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log.info(f"Evaluating model {type(model.multi_output_clf_.estimators_[0]).__name__}")

    return dir_path


def get_feature_importances(model: MultiLabelClassifier, text_model: Union[BagOfWords, WordEmbeddingModel]):
    '''
    Get feature importances for given model and text model.

    Args:
        model (MultiLabelClassifier): Model for which feature importances are to be computed.

        text_model (Union[BagOfWords, WordEmbeddingModel]): Text model used to vectorize text data.

    Returns:
        feat_impts (list of np.ndarray): List of feature importances for each classifier in MultiOutputClassifier
    '''
    feat_impts = []  # contains feature importances for each classifier contained in MultiOutputClassifier
    estimators = model.multi_output_clf_.estimators_

    if hasattr(estimators[0], 'coef_'):
        for clf in estimators:
            # ndarray of shape (1, n_features)
            feat_impts.append(clf.coef_[0])

    elif hasattr(estimators[0], 'feature_importances_'):
        for clf in estimators:
            # ndarray of shape (n_features,)
            feat_impts.append(clf.feature_importances_)
    else:
        raise Warning(
            "Model does not have attribute for feature importance. Returning None")
    return feat_impts
