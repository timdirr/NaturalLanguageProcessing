import numpy as np
from sklearn.metrics import jaccard_score, hamming_loss, accuracy_score, f1_score, precision_score, recall_score, multilabel_confusion_matrix


def signed_overlap(y_true, y_pred):
    """
    Compute signed overlap metric for multilabel classification.

    Args:
        y_true (np.ndarray): Ground truth (binary matrix, shape [n_samples, n_classes]).
        y_pred (np.ndarray): Predictions (binary matrix, shape [n_samples, n_classes]).

    Returns:
        float: Signed overlap score.
    """
    intersection = np.sum(np.logical_and(y_true, y_pred), axis=1)
    false_negatives = np.sum(np.logical_and(y_true, ~y_pred), axis=1)
    false_positives = np.sum(np.logical_and(~y_true, y_pred), axis=1)
    union = np.sum(np.logical_or(y_true, y_pred), axis=1)

    signed_overlap_score = np.mean((intersection - false_negatives - false_positives) / (union + 1e-9))
    return signed_overlap_score


def single_signed_overlap(y_true, y_pred):
    # signed overlap for single observations
    intersection = np.sum(np.logical_and(y_true, y_pred))
    false_negatives = np.sum(np.logical_and(y_true, ~y_pred))
    false_positives = np.sum(np.logical_and(~y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))

    signed_overlap_score = (intersection - false_negatives - false_positives) / (union + 1e-9)
    return signed_overlap_score


def at_least_k(y_true, y_pred, k: int = 1):
    """
    Compute the fraction of samples for which at least one label is predicted correctly.

    Args:
        y_true (np.ndarray): Ground truth (binary matrix, shape [n_samples, n_classes]).
        y_pred (np.ndarray): Predictions (binary matrix, shape [n_samples, n_classes]).

    Returns:
        float: Fraction of samples for which at least k label is predicted correctly
    """
    correct = np.sum(np.logical_and(y_true, y_pred), axis=1)
    at_least_k_score = np.mean(correct >= k)
    return at_least_k_score


def confusion_matrix(y_true, y_pred, plot=False):
    """
    Compute per label confusion matrix

    Args:
        y_true (np.ndarray): Ground truth (binary matrix, shape [n_samples, n_classes]).
        y_pred (np.ndarray): Predictions (binary matrix, shape [n_samples, n_classes]).

    Returns:
        float: list of confusion matrices.
    """

    cfm = multilabel_confusion_matrix(y_true, y_pred)
    return cfm


def score_per_sample(y_true, y_preds, metric=jaccard_score):
    """
    Computes the Jaccard index per sample.

    Args:
        y_true (np.ndarray): Ground truth (binary matrix, shape [n_samples, n_classes]).
        y_pred (np.ndarray): Predictions (binary matrix, shape [n_samples, n_classes]).r.

    Returns:
        list: scores per sample
    """
    scores = []

    for truth, pred in zip(y_true, y_preds):
        scores.append(metric(truth, pred))

    return scores


def compute_metrics(y_true,
                    y_pred,
                    metrics_names: list = ['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall', 'at_least_one', 'at_least_two', 'signed_overlap', 'confusion_matrix']):
    '''
    Get metrics for multilabel classification.

    Args:
        y_true (np.ndarray): Ground truth (binary matrix, shape [n_samples, n_classes]).
        y_pred (np.ndarray): Predictions (binary matrix, shape [n_samples, n_classes]).
        metrics_names (list[str]):  List of metrics to compute. Default: ['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall'].

    Returns:
        metrics (dict): Dictionary containing the computed metrics.
    '''

    metrics = {}
    if 'jaccard' in metrics_names:
        metrics['jaccard'] = jaccard_score(y_true, y_pred, average='samples')
    if 'hamming' in metrics_names:
        metrics['hamming'] = hamming_loss(y_true, y_pred)
    if 'accuracy' in metrics_names:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    if 'f1' in metrics_names:
        metrics['f1'] = f1_score(y_true, y_pred, average='samples')
    if 'precision' in metrics_names:
        metrics['precision'] = precision_score(y_true, y_pred, average='samples')
    if 'recall' in metrics_names:
        metrics['recall'] = recall_score(y_true, y_pred, average='samples')
    if 'at_least_one' in metrics_names:
        metrics['at_least_one'] = at_least_k(y_true, y_pred, 1)
    if 'at_least_two' in metrics_names:
        metrics['at_least_two'] = at_least_k(y_true, y_pred, 2)
    if 'signed_overlap' in metrics_names:
        metrics['signed_overlap'] = signed_overlap(y_true, y_pred)
    if 'confusion_matrix' in metrics_names:
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return metrics
