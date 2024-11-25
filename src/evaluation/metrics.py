import numpy as np
from sklearn.metrics import jaccard_score, hamming_loss, accuracy_score, f1_score, precision_score, recall_score


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


def compute_metrics(y_true,
                    y_pred,
                    metrics_names: list[str] = ['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall', 'at_least_one', 'at_least_two', 'signed_overlap']):
    '''
    Get metrics for multilabel classification.

    Args:
        y_true (np.ndarray): Ground truth (binary matrix, shape [n_samples, n_classes]).
        y_pred (np.ndarray): Predictions (binary matrix, shape [n_samples, n_classes]).
        metrics_names (list[str]):  List of metrics to compute. Default: ['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall'].
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
    return metrics
