import numpy as np


def corr_score(target, prediction, weights):
    """
    Calculates the weighted correlation score as defined in competition
    """
    w = np.ravel(weights)
    a = np.ravel(target)
    b = np.ravel(prediction)

    sum_w = np.sum(w)
    mean_a = np.sum(a * w) / sum_w
    mean_b = np.sum(b * w) / sum_w
    var_a = np.sum(w * np.square(a - mean_a)) / sum_w
    var_b = np.sum(w * np.square(b - mean_b)) / sum_w

    cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
    corr = cov / np.sqrt(var_a * var_b)

    return corr
