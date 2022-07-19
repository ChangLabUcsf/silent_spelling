# -*- coding: utf-8 -*-
"""
Utility functions for signal analysis.

:Author: Jessie R. Liu, 2021
"""

import numpy as np


def balance_classes(X, y, balancer=None, min_samples=None):
    """
    A function to balance a dataset of X features and y labels.

    Parameters
    ----------
    X : nd array of samples with shape (samples, ...)
    y : nd array of labels
    balancer : 1d array of labels
        Used to balance the classes. This can be left as None if the y
        features are the same ones to use to balance the data.

    Returns
    -------
    A subsample of X and y with balanced classes.
    """

    if balancer is None:
        balancer = np.copy(y)

    assert X.shape[0] == y.shape[0] == balancer.shape[0]

    # Get the unique classes and the counts in each.
    unique_classes, num_per_class = np.unique(balancer, return_counts=True)

    # Get the smallest populated class to balance by.
    if min_samples is None:
        min_samples = min(num_per_class)

    # Loop through the classes and collect the indices
    keep_idx = []
    for cur_class in unique_classes:
        cur_class_idx = np.where(balancer == cur_class)[0]
        np.random.shuffle(cur_class_idx)
        keep_idx.append(cur_class_idx[:min_samples])
    keep_idx = np.sort(np.concatenate(keep_idx))

    return X[keep_idx], y[keep_idx]
