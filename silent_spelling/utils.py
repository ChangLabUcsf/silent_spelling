# -*- coding: utf-8 -*-
"""
Utility functions for spelling paper figures.

:Author: Jessie R. Liu and Sean L. Metzger, 2021
"""
import operator

import matplotlib as mpl
import numpy as np
import scipy


def bootstrap_confidence_intervals(x, ci=[0.5, 99.5], B=2000, metric=np.median):
    dist = [metric(np.random.choice(x, size=len(x), replace=True)) for _ in range(B)]
    return np.percentile(dist, ci)


def extract_acc(df):
    df['most_likely'] = [np.argmax(p) for p in df['pred_vec']]
    accs = []
    for cv in range(10):
        sdf = df.loc[df['cv'] == cv]

        acc_cv = np.mean(sdf['most_likely'] == sdf['label'])
        accs.append(acc_cv)
    return accs


def customize_boxplot(bp=None, facecolors=None, linewidth=1.5):
    for bp_element, fc in zip(bp['boxes'], facecolors):
        bp_element.set(facecolor=fc, edgecolor='k', linewidth=linewidth)

    for bp_key in ['medians', 'whiskers', 'caps']:
        for bp_element in bp[bp_key]:
            bp_element.set(color='k', linewidth=linewidth)


def add_significance_annot(ax, x1, x2, y1, y2, marker='*', color='k',
                           marker_height=0.75, marker_prop=0.35,
                           **kwargs):
    for key in ['line', 'annot']:
        if key not in kwargs.keys():
            kwargs[key] = {}

    ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], color=color, **kwargs['line'])
    ax.annotate(marker, (np.mean([x1, x2]), y2 + marker_height),
                va='center', ha='center', **kwargs['annot'])
    return ax


def plotting_defaults(font='Arial', fontsize=16, linewidth=2):
    # Set some plotting colors and styles
    mpl.rcParams.update({'font.size': fontsize})
    mpl.rcParams['font.sans-serif'] = [font]

    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = linewidth
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = linewidth
    mpl.rcParams['lines.linewidth'] = linewidth
    mpl.rcParams['axes.linewidth'] = linewidth
    
    
def holm_bonferroni_correction(hypothesis_p_vals, bonferroni_n=None):
    """
    Written by David A. Moses.
    
    Computes adjusted p-values using Holm-Bonferroni correction.
    Parameters
    ----------
    hypothesis_p_vals : dict
        A dictionary in which each key is an object representing a hypothesis
        label and each value is a p-value associated with that hypothesis (as a
        float).
    bonferroni_n : int or None
        The correction "size" to use, which should be equal to the number of
        hypotheses. If this is not provided, the number of items in the
        `hypothesis_p_vals` dictionary will be used.
    Returns
    -------
    dict
        A dictionary in which each key is a hypothesis label and each value is
        an adjusted p-value after Holm-Bonferroni correction (the keys in this
        dictionary will be the same as the keys in the provided
        `hypothesis_p_vals` dictionary, although the order may be different).
    """

    # Obtains a list of tuples representing the hypotheses sorted by ascending
    # p-value
    sorted_hypothesis_p_vals = sorted(
        hypothesis_p_vals.items(), key=operator.itemgetter(1)
    )

    # Computes the correction size (if needed)
    if bonferroni_n is None:
        bonferroni_n = len(hypothesis_p_vals)

    # Computes and returns the adjusted p-values
    return {
        cur_hypothesis_label: max(
            min(1., (bonferroni_n - j) * sorted_hypothesis_p_vals[j][1])
            for j in range(i + 1)
        )
        for i, (cur_hypothesis_label, _) in enumerate(sorted_hypothesis_p_vals)
    }


def p_value_calc(data, test_statistic=None, axis=0):
    """
    Calculate the p-value for a specific test statistic value from
    the distribution of re-sampled test statistics.
    Parameters
    ----------
    data : nd-array
        The resampled test statistics.
    test_statistic : int or float
        The test statistic to compare against.
    axis : int
        The axis along which to compute (the axis holding the bootstrap or
        permutation iterations).
    Returns
    -------
    p_value : nd-array
        Array of p-values, same shape as incoming data except reducing the
        given axis.
    """
    # Shift the distribution to be centered around 0.
    dmean = data.mean(axis)
    data = data - dmean
    test_statistic = test_statistic - dmean

    # Calculate the number of test statistics resulting from resampling)
    # that are more extreme than our test-statistic we want to calculate the
    # p-value for.
    p_value = np.sum(np.abs(data) > np.abs(test_statistic), axis=axis) / \
              data.shape[axis]

    return p_value


def correlation_permutation(group1, group2, n_permute=1000,
                            corr=scipy.stats.pearsonr, return_dist=False):
    """
    Perform a permutation test for a correlation function. Performs the
    permutation by permuting only 1 group, to assess the null
    hypothesis that the group structure has a significant correlation.
    Parameters
    ----------
    group1 : 1d array
        The first group of observations.
    group2 : 1d array
        The second group of observations, with equal shape to group1.
    n_permute : int
        The number of permutations to
    corr : function
        A correlation function that accepts 2 args (each a group of
        observations to calculate the correlation between) and returns a
        tuple with the first element being the correlation value and the
        second element being a p-value (which is ignored in favor of the
        permutation calculated p-value). Need not be a scipy.stats function.
    return_dist : bool, default False
        Whether to return the distribution of computed correlation values
        for each permutation.
    Returns
    -------
    test_correlation : float
        The correlation between the un-permuted group1 and group2.
    p_value : float
        The computed p-value associated with the permutation distribution
        and test_correlation.
    (optionally) corr_dist : 1d array
        The distribution of correlation values from each permutation.
    """

    # Compute the test statistic on the data.
    test_correlation, _ = corr(group1, group2)

    # Initialize array.
    corr_dist = np.zeros(n_permute)

    # Calculate the correlation distribution for n_permute permutations.
    for p in range(n_permute):
        # Permute 1 group.
        permuted_group1 = np.random.permutation(group1)

        # Calculate and append the correlation value.
        corr_value, _ = corr(permuted_group1, group2)
        corr_dist[p] = corr_value

    # Calculate the p-value.
    p_value = p_value_calc(corr_dist, test_statistic=test_correlation)

    if return_dist:
        return test_correlation, p_value, corr_dist
    else:
        return test_correlation, p_value
    