# -*- coding: utf-8 -*-
"""
Utility functions for speech detection models, specifically functions
associated with the detection probability post-processing.

:Author: Jessie R. Liu
:Copyright: Copyright (c) 2019, Jessie R. Liu, All rights reserved.
"""

import numpy as np

from utils import moving_avg


####################################################################
####### FUNCTIONS TO HELP WITH CALCULATING DETECTION RESULTS #######
####################################################################

def probability_threshold(probs, probability_thresh=0.5):
    """
    Apply a probability threshold to probabilities.

    Parameters
    ----------
    probs : array-like
        The probabilities across time for a single time series.
    probability_thresh : float
        The probability threshold.

    Returns
    -------
    Thresholded probabilities. Values equal to or above the threshold will
    have a value of one, all else will have a value of zero.
    """

    # Ensure 1d shape.
    probs = probs.reshape((-1))

    # Threshold the probabilities.
    return np.where(probs >= probability_thresh, 1, 0)


def time_threshold(times, thresh=10):
    """
    Apply a time threshold (debouncing) to binary data.

    Parameters
    ----------
    times : 1d binary array
        The array of data points coming in. This array should contain 0 for
        no event and 1 for the intended event occurring.
    thresh : int
        The number of time points that must be labeled as a certain event in
        order to determine that the event is truly starting (or ending).

    Returns
    -------
    event_occurring : 1d boolean
        A boolean 1d array with True for the event truly occurring and False
        otherwise.
    """

    # The default event state is False, for no event.
    current_event = False

    # Format shape of incoming probability thresholded predictions,
    # initialize event_occurring array, and add a buffer to slide our
    # threshold window over.
    times = times.reshape((-1))
    event_occurring = np.zeros_like(times)
    buffer_times = np.concatenate([np.zeros((thresh - 1)), times])

    # Consider a buffer window leading up to and including each time point
    # to decide whether an event has occurred for enough time points to be
    # considered an event onset. Similarly, decide whether an event has _not_
    # occurred for long enough to be considered an event offset.
    for cur_point in range(len(times)):

        # Calculate the sum of this buffer window.
        thresh_sum = np.sum(buffer_times[cur_point:(cur_point + thresh)])

        if thresh_sum == thresh:
            # If the binary time points in the buffer window sum to the
            # threshold, then an event has started. Adjust event_occurring
            # for where the event actually started.
            event_occurring[cur_point] = True
            current_event = True
            event_occurring[cur_point - thresh + 1:cur_point] = True

        elif thresh_sum == 0:
            # If the binary time points in the buffer window sum to zero, then
            # an event has ended. Adjust event_occurring for where the event
            # actually ended.
            event_occurring[cur_point] = False
            current_event = False
            event_occurring[cur_point - thresh + 1:cur_point] = False

        else:
            # If there's a mix of 0 and 1 events, then a transition might be
            # happening but has not been confirmed, so assign the last
            # approved state.
            event_occurring[cur_point] = current_event

    return event_occurring.astype(bool)


def calculate_number_of_events(times):
    """
    Calculate the number of events that occur in a single time series.

    Parameters
    ----------
    times : 1d array-like
        Binary time series with 1 denoting an event is occurring,
        and 0 indicating an event is not occuring.

    Returns
    -------
    The number of events as an integer.
    """
    # Calculate the number of events by shifting the events and seeing where
    # there are onsets and offsets.
    shifted = np.concatenate([np.array([times[0]]), times[:-1]])
    return len(np.where(times != shifted)[0]) / 2


def detectionScore(pred_labels=None, act_labels=None, tp_events=None,
                   fp_events=None, fn_events=None,
                   act_num_events=None, positive_frac=0.5, frame_acc_frac=0.5):
    """
    """

    # Computes the frame-by-frame accuracy measure (if appropriate)
    if pred_labels is not None and act_labels is not None:
        negative_frac = 1. - positive_frac
        frame_acc = (((positive_frac * np.sum(pred_labels * act_labels)) +
                      (negative_frac * np.sum(~(pred_labels + act_labels)))) /
                     ((positive_frac * np.sum(act_labels)) +
                      (negative_frac * np.sum(~act_labels))))
    else:
        frame_acc = None

    # Computes the general event detection accuracy measure (if appropriate)
    if tp_events is not None and act_num_events is not None:
        event_acc = float(max(0, (tp_events - fp_events - fn_events) /
                              act_num_events))
    else:
        event_acc = None

    # Counts the number of accuracy measures that were computed
    num_measures = sum(v is not None for v in (frame_acc, event_acc))

    # If both measures were computed, the returned score is a weighted
    # average of the two measures
    if num_measures == 2:
        # return (frame_acc_frac * frame_acc) + ((1. - frame_acc_frac) *
        # event_acc)
        return ((frame_acc_frac * frame_acc) + (
                (1. - frame_acc_frac) * event_acc)), frame_acc, event_acc

    # Otherwise, if only one measure was computed, the returned score is equal
    # to that measure
    elif num_measures == 1:
        return [frame_acc] if frame_acc is not None else [event_acc]

    # Otherwise, a TypeError is raised because the appropriate arguments were
    # not provided
    else:
        raise TypeError('Either the predicted and actual labels or events (or'
                        'both of these pairs) should be provided.')


##########################################################
####### EVALUATE RESULTS USING DETECTION FUNCTIONS #######
##########################################################

def calculate_intersection_over_union(true=None, pred=None):
    """
    Calculates the intersection over union (Jaccard index) of detected versus
    true speech frames.

    Parameters
    ----------
    true : 1d array
        Binary/boolean array of true speech frames.
    pred : 1d array
        Binary/boolean array of detected speech frames.

    Returns
    -------
    iou : float
        The intersection over union for this block.
    """
    # Make sure they are boolean for performing operations.
    true = true.astype(bool)
    pred = pred.astype(bool)

    # Logical and/or operations.
    intersection = true * pred
    union = true + pred

    # Calculate the ratio.
    iou = intersection.sum() / float(union.sum())
    return iou


def calculate_latency(true_starts=None, pred_starts=None):
    """
    Calculates the latency of detected events, defined as the time from
    go-cue to detected onset.

    Parameters
    ----------
    true_starts : 1d array of floats
        The true starting speech times for this block.
    pred_starts : 1d array of floats
        The detected starting speech times for this block.

    Returns
    -------
    2 element tuple of floats
        The average and standard deviation latencies for this block.
    """
    latency = pred_starts - true_starts
    return (np.mean(latency), np.std(latency))


def process_speech_detection(predicted,
                             blocks,
                             detection=None,
                             event_types=None):
    """
    Process predicted probabilities across feature classes to yield boolean
    time courses of predicted events.

    Parameters
    ----------
    predicted : list of nd arrays with shape (time, feature_labels)
        The predicted probabilities across feature classes, floats.
    blocks : list of ints
        The block numbers.
    detection : dict
        Dictionary with the following fields:

        'frame_acc_weight' - Weight for frame accuracy, float.
        'positive_weight' - Weight for true positive event accuracy, float.

        For each event_type, there should be a nested dictionary. For
        example with 'speech':
        detection['speech']['prob_threshold'] - Probability threshold, float.
        detection['speech']['time_threshold'] - Time threshold, float.
        detection['speech']['smooth_size'] - Size, in time-points,
        of smoothing filter, int.
        detection['speech']['smoothing'] - Type of smoothing, str.
    event_types : list of str
        The labels of each feature. First is usually 'silence'.

    Returns
    -------
    predicted_timepoints : dict
        Dictionary with "['block']['event_type'] = 1d bool array" containing
        the processed predicted events for each event type and block.
    """
    predicted_timepoints = {}

    for cur_block, block in enumerate(blocks):

        # Initialize dictionaries.
        predicted_timepoints[str(block)] = {}

        for cur_event, event_type in enumerate(event_types):

            # Only evaluate non-silence event types
            if event_type == 'silence':
                continue
            elif event_type not in detection.keys():
                continue

            # Get probability for this event.
            event_prob = predicted[cur_block][:, cur_event]

            # Apply smoothing.
            if detection[event_type]['smoothing'] == 'average':
                smoothed_event_prob = moving_avg(
                    event_prob,
                    window=detection[event_type]['smooth_size']
                )

            # Apply probability and time thresholds.
            predicted_label = probability_threshold(
                smoothed_event_prob,
                probability_thresh=detection[event_type]['prob_threshold']
            )
            predicted_label = time_threshold(
                predicted_label,
                thresh=detection[event_type]['time_threshold']
            )

            # Save the predicted label for this block and event_type.
            predicted_timepoints[str(block)][event_type] = predicted_label

    return predicted_timepoints
