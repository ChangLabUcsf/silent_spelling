# -*- coding: utf-8 -*-
"""
Utility functions for speech detection models.

:Author: Jessie R. Liu
:Copyright: Copyright (c) 2019, Jessie R. Liu, All rights reserved.
"""
import argparse
import ast
import inspect
import json
import logging
import os.path as op
import re
from copy import copy
from datetime import datetime
from functools import wraps
from itertools import islice

import numpy as np
import pandas as pd
import psutil

logger_config = {
    'filemode': 'a',
    'format'  : '%(asctime)s:%(levelname)s: %(message)s',
    'datefmt' : '%H:%M:%S',
    'level'   : logging.INFO
}


def moving_avg(data, window=20, center=False, axis=0):
    """
    Apply a moving average smoothing operation.

    Parameters
    ----------
    data : 1d or nd array
        Data to be smoothed via moving average.
    window : int
        The number of timepoints to include in the sliding smoothing window.
    center : bool
        Whether the rolling window is centered or not. Default False.

    Returns
    -------
    ndarray of floats of the smoothed data. If the input data is a 1d array,
    the output will also be a 1d array.
    """
    df = pd.DataFrame(data)
    new_data = df.rolling(window, min_periods=1, center=center,
                          axis=axis).mean().values

    # If input data was 1d, return 1d data.
    if len(data.shape) == 1:
        new_data = new_data.squeeze()

    return new_data


def check_memory_usage(pid):
    process = psutil.Process(pid)
    gmem = 1e-9 * process.memory_info().rss
    msg = f'Current memory usage: {gmem} GB'  # in GB
    return msg


def get_memory_usage(pid):
    process = psutil.Process(pid)
    # mem in gb
    gmem = 1e-9 * process.memory_info().rss
    return gmem


# Cribbed from Enteleform here: https://stackoverflow.com/questions/1389180
# decorator
def auto_attribute(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        _assign_args(self, list(args), kwargs, function)
        function(self, *args, **kwargs)

    return wrapped


# Cribbed from Enteleform here: https://stackoverflow.com/questions/1389180
# utilities
def _assign_args(instance, args, kwargs, function):
    def set_attribute(instance, parameter, default_arg):
        if not (parameter.startswith("_")):
            setattr(instance, parameter, default_arg)

    def assign_keyword_defaults(parameters, defaults):
        for parameter, default_arg in zip(
                reversed(parameters), reversed(defaults)):
            set_attribute(instance, parameter, default_arg)

    def assign_positional_args(parameters, args):
        for parameter, arg in zip(parameters, args.copy()):
            set_attribute(instance, parameter, arg)
            args.remove(arg)

    def assign_keyword_args(kwargs):
        for parameter, arg in kwargs.items():
            set_attribute(instance, parameter, arg)

    def assign_keyword_only_defaults(defaults):
        return assign_keyword_args(defaults)

    def assign_variable_args(parameter, args):
        set_attribute(instance, parameter, args)

    (POSITIONAL_PARAMS, VARIABLE_PARAM, _, KEYWORD_DEFAULTS, _,
     KEYWORD_ONLY_DEFAULTS, _) = inspect.getfullargspec(function)
    POSITIONAL_PARAMS = POSITIONAL_PARAMS[1:]  # remove 'self'

    if KEYWORD_DEFAULTS:
        assign_keyword_defaults(
            parameters=POSITIONAL_PARAMS, defaults=KEYWORD_DEFAULTS)
    if KEYWORD_ONLY_DEFAULTS:
        assign_keyword_only_defaults(defaults=KEYWORD_ONLY_DEFAULTS)
    if args:
        assign_positional_args(parameters=POSITIONAL_PARAMS, args=args)
    if kwargs:
        assign_keyword_args(kwargs=kwargs)
    if VARIABLE_PARAM:
        assign_variable_args(parameter=VARIABLE_PARAM, args=args)


def get_train_val(n, split=0.2, num_splits=np.inf, shuffle=True,
                  return_folds=True, equally_spaced=False):
    indices = np.arange(n)

    if equally_spaced:
        trainidx, validx = [], []

        for i in range(int(1 / split)):

            split_step = int(1 / split)

            if i == num_splits:
                break

            i_validx = indices[i::split_step].astype(int)
            i_trainidx = indices[np.where(~np.isin(indices, i_validx))[
                0].astype(int)].astype(int)

            validx.append(i_validx)
            trainidx.append(i_trainidx)

            if not return_folds:
                break

    else:
        if shuffle:
            np.random.shuffle(indices)

        window = int(np.floor(split * n))

        validx = []
        trainidx = []
        for i in range(int(1 / split)):

            if i == num_splits:
                break

            start = i * window
            end = (i + 1) * window

            i_validx = indices[start:end].astype(int)
            i_trainidx = indices[
                np.where(~np.isin(indices, i_validx))[0].astype(int)].astype(
                int)

            validx.append(i_validx)
            trainidx.append(i_trainidx)

            if not return_folds:
                break

    return validx, trainidx


class FLAGS:
    def __init__(self):
        pass


class directories:
    def __init__(self):
        pass


def universal_parser():
    """
    Function to create the parsed arguments for speech detection code.

    Returns
    -------
    parser : ArgumentParser
        The ArgumentParser from argparse with the defaults.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--paradigms", type=str)
    parser.add_argument("--utterance_sets", type=str)
    parser.add_argument("--out_folder", type=str)
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--feature_labels", default=None, type=str)
    parser.add_argument("--total_feature_labels", default=None, type=str)
    parser.add_argument("--model_type", default='lstm', type=str)
    parser.add_argument("--prediction_fold", default=0)
    parser.add_argument("--subject", default=None, type=str)
    parser.add_argument("--block_subjects", default=None, type=str)
    parser.add_argument("--project_name", default=None, type=str,
                        help='wandb project name')
    parser.add_argument("--date_ranges", default=None, type=str)
    parser.add_argument("--cable", default="['patient','cereplex']", type=str)
    parser.add_argument("--use_cue_times", action='store_true')
    parser.add_argument("--data_streams", type=str, default=None)
    parser.add_argument("--metric", type=str, default='both')
    parser.add_argument("--hyperopt_metric", type=str, default='loss')
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--config_prefix", type=str, default='')
    parser.add_argument("--balance_data", action='store_true', default=False)
    parser.add_argument("--behav_quality", type=int, default=2)
    parser.add_argument("--num_train_blocks", type=int, default=None)
    parser.add_argument("--num_val_blocks", type=int, default=None)
    parser.add_argument("--num_test_blocks", type=int, default=None)
    parser.add_argument("--prepare_detected_times", action='store_true')
    parser.add_argument("--no_early_stopping_learning_rate_adjustment",
                        action='store_true', default=False)
    parser.add_argument("--block_split_filename", type=str, default=None)
    parser.add_argument("--hyperopt_date", type=str, default=None)
    parser.add_argument("--use_hyperopted_params", action='store_true',
                        default=False)
    parser.add_argument("--use_stimulus_lengths", action='store_true',
                        default=False)

    parser.add_argument("--process_and_save_only", action='store_true',
                        default=False)
    parser.add_argument("--use_presaved_inference_blocks",
                        action='store_true', default=False)

    parser.add_argument("--do_inference", action='store_true', default=False)
    parser.add_argument("--do_training", action='store_true', default=False)
    parser.add_argument("--get_saliences", action='store_true', default=False)
    parser.add_argument("--prediction_folds", action='store_true',
                        default=False)
    parser.add_argument("--create_predictions", action='store_true',
                        default=False)
    parser.add_argument("--prepared_data_folder_name", type=str, default=None)
    parser.add_argument("--false_positive_weight", type=float, default=None)

    parser.add_argument("--model_version", type=str, default='torch')
    parser.add_argument("--run", type=int, default=None)

    parser.add_argument("--verbose", action='store_true',
                        default=False)

    parser.add_argument("--overwrite_prepared_block_files",
                        action='store_true', default=False)

    return parser


def balance_classes(labels, shuffle=False):
    """
    Returns the indices to keep of an array of labels, in order to balance
    the amount of labels per class.

    Parameters
    ----------
    labels : 1d array of ints
        The labels associated with each class.
    shuffle : bool, default False
        Whether to shuffle the current label indices before choosing the
        subset to keep.

    Returns
    -------
    keep_indices : 1d array of ints
        The indices across all labels to keep, with equal probability that a
        given sample is from any class.
    """
    # Get the unique labels.
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Reduce the amount of labels to the lowest occuring.
    amount_per_label = np.min(counts)

    keep_indices = []
    for cur_label in unique_labels:

        # Get the current indices for this label.
        cur_label_indices = np.where(labels == cur_label)[0]

        if shuffle:
            np.random.shuffle(cur_label_indices)

        # Select only a subset of this label
        keep_indices.append(cur_label_indices[:amount_per_label])

    # Concatenate and sort the indices.
    keep_indices = np.sort(np.concatenate(keep_indices))

    return keep_indices


def load_parameters_and_directories(model_type=None, cur_path=None,
                                    pargs=None, config_prefix='',
                                    config_path=None):
    """
    Load the initial parameters and directories based on the model
    configuration JSON and parsed arguments.

    Parameters
    ----------
    model_type : str
        The type of model. In the form "{supervised or unsupervised}_{main
        layer type}_model".
    cur_path : str
        The path of the repo.
    pargs : Namespace object from argparse
        The parsed arguments from argparse in the main script.
    config_prefix : str
        The prefix of the config JSON. Default is nothing, empty str of ''.

    Returns
    -------
    parameters : object
        Object with attributes that are the relevant parameters.
    dirs : object
        Object with attributes that are relevant directory information.
    """
    subject_split = re.findall('\d+|\D+', pargs.subject)

    # Determine BRAVO or EC subject.
    if subject_split[0] == 'EC':
        subject_type = 'grid_case'
    elif subject_split[0] == 'bravo':
        subject_type = 'bravo'

    # Open the path to the model configuration JSON.
    if config_path is None:
        model_config_path = op.join(
            cur_path,
            'utilities',
            '{}speech_detection_model_config.json'.format(config_prefix)
        )
    else:
        model_config_path = op.join(
            config_path,
            '{}speech_detection_model_config.json'.format(config_prefix)
        )
    with open(model_config_path, 'r') as f:
        model_configs = json.load(f)

    # Neat class to auto-attribute inputs and kwargs as attributes. Useful
    # for neater declaration of parameters.
    class relevant_parameters:
        @auto_attribute
        def __init__(self, subj_id, id, **kwargs):
            self.subject_id = subj_id
            self.id = int(id)

    # Attribute the relevant parameters.
    parameters = relevant_parameters(pargs.subject,
                                     subject_split[1],
                                     **model_configs[model_type][
                                         'model_parameters'],
                                     **model_configs[model_type][
                                         'training_parameters'],
                                     **model_configs['data_parameters'],
                                     **model_configs['subject_specific'][
                                         subject_type])

    # Keep dictionaries for wandb model tracking.
    parameters.model_parameters = model_configs[model_type]['model_parameters']
    parameters.training_parameters = model_configs[model_type][
        'training_parameters']

    # Assign parameters from pargs.

    ## Feature labels ##
    parameters.feature_labels = ast.literal_eval(pargs.feature_labels)

    # total_feature_labels are all the labels that will be used to train a
    # given model. this may encompass more features than will be on a
    # particular dataset prep.
    parameters.total_feature_labels = ast.literal_eval(
        pargs.total_feature_labels)

    # all_feature_labels are all the labels for this dataset including
    # garbage labels. feature_labels removes these labels.
    parameters.all_feature_labels = copy(parameters.feature_labels)
    parameters.feature_labels = [lab for lab in
                                 parameters.all_feature_labels if not
                                 lab.startswith('garbage')]

    parameters.paradigms = ast.literal_eval(pargs.paradigms)
    parameters.cable = ast.literal_eval(pargs.cable)
    parameters.num_gpus = pargs.num_gpus
    parameters.metric = pargs.metric
    parameters.balance_data = pargs.balance_data
    parameters.verbose = pargs.verbose
    parameters.project_name = pargs.project_name
    parameters.behav_quality = list(range(1, pargs.behav_quality + 1))

    if pargs.block_subjects is not None:
        parameters.block_subjects = ast.literal_eval(pargs.block_subjects)
    else:
        parameters.block_subjects = [pargs.subject for _ in range(len(
            parameters.paradigms))]

    if pargs.date_ranges is not None:
        date_ranges = ast.literal_eval(pargs.date_ranges)
        parameters.date_range_params = {
            'date_start'       : date_ranges[0],
            'date_stop'        : date_ranges[1],
            'include_stop_date': True
        }
    else:
        parameters.date_range_params = None

    if pargs.false_positive_weight is not None:
        parameters.false_positive_weight = pargs.false_positive_weight
        parameters.model_parameters['false_positive_weight'] = \
            pargs.false_positive_weight

    # Set the utterance set.
    parameters.stim_set = ast.literal_eval(pargs.utterance_sets)
    for cur_ss in range(len(parameters.stim_set)):
        if parameters.stim_set[cur_ss][0] == 'english_words2':
            parameters.stim_set[cur_ss] = ['english_words2_1',
                                           'english_words2_2',
                                           'english_words2_3']

    # Other dependent and detection parameters.
    parameters.per_gpu_batch_size = int(
        parameters.total_batch_size / parameters.num_gpus)
    parameters.model_type = model_type
    parameters.detection = model_configs['detection_parameters']
    parameters.int_window = int(np.ceil(parameters.window *
                                        parameters.default_sr))
    parameters.relevant_elecs = [int(e) for e in
                                 np.arange(parameters.grid_size)]
    parameters.subject_type = subject_type

    # Assign hyperopt detection parameters if present.
    if 'hyperopt_detection_parameters' in model_configs.keys():
        parameters.hyperopt_detection = model_configs[
            'hyperopt_detection_parameters']

    # Assign which data streams to use. The default is 'high_gamma'.
    if pargs.data_streams is None:
        parameters.data_streams = ['high_gamma']
    else:
        parameters.data_streams = ast.literal_eval(pargs.data_streams)

    # Save the date and time of the run.
    parameters.date = datetime.today().strftime('%Y%m%d.%H%M%S')

    # Define some directory relevant information.
    dirs = directories()
    dirs.out_folder_name = pargs.out_folder
    dirs.model_folder_name = pargs.model_folder
    dirs.prepared_data_folder_name = pargs.prepared_data_folder_name

    return parameters, dirs


def update_parameters_and_flags(parameters, flags, pargs=None):
    """
    Update parameters and flags, based on parameters and flags.

    Parameters
    ----------
    parameters : object
        Object with attributes that are the relevant parameters.
    flags : object
        Object with attributes that are the relevant flags.
    pargs : Namespace object from argparse
        The parsed arguments from argparse in the main script.

    Returns
    -------
    parameters : object
        Object with updated parameters.
    flags : object
        Object with updated flags.
    """
    # By default, attempt to set all overt blocks to use acoustic times.
    if parameters.paradigms[0] == 'overt':
        flags.real_times = True
    else:
        flags.real_times = False

    # If flagged to use cue times, flags.real_times is overridden
    if pargs.use_cue_times:
        flags.real_times = False

    # Whether stateful inference needs to be invoked.
    if parameters.model_type == 'supervised_lstm_model':
        flags.stateful_inference = True
    else:
        flags.stateful_inference = False

    # Whether specific folds are being used to generate detected times for
    # all blocks.
    if flags.prediction_folds:
        parameters.fold = pargs.prediction_fold

    return parameters, flags


def update_dependent_parameters(parameters):
    """
    Update parameters that are dependent on other parameters. This is useful
    if a couple parameters are changed by hand for testing.

    Parameters
    ----------
    parameters : object
        Object with attributes that are the relevant parameters.

    Returns
    -------
    parameters : object
        Same object, with a couple updated parameters. No new parameters.
    """

    # The number of samples that comprise a window, as an integer.
    parameters.int_window = int(np.ceil(parameters.window *
                                        parameters.default_sr))

    # The batch size for every GPU. Models trained across several GPUs will
    # have the total batch computed in smaller batches (across the GPUs).
    parameters.per_gpu_batch_size = int(
        parameters.total_batch_size / parameters.num_gpus)

    return parameters


def print_run_information(parameters, flags, logger):
    """
    Print some relevant information about the specific training or
    inference run.
    """
    logger.info('Date as of runtime: {}'.format(parameters.date))
    logger.info('Subject: {}'.format(parameters.subject_id))
    logger.info('Fully train model: {}'.format(flags.do_training))
    logger.info(
        'Do inference with pre-trained model: {}'.format(flags.do_inference))
    logger.info('*************************************\n')
    logger.info('MODEL PARAMETERS:')
    logger.info('Model type: {}'.format(parameters.model_type))
    logger.info('Layers: {}'.format(parameters.layers))
    logger.info('Nodes: {}'.format(parameters.nodes))
    logger.info('Batch size: {}'.format(parameters.total_batch_size))
    logger.info('Learning rate: {}'.format(parameters.learning_rate))
    logger.info('Dropout rate: {}'.format(parameters.dropout_rate))
    logger.info('Window: {} s'.format(parameters.window))
    logger.info('\n\n')
    pass


def boolean_to_times(bools, sr):
    """
    Convert boolean time courses of events to times that define the start
    and stop of events. Assumes t = 0 s at start of array.

    Parameters
    ----------
    bools : dict
        Dictionary of the boolean time courses with organization bools[block][
        event_type] yielding 1d boolean array with shape (time,).
    sr : float
        The sampling rate.

    Returns
    -------
    times : dict
        Same organization as bools but instead of a 1d boolean arrays of
        event time courses, there are 2d arrays of floats denoting the start
        (first column) and stop (second column) times of each event type.
        Arrays have shape of (num_events, 2).
    """
    times = {}

    for block_key in bools.keys():

        # Initialize dictionary for times.
        times[block_key] = {}

        for event_type in bools[block_key].keys():

            # Get the event time course for this block and event_type.
            block_event_type = bools[block_key][event_type].astype(int)

            # Get indices where there is a change in "state" (i.e. 0 to 1
            # and 1 to 0).
            switches = np.where(block_event_type[:-1] != block_event_type[1:])[
                0]
            switches += 1

            # Convert these indices to seconds.
            switches = np.round((1.0 / sr) * switches, decimals=3)

            # If there's an odd number of switches, this means there was an
            # onset of an event with no offset. Default that offset to the
            # end of the block.
            if len(switches) % 2 == 1:
                switches = np.concatenate([switches,
                                           np.array([len(block_event_type)])])

            # Append 2d array of start/stop times.
            times[block_key][event_type] = switches.reshape((-1, 2))

    return times


def write_model_description_json(parameters, blocks_train, blocks_val, out):
    """
    Writes json file with the appropriate fields for model_description.json.
    This version was updated on September 1, 2020 during RT task
    reorganization.

    Parameters
    ----------
    parameters : class
        Class object of all the parameters (as attributes) used for training
        the model.
    blocks_train : list
    blocks_val : list
    out : str

    Returns
    -------
        Write the file and does not return anything.
    """
    # Determine the "total relevant elecs" by multiplying the relevant elecs
    # by the number of data streams.
    parameters.total_relevant_elecs = []
    for cur_stream in range(parameters.num_data_streams):
        additive = cur_stream * parameters.grid_size
        new_elecs = list(additive + np.array(parameters.relevant_elecs))
        parameters.total_relevant_elecs.extend(new_elecs)

    # Make sure everything is int
    parameters.relevant_elecs = [int(e) for e in parameters.relevant_elecs]
    parameters.total_relevant_elecs = [int(e) for e in
                                       parameters.total_relevant_elecs]

    model_meta = {
        "model_class"         : str(parameters.rt_class),
        "input_streams"       : [parameters.input_streams],
        "training"            : {
            "utterance_set": parameters.stim_set,
            "paradigm"     : parameters.paradigms,
            "blocks"       : [int(b) for b in blocks_train]
        },
        "training_validation" : {
            "utterance_set": parameters.stim_set,
            "paradigm"     : parameters.paradigms,
            "blocks"       : [int(b) for b in blocks_val]
        },
        "constructor_kwargs"  : {
            "num_features"      : parameters.num_ecog_features,
            "data_dtype"        : "float32",
            "data_buffer_size"  : 15000,
            "likelihood_dtype"  : "float32",
            "initial_ignore_num": 0
        },
        "model_parameters"    : {
            "num_layers"      : parameters.layers,
            "num_nodes"       : [int(n) for n in parameters.nodes],
            "inference_window": 1,
            "lstm_type"       : parameters.lstm_type,
            "model_scope"     : parameters.model_scope
        },
        "data_parameters"     : {
            "num_ecog_features" : parameters.num_ecog_features,
            "num_event_features": parameters.num_event_features,
            "feature_labels"    : parameters.feature_labels,
            "relevant_elecs"    : parameters.total_relevant_elecs
        },
        "detection_parameters": {},
        "output_parameters"   : {
            "outputs"     : ["event_num", "start_index", "stop_index"],
            "output_dtype": "uint32"
        }
    }

    for event_type in parameters.detection.keys():

        if event_type == 'silence':
            continue
        elif type(parameters.detection[event_type]) is not dict:
            continue

        cur_det_params = parameters.detection[event_type]
        event_detection = {
            "smoothing_window_size" : cur_det_params['smooth_size'],
            "thresh"                : cur_det_params['prob_threshold'],
            "time_thresholding_size": cur_det_params['time_threshold'],
            "active_state_nums"     : [2],
            "before_points"         : 0,
            "after_points"          : 0
        }

        model_meta["detection_parameters"][event_type] = event_detection

    with open(op.join(out, 'model_description.json'), 'w') as f:
        json.dump(model_meta, f, indent=4)
    pass


def get_window_iterator(data, size=10):
    """
    Generate the sliding window iterator.

    Parameters
    ----------
    data : 2d array
        The data to be windowed with shape (time, features).
    size : int, default 10
        The size of the window in number of timepoints.

    Yields
    -------
    Sliding window over the data. Each iteration it returns a slice of the
    data (over the time dimension) of size "size".
    """
    a = iter(np.arange(data.shape[0]))
    sliding_indices = tuple(islice(a, size))
    if len(sliding_indices) == size:
        yield data[sliding_indices, :]
    for elem in a:
        sliding_indices = sliding_indices[1:] + (elem,)
        yield data[sliding_indices, :]


def get_windowed_major_matrix(data, window=10, single_block=False):
    """
    Get individual windows of data by applying a sliding window iterator.

    Parameters
    ----------
    data : list of 2d arrays
        List of the data blocks to be windowed, with each block having shape
        (time, features).
    window : int, default 10
        The size of the window in number of timepoints.

    Returns
    -------
    The windowed matrix with shape (num_samples, time_window, features).
    """
    final = []

    for block in data:
        # Get an iterator object to slide a window.
        slider = get_window_iterator(block, size=window)

        # Apply the sliding window to the block.
        windowed_block = [a for a in slider]
        windowed_block = np.stack(windowed_block, axis=0)

        # Append each block's trials.
        final.append(windowed_block)

    # Concatenate all the trials from all blocks together.
    if single_block:
        return final[0]
    else:
        return np.concatenate(final, axis=0)


def get_sliding_window_indices(num_samples=None, step=1, window_size=None):
    window = np.arange(window_size).astype(int)
    inputs = [[window[0], window[-1] + 1]]
    outputs = [window[-1]]

    for _ in range(num_samples - window_size):
        window = window + step
        inputs.append([window[0], window[-1] + 1])
        outputs.append(window[-1])

    return inputs, outputs
