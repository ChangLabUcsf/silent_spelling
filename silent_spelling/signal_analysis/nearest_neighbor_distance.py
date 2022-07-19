# -*- coding: utf-8 -*-
"""
Code to calculate the nearest class distance based on neural activity alone,
via a bootstrapping estimation method.

:Author: Jessie R. Liu
:Copyright: Copyright (c) 2021, Jessie R. Liu, All rights reserved.
"""

# Standard libraries
import itertools
import pickle

# Third party libraries
import numpy as np
from scipy import signal
from tensorflow import keras as K

# Custom packages
from RT.results import savedData
from RT.tasks import taskData
from RT.util import fileHandler

from utils import balance_classes

# Define the utterance set and paradigm
utterance_sets = ['alphabet1_1', 'alphabet1_2']
paradigm = 'mimed'
subject = 'bravo1'
data_stream = 'neural/hgarawcar200_running30'
downsample_trials = True
downsampling_factor = 6  ## factor to decimate 200 by to get 33.33 Hz
go_phase = 5
window = np.array([0, 2.5])
trial_limit = 1000
sig_thresh = 0.01
save_results = True

all_blocks = {}
for stim_set in utterance_sets:
    # Find relevant blocks
    blocks = savedData.filter_result_numbers(
        paradigm=[paradigm],
        utterance_set=[stim_set],
        behavioral_quality=[2, 3]
    )

    all_blocks[stim_set] = blocks

# Balance blocks between utterance sets
min_blocks = min([len(b) for b in all_blocks.values()])
for stim_set in utterance_sets:
    all_blocks[stim_set] = all_blocks[stim_set][:min_blocks]

# Get event label and num mapping for all utterance sets desired.
tmd = taskData.TaskMetadata()
event_info = {}

for stim_set in utterance_sets:
    text_words = tmd.loadUtteranceSet(stim_set, convert_to_text=True,
                                      remove_descriptors=True)[1]
    text_words = [w.lower() for w in text_words]
    text_labels = tmd.loadUtteranceSet(stim_set, convert_to_text=False)[1]

    # Create dictionary of the labels.
    event_info.update(dict(zip(text_labels, text_words)))

    if stim_set == 'alphabet1_2':
        event_info.update(dict(zip(range(4250, 4250 + 26), text_words)))

ecog_trials = {}
ecog_trial_labels = {}

for stim_set, blocks in all_blocks.items():

    ecog_trials[stim_set] = []
    ecog_trial_labels[stim_set] = []
    #     pre_silence_trials[stim_set] = []

    for block in blocks:

        try:
            with savedData.DataInterface(result_num=block,
                                         subject=subject) as di:
                ecog, sr = di.load_continuous_data(data_stream,
                                                   convert_to_array=True)
                events = di.events
        except KeyError:
            print(f'Skipping block {block}, no rawcar200.')
            continue

        for cur_trial in events.event_num.dropna().unique():
            # Get the label for this trial
            event_label = events.loc[(events.event_num == cur_trial) & (
                    events.phase_num == go_phase)].event_label.values[0]
            ecog_trial_labels[stim_set].append(event_info[int(event_label)])

            # Grab this trial
            go_time = events.loc[(events.event_num == cur_trial) & (
                    events.phase_num == go_phase)].elapsed_time.values[0]
            start = int(sr * (go_time + window[0]))
            stop = start + int(sr * sum(abs(window)))

            # Downsample the trials
            if downsample_trials:
                single_ecog_trial = signal.decimate(
                    ecog[start:stop, :],
                    downsampling_factor,
                    axis=0
                )
            else:
                single_ecog_trial = ecog[start:stop, :]

            ecog_trials[stim_set].append(single_ecog_trial)

    ecog_trials[stim_set] = np.stack(ecog_trials[stim_set], axis=0)
    ecog_trial_labels[stim_set] = np.array(ecog_trial_labels[stim_set])

    # Normalize the data along the electrode dimension
    norm1 = K.utils.normalize(ecog_trials[stim_set][:, :, :128],
                              axis=-1, order=2)
    norm2 = K.utils.normalize(ecog_trials[stim_set][:, :, 128:],
                              axis=-1, order=2)
    ecog_trials[stim_set] = np.concatenate([norm1, norm2], axis=2)

# Make sure equal reps of each word to start
assert len(np.unique(
    np.unique(ecog_trial_labels['alphabet1_1'], return_counts=True)[1])) == 1
assert len(np.unique(
    np.unique(ecog_trial_labels['alphabet1_2'], return_counts=True)[1])) == 1

# Balance the reps per word
min_reps = min([np.unique(
    np.unique(ecog_trial_labels[stim_set], return_counts=True)[1])[0] for
                stim_set in utterance_sets])

for stim_set in utterance_sets:
    ecog_trials[stim_set], ecog_trial_labels[stim_set] = balance_classes(
        ecog_trials[stim_set],
        ecog_trial_labels[stim_set],
        min_samples=min_reps
    )

if save_results:
    save_path = fileHandler.getSubResultFilePath(
        sub_dir_key='analysis',
        result_label='spelling_paper_signal_analyses',
        extension='.pkl',
        next_file_sub_label='alphabet1_1_vs_1_2_ecog_trials'
    )
    print('Saved to:', save_path)
    with open(save_path, 'wb') as f:
        pickle.dump({'ecog': ecog_trials, 'labels': ecog_trial_labels}, f)

# Get the word trials for each stim set
word_trials = {}
for stim_set in utterance_sets:

    word_trials[stim_set] = {}

    for word in np.unique(ecog_trial_labels[stim_set]):
        idx = np.where(ecog_trial_labels[stim_set] == word)[0]
        word_trials[stim_set][word] = ecog_trials[stim_set][idx, :, :]

# number of bootstrap iterations
B = 1000

confusion_dfs = {
    'alphabet1_1': [],
    'alphabet1_2': []
}
all_mapping = {}

for stim_set in utterance_sets:

    words = list(word_trials[stim_set].keys())
    mapping = dict(zip(words, range(len(words))))
    all_mapping[stim_set] = mapping

    for cur_bootstrap in range(B):
        # On each bootstrap iteration, estimate the confusion matrix

        confusion_dfs[stim_set].append(np.zeros((len(words), len(words))))

        for w1, w2 in itertools.combinations(words, 2):
            # Sample, with replacement the word trials
            x = word_trials[stim_set][w1][
                np.random.choice(range(min_reps), size=min_reps, replace=True),
                :, :].mean(0)
            y = word_trials[stim_set][w2][
                np.random.choice(range(min_reps), size=min_reps, replace=True),
                :, :].mean(0)
            distance = np.linalg.norm(x - y)

            # Keep track of the distance
            confusion_dfs[stim_set][cur_bootstrap][
                mapping[w1], mapping[w2]] = distance

        if cur_bootstrap % 100 == 0:
            print(f'{cur_bootstrap}/{B},', end='')

# if save_results:
#     save_path = fileHandler.getSubResultFilePath(
#         sub_dir_key='analysis',
#         result_label='spelling_paper_signal_analyses',
#         extension='.pkl',
#         next_file_sub_label='utterance_set_mapping'
#     )
#     print('Saved to:', save_path)
#     with open(save_path, 'wb') as f:
#         pickle.dump(all_mapping, f)

confusion_df_avg = {}
confusion_df_stack = {}
for stim_set in utterance_sets:
    A = np.mean(np.stack(confusion_dfs[stim_set], axis=0), axis=0)
    W = np.triu(A) + np.tril(A.T, 1)
    confusion_df_avg[stim_set] = W
    confusion_df_stack[stim_set] = np.stack(confusion_dfs[stim_set], axis=0)

# Find the nonzero row minimum (nearest neighbor distance in a sense)
nearest_class_distance = {}

for stim_set in utterance_sets:

    nearest_class_distance[stim_set] = []

    for cur_row in range(confusion_df_avg[stim_set].shape[0]):
        row = list(confusion_df_avg[stim_set][cur_row, :])
        row.remove(0.0)
        nearest_class_distance[stim_set].append(np.min(row))

if save_results:
    save_path = fileHandler.getSubResultFilePath(
        sub_dir_key='analysis',
        result_label='spelling_paper_signal_analyses',
        extension='.pkl',
        next_file_sub_label='alphabet1_1_vs_1_2_bootstrap_L2_distance'
    )
    print('Saved to:', save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(confusion_df_stack, f)
