# -*- coding: utf-8 -*-
"""
Code for loading, preparing, and returning data for speech detection.

:Author: Jessie R. Liu
:Copyright: Copyright (c) 2019, Jessie R. Liu, All rights reserved.
"""
import json
import logging
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch

# Custom code
from utils import logger_config


def get_batch_from_trial_list(trial_list, data_dir=None, tm=None):
    # should return a dictionary of numpy arrays
    # tm is the trial map
    # data_dir has the h5 files

    batch = {
        'ecog'  : [],
        'labels': []
    }
    tl = torch.stack(trial_list).numpy().astype(int)
    tm = tm.iloc[tl, :]

    for block_name, (start, stop), out_idx in zip(tm.block_name,
                                                  tm.input_index,
                                                  tm.output_index):
        with h5py.File(os.path.join(data_dir, block_name + '.h5'), 'r') as hf:
            ds = hf.get('ecog')
            batch['ecog'].append(ds[start:stop])
            ds = hf.get('events')
            batch['labels'].append(ds[out_idx])

    batch['ecog'] = np.stack(batch['ecog'], axis=0)
    batch['labels'] = np.array(batch['labels'])

    return batch


class data:

    def __init__(self, log_filename):

        logging.basicConfig(**logger_config)
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            fh = logging.FileHandler(log_filename)
            fh.setLevel(logging.INFO)
            fh_format = logging.Formatter(logger_config['format'],
                                          logger_config[
                                              'datefmt'])
            fh.setFormatter(fh_format)
            self.logger.addHandler(fh)

    # def get_blocks_and_load_data(self, FLAGS=None, dirs=None, params=None,
    #                              block_breakdown=None, args=None):
    #
    #     self._get_blocks(FLAGS=FLAGS, dirs=dirs, params=params, args=args)
    #     self._load_data(FLAGS=FLAGS, dirs=dirs, params=params,
    #                     block_breakdown=block_breakdown)

    def _get_blocks(self, FLAGS=None, dirs=None, params=None,
                    args=None, default_dataset=None):

        self.logger.info('Paradigms: {}'.format(params.paradigms))
        self.logger.info('Utterance set: {}'.format(params.stim_set))
        self.logger.info('Feature labels: {}'.format(params.feature_labels))

        if FLAGS.do_inference and FLAGS.use_presaved_inference_blocks and \
                (not FLAGS.do_training):
            # This is for the situation where we have a pretrained model and
            # wish to re-evaluate the inference. So we want to only load the
            # test blocks that were associated with that model training.
            test_blocks = []
            for ifold in range(params.num_folds):
                model_json = os.path.join(dirs.fold_models[ifold],
                                          'model_description.json')

                with open(model_json, 'r') as f:
                    model_description = json.load(f)
                    test_blocks.append(model_description['validation'][
                                           'blocks'])

            # Put the blocks into the form of all blocks and list of fold
            # indices.
            self.all_blocks = [np.concatenate(test_blocks)]
            self.test_indices = []
            for ifold in range(params.num_folds):
                self.test_indices.append(np.where(np.in1d(
                    self.all_blocks[0],
                    test_blocks[ifold]
                ))[0])

        else:

            if FLAGS.prediction_folds:

                # If using predefined prediction folds.
                block_split_dir = os.path.join(dirs.general_output_dir,
                                               params.subject_id)
                with open(os.path.join(block_split_dir,
                                       args.block_split_filename), 'rb') as f:
                    block_splits = pickle.load(f)[str(params.fold)]

                te_blocks = np.concatenate(list(
                    block_splits['testing'].values()))

                tr_blocks = []
                # Limit the number of training blocks from each stim set
                for para, stim_set in zip(params.paradigms, params.stim_set):

                    # See what kind of labeling was used for this block split
                    if f'{para}-{stim_set}' in block_splits['training'].keys():
                        set_label = f'{para}-{stim_set}'
                    else:
                        set_label = stim_set

                    if (args.num_train_blocks is not None) or (
                            args.num_val_blocks is not None):

                        num_avail = len(block_splits['training'][set_label])
                        if args.num_train_blocks is not None:
                            num_tr_blocks = args.num_train_blocks
                        elif args.num_val_blocks is not None:
                            num_tr_blocks = num_avail - args.num_val_blocks

                        sc = np.linspace(0, num_avail - 1,
                                         num=num_tr_blocks).astype(int)
                        tr_blocks_to_add = np.array(block_splits['training'][
                                                        set_label])[sc]

                    else:
                        tr_blocks_to_add = block_splits['training'][set_label]
                    tr_blocks.append(tr_blocks_to_add)

                val_blocks = []
                # Get the validation blocks from the train blocks
                for cur_set, (para, stim_set) in enumerate(
                        zip(params.paradigms, params.stim_set)):

                    # See what kind of labeling what used for this block split
                    if f'{para}-{stim_set}' in block_splits['training'].keys():
                        set_label = f'{para}-{stim_set}'
                    else:
                        set_label = stim_set

                    if 'validation' in block_splits.keys():

                        val_blocks_to_add = block_splits['validation'][
                            set_label]

                        if args.num_val_blocks is not None:
                            # Take a subset of those available blocks
                            sc = np.linspace(0,
                                             len(val_blocks_to_add) - 1,
                                             num=args.num_val_blocks).astype(
                                int)
                            val_blocks_to_add = val_blocks_to_add[sc]

                    else:
                        # Get the blocks that aren't in the training set
                        available_blocks = np.setdiff1d(
                            block_splits['training'][set_label],
                            tr_blocks[cur_set]
                        )

                        if args.num_val_blocks is not None:
                            # Take a subset of those available blocks
                            sc = np.linspace(0,
                                             len(available_blocks) - 1,
                                             num=args.num_val_blocks).astype(
                                int)
                            val_blocks_to_add = available_blocks[sc]
                        else:
                            val_blocks_to_add = available_blocks

                    val_blocks.append(val_blocks_to_add)

                val_blocks = np.concatenate(val_blocks)
                tr_blocks = np.concatenate(tr_blocks)

                self.all_blocks = [np.concatenate([tr_blocks, te_blocks,
                                                   val_blocks])]

                # Create indices so that this jives with the rest of the
                # code that uses indices. It just indexes all of the
                # blocks we already just chose.
                bool_train_indices = np.in1d(self.all_blocks[0], tr_blocks)
                self.training_indices = [
                    np.where(bool_train_indices == True)[0]]

                bool_val_indices = np.in1d(self.all_blocks[0], val_blocks)
                self.validation_indices = [
                    np.where(bool_val_indices == True)[0]]

                bool_test_indices = np.in1d(self.all_blocks[0], te_blocks)
                self.test_indices = [np.where(bool_test_indices == True)[0]]

    # def _load_data(self, FLAGS=None, dirs=None, params=None,
    #                block_breakdown=None):
    #
    #     # Update dependent parameters before starting to prepare the data.
    #     params = update_dependent_parameters(params)
    #
    #     # Load the data.
    #     self.ecog, self.event_times = data_loading_util(
    #         block_breakdown,
    #         parameters=params,
    #         all_blocks=self.all_blocks,
    #         data_dir=dirs.data,
    #         gen_data_dir=dirs.raw_features,
    #         real_times=FLAGS.real_times,
    #         acoustic_dir=dirs.acoustic,
    #         stim_dir=dirs.stimulus_dir,
    #         use_stimulus_lengths=FLAGS.use_stimulus_lengths,
    #         event_window=params.event_window
    #     )
    #
    #     # Define the number of neural and event features for model
    #     # construction.
    #     params.num_ecog_features = self.ecog[0][0].shape[-1]
    #     params.num_event_features = len(params.total_feature_labels)
    #     params.num_input_tokens = params.num_ecog_features
    #     params.num_output_tokens = params.num_event_features

    # def prepare_data_to_save(self, params=None):
    #     """
    #     TODO update
    #     This function is used to prepare data inference style (can be
    #     converted to training style when re-loaded just by windowing). Will
    #     save to the output dir in script where it's called.
    #
    #     Parameters
    #     ----------
    #     FLAGS
    #     params
    #     fold : int
    #         The fold this is being executed for.
    #     """
    #
    #     # Get the testing data for this fold.
    #     self.neural = np.array(self.ecog[0], dtype=object)
    #     event_times = np.array(self.event_times[0])
    #
    #     block_lengths = [block.shape[0] for block in self.neural]
    #
    #     # Prepare the data.
    #     res = data_preparation_util(
    #         y=event_times,
    #         total_lengths=block_lengths,
    #         parameters=params
    #     )
    #     self.events, self.trial_inds_inputs, self.trial_inds_outputs = res
    #
    #     # Get rid of the trial inds that correspond to garbage labels.
    #     if any([t.startswith('garbage') for t in params.all_feature_labels]):
    #
    #         for cur_block in range(len(self.events)):
    #             # Find indices that are and aren't NaN and Inf.
    #             nan_inds = np.where(np.isnan(self.events[cur_block]))[0]
    #             inf_inds = np.where(np.isinf(self.events[cur_block]))[0]
    #             not_nan_inds = np.where(~np.isnan(self.events[cur_block]))[0]
    #             not_inf_inds = np.where(~np.isinf(self.events[cur_block]))[0]
    #
    #             # Only keep time points that are not NaN and are not Inf.
    #             keep_inds = np.intersect1d(not_nan_inds,
    #                                        not_inf_inds).astype(int)
    #
    #             # Find the trials that correspond to those time points to
    #             keep.
    #             keep_inds = np.where(np.isin(
    #                 self.trial_inds_outputs[cur_block], keep_inds))[0]
    #
    #             # Keep the non-garbage label trial mappings. Use this later
    #             # to only save trials for these.
    #             self.trial_inds_outputs[cur_block] = \
    #                 self.trial_inds_outputs[cur_block][keep_inds]
    #             self.trial_inds_inputs[cur_block] = \
    #                 self.trial_inds_inputs[cur_block][keep_inds, :]
    #
    #             # Get the silence label.
    #             sil_label = np.where(np.array(params.feature_labels) ==
    #                                  'silence')[0][0]
    #
    #             # For saving the data, turn NaN into preparation label and
    #             # Inf into silence label.
    #             try:
    #                 prep_label = np.where(np.array(params.feature_labels) ==
    #                                       'preparation')[0][0]
    #                 self.events[cur_block][nan_inds] = prep_label
    #             except:
    #                 print('No preparation label, replacing NaN labels with '
    #                       'silence label instead.')
    #                 self.events[cur_block][nan_inds] = sil_label
    #
    #             self.events[cur_block][inf_inds] = sil_label

    def load_prepared_data(self, FLAGS=None, params=None, fold=None,
                           block_breakdown=None, data_dir=None):

        if FLAGS.do_inference or FLAGS.get_saliences:
            self.neural_test = []
            self.events_test = []

            # Get the testing data for this fold.
            self.blocks_for_testing = []
            for cur_set in range(len(self.all_blocks)):
                self.blocks_for_testing.append(np.array(
                    self.all_blocks[cur_set])[self.test_indices[fold]])

            self.blocks_for_testing = np.concatenate(self.blocks_for_testing)

            for block in self.blocks_for_testing:

                # TODO: old method is still down below when loading
                #  validation blocks
                block_name = '{}-{}-{}.h5'.format(
                    block,
                    block_breakdown[params.subject_id][str(block)]['date'],
                    block_breakdown[params.subject_id][str(block)]['time']
                )
                block_path = f'{data_dir}/{block_name}'
                try:
                    assert os.path.isfile(block_path)
                except AssertionError:
                    raise AssertionError(block_path)

                with h5py.File(block_path, 'r') as hf:
                    ecog_ds = hf.get('ecog')[()]
                    self.neural_test.append(np.expand_dims(ecog_ds, axis=0))

                    events_ds = hf.get('events')[()].squeeze()
                    self.events_test.append(events_ds.astype(np.int32))

            num_ecog_features = self.neural_test[0].shape[-1]

        if FLAGS.do_training:
            # Get the training data for this fold.
            self.blocks_for_training = []
            self.blocks_for_validation = []
            for cur_set in range(len(self.all_blocks)):
                self.blocks_for_training.append(np.array(self.all_blocks[
                                                             cur_set])[
                                                    self.training_indices[
                                                        fold]])
                self.blocks_for_validation.append(np.array(self.all_blocks[
                                                               cur_set])[
                                                      self.validation_indices[
                                                          fold]])
            self.blocks_for_training = np.concatenate(self.blocks_for_training)
            self.blocks_for_validation = np.concatenate(
                self.blocks_for_validation)

            self.neural_val = []
            self.events_val = []

            with pd.HDFStore(f'{data_dir}/trial_mapping.h5', 'r') as f:
                self.trial_map = f['trial_mapping']

            self.trial_map = self.trial_map.loc[self.trial_map.block.isin(
                self.blocks_for_training)]

            for block in self.blocks_for_validation:

                block_name = '{}-{}-{}.h5'.format(
                    block,
                    block_breakdown[params.subject_id][str(block)]['date'],
                    block_breakdown[params.subject_id][str(block)]['time']
                )
                block_path = f'{data_dir}/{block_name}'
                try:
                    assert os.path.isfile(block_path)
                except AssertionError:
                    print(f'Block {block_name} not prepared, skipping.')

                # # TODO: get rid of this
                # except:
                #     other_subj = params.block_subjects[1]
                #     block_name = '{}-{}-{}.h5'.format(
                #         block,
                #         block_breakdown[other_subj][str(block)]['date'],
                #         block_breakdown[other_subj][str(block)]['time']
                #     )
                #     block_path = f'{data_dir}/{block_name}'
                #     assert os.path.isfile(block_path)

                with h5py.File(block_path, 'r') as hf:
                    ecog_ds = hf.get('ecog')[()]
                    self.neural_val.append(np.expand_dims(ecog_ds, axis=0))

                    events_ds = hf.get('events')[()].squeeze()
                    self.events_val.append(events_ds.astype(np.int32))

            num_ecog_features = self.neural_val[0].shape[-1]

        # Define features based on the last opened block.
        params.num_ecog_features = num_ecog_features
        params.num_event_features = len(params.feature_labels)
        params.num_input_tokens = params.num_ecog_features
        params.num_output_tokens = params.num_event_features

    # def prepare_data(self, FLAGS=None, params=None, fold=None):
    #     """
    #     This method should be called on every fold, since train/val and test
    #     data are prepared differently.
    #
    #     Parameters
    #     ----------
    #     FLAGS
    #     params
    #     fold : int
    #         The fold this is being executed for.
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     if FLAGS.do_inference or FLAGS.get_saliences:
    #         # Get the testing data for this fold.
    #         ecog_test = []
    #         event_times_test = []
    #         self.blocks_for_testing = []
    #         for cur_set in range(len(self.all_blocks)):
    #             ecog_test.append(np.array(self.ecog[cur_set])[
    #                                  self.test_indices[fold]])
    #             event_times_test.append(np.array(self.event_times[cur_set])[
    #                                         self.test_indices[fold]])
    #             self.blocks_for_testing.append(np.array(self.all_blocks[
    #                                                         cur_set])[
    #                                                self.test_indices[fold]])
    #
    #         ecog_test = np.concatenate(ecog_test)
    #         event_times_test = np.concatenate(event_times_test)
    #         self.blocks_for_testing = np.concatenate(self.blocks_for_testing)
    #     else:
    #         ecog_test, event_times_test = [None], [None]
    #         self.blocks_for_testing = [0]
    #
    #     if FLAGS.do_training:
    #         # Get the training data for this fold.
    #         ecog_train = []
    #         event_times_train = []
    #         self.blocks_for_training = []
    #         for cur_set in range(len(self.all_blocks)):
    #             ecog_train.append(np.array(self.ecog[cur_set])[
    #                                   self.training_indices[fold]])
    #             event_times_train.append(
    #                 np.array(self.event_times[cur_set])[
    #                 self.training_indices[
    #                     fold]])
    #             self.blocks_for_training.append(np.array(self.all_blocks[
    #                                                          cur_set])[
    #                                                 self.training_indices[
    #                                                     fold]])
    #
    #         ecog_train = np.concatenate(ecog_train)
    #         event_times_train = np.concatenate(event_times_train)
    #         self.blocks_for_training = np.concatenate(
    #         self.blocks_for_training)
    #
    #         # Get the validation data for this fold.
    #         ecog_val = []
    #         event_times_val = []
    #         self.blocks_for_validation = []
    #         for cur_set in range(len(self.all_blocks)):
    #             ecog_val.append(np.array(self.ecog[cur_set])[
    #                                 self.validation_indices[fold]])
    #             event_times_val.append(
    #                 np.array(self.event_times[cur_set])[
    #                     self.validation_indices[fold]])
    #             self.blocks_for_validation.append(np.array(self.all_blocks[
    #                                                            cur_set])[
    #                                                   self.validation_indices[
    #                                                       fold]])
    #
    #         ecog_val = np.concatenate(ecog_val)
    #         event_times_val = np.concatenate(event_times_val)
    #         self.blocks_for_validation = np.concatenate(
    #             self.blocks_for_validation)
    #
    #     else:
    #         ecog_train, event_times_train = [None], [None]
    #         ecog_val, event_times_val = [None], [None]
    #         self.blocks_for_training = [0]
    #         self.blocks_for_validation = [0]
    #
    #     # Prepare the data.
    #     neural_pretrain, events_pretrain, self.neural_test, \
    #     self.events_test = data_preparation_util(
    #         x_pretrain=np.concatenate([ecog_train, ecog_val]),
    #         y_pretrain=np.concatenate([event_times_train, event_times_val]),
    #         x_test=ecog_test,
    #         y_test=event_times_test,
    #         parameters=params,
    #         flags=FLAGS
    #     )
    #
    #     if FLAGS.do_training:
    #         # Separate out into train and val again.
    #         self.neural_train = np.concatenate(neural_pretrain[:len(
    #             ecog_train)], axis=0)
    #         self.events_train = np.concatenate(events_pretrain[:len(
    #             ecog_train)], axis=0)
    #         self.neural_val = neural_pretrain[len(ecog_train):]
    #         self.events_val = events_pretrain[len(ecog_train):]
    #
    #         if params.balance_data:
    #
    #             self.logger.info('\nBefore balancing training set:')
    #             for cur_label, feature_label in enumerate(
    #                     params.feature_labels):
    #                 feature_percent = len(np.where(self.events_train ==
    #                                                cur_label)[0]) / len(
    #                     self.events_train)
    #                 self.logger.info(
    #                     '{}: {}%'.format(feature_label, 100 *
    #                     feature_percent))
    #
    #             train_balancing_inds = balance_classes(self.events_train,
    #                                                    shuffle=True)
    #             self.neural_train = self.neural_train[
    #             train_balancing_inds, :,
    #                                 :]
    #             self.events_train = self.events_train[train_balancing_inds]
    #
    #             self.logger.info('\nAfter balancing training set:')
    #             for cur_label, feature_label in enumerate(
    #                     params.feature_labels):
    #                 feature_percent = len(
    #                     np.where(self.events_train == cur_label)[0]) / \
    #                                   len(self.events_train)
    #                 self.logger.info(
    #                     '{}: {}%'.format(feature_label, 100 *
    #                     feature_percent))
    #
    #         # Convert validation data to block style.
    #         self.neural_val = [np.expand_dims(block_val[:, -1, :].squeeze(),
    #                                           axis=0)
    #                            for block_val in self.neural_val]
    #
    #         if 'garbage1' in params.all_feature_labels:
    #             events_train = np.copy(self.events_train)
    #             neural_train = np.copy(self.neural_train)
    #
    #             self.logger.info('Before removing garbage labels: ')
    #             self.logger.info('{}'.format(str(self.events_train.shape)))
    #             # For training blocks, take out nan's and inf's
    #             keep_inds = np.intersect1d(
    #                 np.where(~np.isnan(events_train))[0],
    #                 np.where(~np.isinf(events_train))[0]).astype(int)
    #
    #             self.events_train = events_train[keep_inds]
    #             self.neural_train = neural_train[keep_inds, :, :]
    #
    #             self.logger.info('After removing garbage labels: ')
    #             self.logger.info('{}'.format(str(self.events_train.shape)))
    #
    #             # For validation blocks, turn nan's into prep. Turn inf's
    #             into
    #             # silence.
    #             prep_label = np.where(np.array(params.feature_labels) ==
    #                                   'preparation')[0][0]
    #
    #             for b in range(len(self.events_val)):
    #                 event_block = np.copy(self.events_val[b])
    #                 nan_inds = np.where(np.isnan(event_block))[0]
    #                 inf_inds = np.where(np.isinf(event_block))[0]
    #
    #                 event_block[nan_inds] = prep_label
    #                 event_block[inf_inds] = 0
    #
    #                 self.events_val[b] = event_block
    #
    #     if FLAGS.do_inference or FLAGS.get_saliences:
    #
    #         if 'garbage1' in params.all_feature_labels:
    #             # For test blocks, turn nan's into prep. Turn inf's into
    #             # silence.
    #             prep_label = np.where(np.array(params.feature_labels) ==
    #                                   'preparation')[0][0]
    #
    #             for b in range(len(self.events_test)):
    #                 # event_block = np.copy(self.events_test[b])
    #                 nan_inds = np.where(np.isnan(self.events_test[b]))[0]
    #                 inf_inds = np.where(np.isinf(self.events_test[b]))[0]
    #
    #                 self.events_test[b][nan_inds] = prep_label
    #                 self.events_test[b][inf_inds] = 0
    #
    #                 # self.events_test[b] = event_block
