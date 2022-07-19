# -*- coding: utf-8 -*-
"""
RNN speech detection model class, using PyTorch 1.6.0.

:Author: Jessie R. Liu
:Copyright: Copyright (c) 2020, Jessie R. Liu, All rights reserved.
"""
import logging
import numpy as np
import os
import pathlib
import time
import torch
import torch.nn.functional as F
import wandb
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader

# Custom code.
from data import get_batch_from_trial_list
from detection_utils import process_speech_detection
from torch_helpers import LSTMWithDropout, fileDataset, ecogDataset
from utils import logger_config, check_memory_usage, boolean_to_times


class base_lstm_model(nn.Module):
    def __init__(self, nodes=None, num_input_features=None,
                 num_output_features=None, droprate=None):
        super().__init__()

        input_sizes = [num_input_features]
        input_sizes += list(nodes[:-1])

        self.lstms = nn.ModuleList()
        for cur_layer, (num_nodes, num_inputs) in enumerate(
                zip(nodes, input_sizes)):

            if cur_layer < (len(nodes) - 1):
                self.lstms.append(LSTMWithDropout(input_size=num_inputs,
                                                  hidden_size=num_nodes,
                                                  batch_first=True,
                                                  dropout=droprate))
            else:
                self.lstms.append(nn.LSTM(input_size=num_inputs,
                                          hidden_size=num_nodes,
                                          batch_first=True))

        self.dense = nn.Linear(in_features=num_nodes,
                               out_features=num_output_features)

    def forward(self, x):
        for layer in self.lstms:
            x, _ = layer(x)
        x = self.dense(x)
        x = x[:, -1, :]
        return torch.squeeze(x)


class inf_base_lstm_model(base_lstm_model):
    """
    this version just subclasses and rewrites the forward method
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        for layer in self.lstms:
            x, _ = layer(x)
        x = self.dense(x)
        return torch.squeeze(x)


class speech_detector:
    """
    Model class handling the building and execution of the LSTM speech
    detection model.
    """

    def __init__(self, parameters, flags, log_filename, use_wandb=True,
                 device='cuda:0', *args):
        """
        Initialize the model class.

        Parameters
        ----------
        parameters : object
            Object whose attributes are the relevant parameters.
        flags : object
            Object whose attributes contain boolean flags.
        """

        self.verbose = parameters.verbose
        self.device = device

        self.batch_time = []
        self.epoch_time = []
        self.block_inf_time = []

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

        self.sr = parameters.default_sr
        self.date = parameters.date
        self.subject = parameters.subject_id

        # wandb model tracking dictionaries
        self.model_parameters = parameters.model_parameters
        self.training_parameters = parameters.training_parameters
        self.project_name = parameters.project_name

        self.electrodes = parameters.relevant_elecs

        self.layers = parameters.layers
        self.nodes = parameters.nodes
        self.window_size = parameters.int_window
        self.num_input_features = parameters.num_input_tokens
        self.num_output_features = parameters.num_output_tokens
        self.feature_labels = parameters.feature_labels
        self.tolerance = parameters.loss_tolerance

        self.droprate = parameters.dropout_rate
        self.model_ind = parameters.model_ind
        self.lstm_type = parameters.lstm_type
        self.model_scope = parameters.model_scope

        self.num_gpus = parameters.num_gpus
        if self.num_gpus is not None:
            self.device_ids = [f'cuda:{i}' for i in range(self.num_gpus)]

        self.per_gpu_batch_size = parameters.per_gpu_batch_size
        self.batch_size = parameters.total_batch_size
        self.batches_per_epoch = parameters.batches_per_epoch

        self.learning_rate = parameters.learning_rate

        self.min_epochs = parameters.min_epochs
        self.max_epochs = parameters.max_epochs
        self.early_stopping = parameters.early_stopping
        self.stop_training = False

        self.detection = parameters.detection

        self.trials_per_file = parameters.trials_per_file

        self.reuse_flag = False
        self.store_weights = flags.store_weights
        self.save_logits = flags.save_logits

        # Whether to track training progress on wandb
        self.use_wandb = use_wandb

        # related to the loss function
        self.false_positive_weight = parameters.false_positive_weight

        if 'speech' in parameters.feature_labels:
            self.speech_label = np.where(np.array(
                parameters.feature_labels) == 'speech')[0][0]

        if 'motor' in parameters.feature_labels:
            self.motor_label = np.where(np.array(
                parameters.feature_labels) == 'motor')[0][0]

        if 'silence' in parameters.feature_labels:
            self.silence_label = np.where(np.array(
                parameters.feature_labels) == 'silence')[0][0]

        if 'preparation' in parameters.feature_labels:
            self.prep_label = np.where(np.array(
                parameters.feature_labels) == 'preparation')[0][0]

        # Define the model with training or inference mode.
        self.model = base_lstm_model(
            nodes=self.nodes,
            num_input_features=self.num_input_features,
            num_output_features=self.num_output_features,
            droprate=self.droprate
        )
        # self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model.to(self.device)

        self.inf_model = inf_base_lstm_model(
            nodes=self.nodes,
            num_input_features=self.num_input_features,
            num_output_features=self.num_output_features,
            droprate=self.droprate
        )
        self.inf_model.to(self.device)

        # Set the optimizer.
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=1e-5)

        # Set the learning rate scheduler.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, verbose=True)

    def loss_function(self, true, pred):
        """
        Penalize:
        - true speech timepoints predicted to be motor
        - true motor timepoints predicted to be speech

        Parameters
        ----------
        true
        pred

        Returns
        -------

        """

        # Calculate the loss over each batch.
        loss_calc = F.cross_entropy(input=pred, target=true, reduction='none')

        if self.false_positive_weight == 1.0:
            return torch.sum(loss_calc), torch.mean(loss_calc)

        else:
            # Get the class each sample is predicted to be.
            pred_classes = torch.argmax(F.softmax(pred, dim=-1), dim=-1)

            # Find which samples are predicted to be speech or motor.
            pred_speech = torch.eq(pred_classes, self.speech_label)
            pred_motor = torch.eq(pred_classes, self.motor_label)
            # pred_sil = torch.eq(pred_classes, self.silence_label)

            # Find which samples are truly speech or motor.
            true_speech = torch.eq(true, self.speech_label).int()
            true_motor = torch.eq(true, self.motor_label).int()
            true_sil = torch.eq(true, self.silence_label).int()

            # Things we want to penalize:
            #   - speech and motor overlapping
            #   - speech being predicted during silence
            pred_speech_true_motor_idx = torch.where(
                pred_speech + true_motor == 2)[0]
            pred_motor_true_speech_idx = torch.where(
                pred_motor + true_speech == 2)[0]
            pred_speech_true_sil_idx = torch.where(
                pred_speech + true_sil == 2)[0]
            all_fp_idx = torch.cat([pred_speech_true_motor_idx,
                                    pred_motor_true_speech_idx,
                                    pred_speech_true_sil_idx])
            reg_fp_loss = loss_calc[all_fp_idx]
            new_fp_loss = self.false_positive_weight * loss_calc[all_fp_idx]
            loss_sum = torch.sum(loss_calc) - torch.sum(reg_fp_loss) + \
                       torch.sum(new_fp_loss)

            loss_avg = torch.true_divide(loss_sum, true.shape[0])

            return loss_sum, loss_avg

    def accuracy(self, true, pred):

        pred_labels = torch.argmax(torch.softmax(pred, dim=-1), dim=-1)
        sum_acc = (pred_labels == true).float().sum()
        avg_acc = sum_acc / true.shape[0]

        return sum_acc, avg_acc

    def loss(self, input, true, mode=None, return_acc=False, inference=False):

        if inference:
            pred = self.inf_model(input)
        else:
            pred = self.model(input)

        loss_tuple = self.loss_function(true, pred)
        acc_tuple = self.accuracy(true, pred)

        if return_acc:
            return loss_tuple, acc_tuple

        else:
            return loss_tuple

    def _batch_evaluate(self, dataloader, num_samples, return_acc=False):
        """
        Use the training style graph to evaluate training data during
        epoch-ing. Evaluates the loss of the model by batch processing for
        large datasets.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader object
            A DataLoader object with a dataset object based on
            torch.utils.data.Dataset. Iterable to give batched data as a
            dictionary of incoming 'ecog' and corresponding 'labels'.
        batch_size : int
            The number of samples per batch.

        Returns
        -------
        avg_loss : float
            The average loss across all samples.
        """

        # Put model into inference mode.
        self.inf_model.eval()

        # Evaluate each batch of data.
        batch_loss = 0
        batch_acc = 0
        with torch.no_grad():
            for data in dataloader:
                (sum_loss, _), (sum_acc, _) = self.loss(
                    data['ecog'].to(self.device),
                    data['labels'].to(self.device),
                    mode='windowed',
                    return_acc=True)
                batch_loss += sum_loss.item()
                batch_acc += sum_acc.item()

        # Compute the average loss and accuracy across all samples.
        avg_loss = batch_loss / num_samples
        avg_acc = batch_acc / num_samples

        if return_acc:
            return avg_loss, avg_acc
        else:
            return avg_loss

    def _evaluate(self, dataset, model_dir, return_acc=False):
        """
        Evaluate the loss across many blocks, inference style.

        Parameters
        ----------
        data: zip'ed tuple of 3d arrays
            Zip'ed list of 3d arrays, for each block, with the incoming
            neural features and matching labeled event class labels. The
            neural data is of shape (1, n_time, n_elecs) and the events are of
            shape (n_time).

        Returns
        -------
        avg_loss : float or list
            The loss across all samples, either as an average or as a list.
        """

        # Initialize lists and variables.
        sample_loss = []
        sample_acc = []
        num_samples = []

        self.update_inference_graph(model_dir)
        self.inf_model.eval()

        with torch.no_grad():
            for _input, _target in dataset:
                (sum_loss, _), (sum_acc, _) = self.loss(_input.to(self.device),
                                                        _target.to(
                                                            self.device),
                                                        mode='inference',
                                                        return_acc=True,
                                                        inference=True)
                sample_loss.append(sum_loss.item())
                sample_acc.append(sum_acc.item())
                num_samples.append(_input.shape[1])

        loss = np.sum(sample_loss) / np.sum(num_samples)
        acc = np.sum(sample_acc) / np.sum(num_samples)

        # Round loss to number of decimal places in the tolerance
        loss = round(loss, len(str(self.tolerance).split('.')[1]))

        if return_acc:
            return loss, acc
        else:
            return loss

    def train(self,
              trial_map,
              x_val,
              y_val,
              output_model_dir=None,
              data_dir=None,
              training_blocks=None,
              validation_blocks=None,
              **kwargs):
        """
        Train the model.

        Parameters
        ----------
        trial_map : pandas.DataFrame

        x_val : 3d array of floats
            The neural validation data with shape (samples, window,
            electrodes).
        y_val : 1d array of ints
            Sparse labels for the validation data with shape (samples).
        output_model_dir : str
            Where to save the model to.
        model_tracking : boolean
            Whether to track the model via summary stats.

        Returns
        -------
        Nothing.
        """
        self.cur_epoch = 0
        self.model.to(self.device)
        self.output_model_dir = pathlib.Path(output_model_dir)
        model_fold_part = self.output_model_dir.parts[-1]
        try:
            _, model_fold, _, model_run = model_fold_part.split('_')
        except:
            raise ValueError(model_fold_part)
        self.temp_model_dir = os.path.join(self.output_model_dir.parent,
                                           f'temporary_model_{model_fold}'
                                           f'_{model_run}')
        if not os.path.isdir(self.temp_model_dir):
            os.mkdir(self.temp_model_dir)

        self.save_training_graph(self.temp_model_dir)

        # Create summary writer with wandb.
        if self.use_wandb:
            wandb.init(project=self.project_name)
            wandb.run.name = datetime.today().strftime('%Y_%m_%d-%H_%M_%S')
            # wandb.run.save()
            wandb.config.update(self.model_parameters)
            wandb.config.update(self.training_parameters,
                                allow_val_change=True)
            wandb.config.update({
                'train_blocks'         : training_blocks,
                'num_train_blocks'     : len(training_blocks),
                'validation_blocks'    : validation_blocks,
                'num_validation_blocks': len(validation_blocks),
                'false_positive_weight': self.false_positive_weight
            }, allow_val_change=True)

        # Initialize loss lists for epochs and the cross-validation losses.
        train_epoch_losses = []
        val_epoch_losses = []

        # Initialize number of "worse" epochs and the best loss (super high
        # so that it's beaten immediately and then kept updated.
        self.num_worse = 0
        self.val_best = 100000

        # Balance the trial dataset first.
        keep_idx = []
        unique_labels, num_unique_labels = np.unique(trial_map.event_label,
                                                     return_counts=True)
        num_to_keep = min(num_unique_labels)
        for cur_label in unique_labels:
            cur_idx = trial_map.loc[trial_map.event_label ==
                                    cur_label].index.values
            np.random.shuffle(cur_idx)
            keep_idx.append(cur_idx[:num_to_keep])
        keep_idx = np.concatenate(keep_idx)
        trial_map = trial_map.loc[keep_idx]

        # Create the training dataset.
        # trial_map_dataset = fileDataset(trial_map['filename'].values)
        trial_map_dataset = fileDataset(np.arange(trial_map.shape[0]))

        # trial_map_dataset = fileDataset(trial_map)
        train_trial_dataloader = DataLoader(
            trial_map_dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            num_workers=1
        )

        # Format the validation data tensors.
        x_val = [torch.tensor(b).float() for b in x_val]
        y_val = [torch.tensor(b).long() for b in y_val]

        # Get overall validation losses pre-training.
        pre_val_loss, pre_val_acc = self._evaluate(zip(x_val, y_val),
                                                   self.temp_model_dir,
                                                   return_acc=True)
        self.logger.info('Pre-training val loss: {}'.format(pre_val_loss))
        val_epoch_losses.append(pre_val_loss)

        # Track loss and accuracy.
        if self.use_wandb:
            wandb.log({
                'loss/val'   : pre_val_loss,
                'acc/val'    : pre_val_acc,
                'custom_step': 0
            })

        # Loop through the epochs.
        cur_iter = 1
        for i_epoch in range(self.max_epochs):
            self.cur_epoch = i_epoch

            self.logger.info(check_memory_usage(os.getpid()))

            epoch_t = time.time()

            # Keep track of the learning rate.
            if self.use_wandb:
                wandb.log({
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'custom_step'  : cur_iter
                })

            # collect all the data here, then cycle through
            all_batch_trials = []
            for cur_batch, batch_trials in enumerate(train_trial_dataloader):

                # TODO: need a more elegant solution to this
                if cur_batch == self.batches_per_epoch:
                    break

                all_batch_trials.extend(list(batch_trials))

            data_time = time.time()
            # batch_data = get_batch_from_trial_list(
            #     trial_map.iloc[all_batch_trials],
            #     data_dir=data_dir)
            batch_data = get_batch_from_trial_list(
                all_batch_trials,
                data_dir=data_dir,
                tm=trial_map
            )
            batch_data = ecogDataset(batch_data)
            print('Data prep. time: {:.3f} s.'.format(time.time() - data_time))

            data_loader_time = time.time()
            train_dataloader = DataLoader(batch_data,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=1)
            print('Data loader time: {:.3f} s.'.format(time.time() -
                                                       data_loader_time))

            # for cur_batch, batch_trials in enumerate(train_trial_dataloader):
            for cur_batch, train_batch in enumerate(train_dataloader):

                # TODO: this is because we might have more samples in our
                #  data loader than intended
                if cur_batch == self.batches_per_epoch:
                    break

                batch_t = time.time()

                # Run the training op.
                self.model.train()
                self.optimizer.zero_grad()

                (_, train_loss), (_, train_acc) = self.loss(
                    train_batch['ecog'].to(self.device),
                    train_batch['labels'].to(self.device),
                    mode='windowed',
                    return_acc=True
                )

                # Update summary writer.
                if self.use_wandb:
                    wandb.log({
                        'loss/train' : train_loss,
                        'acc/train'  : train_acc,
                        'custom_step': cur_iter
                    })

                train_loss.backward()

                # clip 'em like coupons
                nn.utils.clip_grad_norm_(self.model.parameters(), 1, 2)

                self.optimizer.step()
                self.batch_time.append(time.time() - batch_t)
                cur_iter += 1

            self.save_training_graph(self.temp_model_dir)

            self.epoch_time.append(time.time() - epoch_t)

            self.cur_train_loss = train_loss
            self.cur_val_loss, cur_val_acc = self._evaluate(
                zip(x_val, y_val),
                self.temp_model_dir,
                return_acc=True
            )
            train_epoch_losses.append(self.cur_train_loss)
            val_epoch_losses.append(self.cur_val_loss)

            # Step the scheduler if necessary.
            self.scheduler.step(self.cur_val_loss)

            # Update summary writer.
            if self.use_wandb:
                wandb.log({
                    'loss/val'   : self.cur_val_loss,
                    'acc/val'    : cur_val_acc,
                    'custom_step': cur_iter
                })

            # Print losses every epoch.
            self.logger.info(
                '\nEpoch {}/{}: train loss {:.3f}, val loss {:.3f}, '
                'time {:.3f} s'.format(
                    i_epoch + 1,
                    self.max_epochs,
                    self.cur_train_loss,
                    self.cur_val_loss,
                    time.time() -
                    epoch_t))

            if i_epoch < self.min_epochs:
                # If less than the minimum epochs, keep training.
                self.logger.info('continue: less than min_epochs')

            else:
                self.training_evaluation()
                if self.stop_training:
                    break

    def save_training_graph(self, model_dir):
        """Saves the training graph when the epoch is determined to be the
        best yet."""
        self.logger.info('Saving trained model.')
        torch.save(self.model.state_dict(),
                   os.path.join(model_dir, 'model.pt'))

    def update_inference_graph(self, model_dir):
        self.inf_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'model.pt')))

    def training_evaluation(self):
        """
        Evaluate whether to stop training or not. Saves the weights if
        necessary.
        """

        # If greater than the minimum epochs, evaluate whether to
        # continue training.
        if self.cur_val_loss < self.tolerance + self.val_best:

            if self.cur_val_loss < self.val_best:
                # Loss is getting better. Update the new best loss.
                self.val_best = self.cur_val_loss

            self.logger.info('Continue: current val_loss={:0.3f}, val_best={'
                             ':0.3f}'.format(self.cur_val_loss, self.val_best))

            # Reset the number of "worse" epochs, if it wasn't zero
            # already.
            self.num_worse = 0

            # Save fully trained model if the model is still getting
            # better.
            if self.store_weights:
                self.save_training_graph(self.output_model_dir)

        # Start initiating early stopping if it's 10% worse than the lowest
        # val loss so far.
        elif self.cur_val_loss >= self.tolerance + self.val_best:

            # Loss is not improving. Uptick the count of "worse" epochs.
            self.num_worse += 1

            if (self.num_worse > 0) and (
                    self.num_worse < self.early_stopping):
                # If not yet time for early stopping, print the losses.
                self.logger.info(
                    '{}<{} worse epochs: current val_loss={:0.3f}, '
                    'val_best={:0.3f}'.format(
                        self.num_worse, self.early_stopping,
                        self.cur_val_loss,
                        self.val_best))

            else:
                # If early stopping is necessary now, print the
                # losses and break out of the training loop.
                self.logger.info(
                    'Early stopping: current val_loss={:0.3f}, '
                    'val_best={:0.3f}'.format(
                        self.cur_val_loss, self.val_best))
                self.stop_training = True

    def sample_speed_test(self, x_test, load_model_dir=None, output_dir=None):

        self.model.load_state_dict(
            torch.load(os.path.join(load_model_dir, 'model.pt'),
                       map_location=torch.device('cpu')))
        self.model.cpu()
        self.model.eval()

        self.sample_time = []

        with torch.no_grad():
            for block in x_test:
                for cur_sample in range(block.shape[1]):
                    sample = np.squeeze(block[0, cur_sample, :])
                    sample = torch.tensor(sample.reshape((1, 1, -1))).float()

                    sample_t = time.time()
                    _ = self.model(sample)
                    self.sample_time.append(time.time() - sample_t)

        np.save(os.path.join(output_dir, 'sample_time'), self.sample_time)

    def inference(self,
                  x_test,
                  y_test,
                  blocks_order,
                  prefix=None,
                  load_model_dir=None,
                  output_dir=None,
                  metric='scores',
                  process_and_save_only=False,
                  block_breakdown=None):
        """
        Call the function to build the offline inference graph and perform
        inference.

        Parameters
        ----------
        x_test : list of 3d arrays
            The neural test data. 3d array of floats for each block (in a
            list) with shape (1, time, electrodes).
        y_test : list of 1d arrays
            The events test labels. List equal to the number of blocks,
            each of which has a 1d array with shape (time,). Contains the
            sparse labels.
        blocks_order : list of ints
            List of test blocks in order.
        prefix : str
            The relevant prefix for saving files.
        load_model_dir : str
            Path to load the model from.
        output_dir : str
            Path to save to.
        metric : str
            The metric to return. Either 'score' (detection scores),
            'loss', or 'both'.

        Returns
        -------
        The desired metrics and:
        block_predictions_event_times : dict
            Same organization as bools but instead of a 1d boolean arrays of
            event time courses, there are 2d arrays of floats denoting the
            start
            (first column) and stop (second column) times of each event type.
            Arrays have shape of (num_events, 2).
        """
        self.inf_model.to(self.device)
        self.load_model_dir = load_model_dir

        # Load the pretrained model and set it to evaluation mode.
        self.update_inference_graph(load_model_dir)
        self.inf_model.eval()

        # Initialize lists for tracking the raw logits, raw predicted
        # probabilities and the losses.
        block_prediction_logits = []
        block_predictions = []

        # Convert test neural data to to tensors. Will convert labels later
        # for loss calculations.
        x_test = [torch.tensor(b).float() for b in x_test]

        for cur_block, ecog_block in enumerate(x_test):
            # Predict the current test block. Handle different amount of
            # returned arguments.
            block_inf_t = time.time()
            block_prediction = self.inf_model(ecog_block.to(self.device))
            self.block_inf_time.append(time.time() - block_inf_t)

            # Collect the raw logits
            block_prediction_logits.append(
                block_prediction.detach().cpu().numpy())

            # Apply a softmax to the raw logits. Save this as the prediction
            # for this block.
            block_prediction_softmax = F.softmax(block_prediction, dim=-1)
            block_predictions.append(
                block_prediction_softmax.detach().cpu().numpy())

        if self.save_logits:
            np.save(
                os.path.join(output_dir,
                             '{}_pred_logits_inference'.format(
                                 prefix)), block_prediction_logits)

        if process_and_save_only:

            # Try to make output folder, unless it already exists.
            try:
                os.mkdir(os.path.join(output_dir, 'event_predictions'))
            except FileExistsError:
                print('Folder already exists.')

            processed_block_predictions = process_speech_detection(
                block_predictions,
                blocks_order,
                detection=self.detection,
                event_types=self.feature_labels
            )

            # Convert boolean event time courses to event times.
            block_predictions_event_times = boolean_to_times(
                processed_block_predictions, self.sr)

            # Save the predicted probabilities.
            save_path = os.path.join(
                output_dir, f'{prefix}_pred_probability_inference')
            np.save(save_path, block_predictions)

            save_path = os.path.join(
                output_dir, f'{prefix}_blocks_inference')
            np.save(save_path, blocks_order)

            save_path = os.path.join(
                output_dir, f'{prefix}_true_events_inference')
            np.save(save_path, y_test)

            return None, None, None
