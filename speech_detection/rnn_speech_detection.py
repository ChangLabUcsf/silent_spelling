# -*- coding: utf-8 -*-
"""
Real-time constructor for the LSTM speech detection model, using PyTorch 1.6.0.
:Author: Jessie R. Liu
"""

# Import standard libraries.
import json
import os

# Import third party libraries.
import torch


class distantLSTM(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        An LSTM layer is exactly the same as default but is one naming
        branch further (i.e. `model.lstm.lstm` instead of `model.lstm`) in
        order to keep with the training model naming scheme that incorporates
        dropout and such, and requires subclassed layers.

        Parameters
        ----------
        args
        kwargs
        """
        super().__init__()
        self.lstm = torch.nn.LSTM(*args, **kwargs)

    def forward(self, x, *args):
        """
        Optionally pass in hidden/cell states as *args.
        """
        x, out_state = self.lstm(x, *args)
        return x, out_state


class base_lstm_model(torch.nn.Module):
    def __init__(self, nodes=None, num_input_features=None,
                 num_output_features=None):
        super().__init__()

        input_sizes = [num_input_features]
        input_sizes += list(nodes[:-1])

        self.num_layers = len(nodes)

        self.lstms = torch.nn.ModuleList()
        for cur_layer, (num_nodes, num_inputs) in enumerate(
                zip(nodes, input_sizes)):

            if cur_layer < (len(nodes) - 1):
                # Using a subclassed LSTM layer to keep with the naming
                # conventions of the saved model.
                self.lstms.append(distantLSTM(input_size=num_inputs,
                                              hidden_size=num_nodes,
                                              batch_first=True))
            else:
                self.lstms.append(torch.nn.LSTM(input_size=num_inputs,
                                                hidden_size=num_nodes,
                                                batch_first=True))

        self.dense = torch.nn.Linear(in_features=num_nodes,
                                     out_features=num_output_features)
        self.softmax = torch.nn.functional.softmax
        self.model_states = [None for _ in range(len(nodes))]

    def forward(self, x):
        for cur_layer, layer in enumerate(self.lstms):
            # Uses the previously saved model states before overwriting with
            # the currently outputted states.
            x, self.model_states[cur_layer] = layer(
                x, *[self.model_states[cur_layer]])
        x = self.dense(x)
        x = self.softmax(torch.squeeze(x), dim=-1)
        return x


class PyTorchEventDetector:
    """
    Class to construct the LSTM Speech Detection Model. Loads configuration
    variables and constructs the PyTorch model for use during real-time,
    online experiments.
    """

    def __init__(self, restore_path=None, testing=False, **kwargs):
        """
        Constructor for TensorflowEventDetector.

        Parameters
        ----------
        restore_path : str
            Path to the directory containing model.ckpt files and the
            model_description.json configuration file.
        **kwargs
            Additional keyword arguments.
        """

        # Get the model number from the folder name.
        if not testing:
            self.model_num = int(os.path.basename(os.path.normpath(
                restore_path)).split('-')[0])

        # Load configuration file for the model.
        with open(os.path.join(restore_path, 'model_description.json')) as f:
            cfg = json.load(f)

        # Set model parameters.
        self.num_nodes = cfg["model_parameters"]["num_nodes"]
        self.inference_window = cfg["model_parameters"]["inference_window"]

        # Set data parameters.
        self.num_ecog_features = cfg["data_parameters"]["num_ecog_features"]
        self.num_event_features = cfg["data_parameters"]["num_event_features"]
        self.event_mapping = cfg["data_parameters"]["feature_labels"]
        self.relevant_elecs = cfg["data_parameters"]["relevant_elecs"]

        # Set the data type.
        self.training_dtype = cfg["constructor_kwargs"]["data_dtype"]

        # Get the model ensemble numbers (list).
        if "model_ensemble_nums" in cfg["constructor_kwargs"].keys():
            self.ensemble_nums = cfg["constructor_kwargs"][
                "model_ensemble_nums"]
        else:
            # If ensemble_nums was not specified, then there is just a
            # single model with no numerical suffix. Thus, use an empty
            # string as the model number.
            self.ensemble_nums = ['']

        # Store the number of models that we are ensembling.
        self.num_models = len(self.ensemble_nums)

        # Get the model paths. If there is only a single model, this will
        # just a list of length 1.
        self.model_paths = [os.path.join(restore_path, f'model{m}.pt') for m
                            in self.ensemble_nums]

    def build(self, device='cpu'):
        """
        Builds the PyTorch model, loads the pre-trained model, and handles
        predictions and model states.

        Returns
        -------
        predict : function
            Function to evaluate model predictions.
        """
        # Create the model.
        self.models = []
        for model_path in (self.model_paths):
            self.models.append(base_lstm_model(
                nodes=self.num_nodes,
                num_input_features=self.num_ecog_features,
                num_output_features=self.num_event_features
            )
            )

            # Load the pre-trained model and set it to evaluation mode.
            self.models[-1].load_state_dict(
                torch.load(model_path, map_location=torch.device(device))
            )
            self.models[-1].to(device)
            self.models[-1].eval()

        def ensemble_predict(input_seq):
            """
            Function to take in an input sample and return a prediction when
            there are multiple models to ensemble.

            Parameters
            ----------
            input_seq : nd-array of floats (float32)
                The input sample of ECoG data. The only dimension not of
                shape 1 should be the number of electrodes.

            Returns
            -------
            output : 1d-array of floats (float32)
                Output probabilities across the feature labels for the
                input sample, with shape (feature_labels,).
            """

            # Reshape and format the input sample.
            input_seq = torch.tensor(
                input_seq.reshape((1, self.inference_window, -1)),
                dtype=torch.float32
            )

            with torch.no_grad():
                output = torch.stack([m(input_seq) for m in self.models], 0)
                output = torch.mean(output, 0)

            return output.numpy()

        def predict(input_seq):
            """
            Function to take in an input sample and return a prediction when
            there is a single model.

            Parameters
            ----------
            input_seq : nd-array of floats (float32)
                The input sample of ECoG data. The only dimension not of
                shape 1 should be the number of electrodes.
            Returns
            -------
            output : 1d-array of floats (float32)
                Output probabilities across the feature labels for the
                input sample, with shape (feature_labels,).
            """

            # Reshape and format the input sample.
            input_seq = torch.tensor(
                input_seq.reshape((1, self.inference_window, -1)),
                dtype=torch.float32
            )

            with torch.no_grad():
                output = self.models[0](input_seq)

            return output.numpy()

        if self.num_models == 1:
            return predict
        else:
            return ensemble_predict
