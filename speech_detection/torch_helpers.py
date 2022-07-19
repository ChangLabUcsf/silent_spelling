# -*- coding: utf-8 -*-
"""
Helper functions for PyTorch based models.

:Author: Jessie R. Liu
:Copyright: Copyright (c) 2020, Jessie R. Liu, All rights reserved.
"""

import torch
from torch import nn
from torch.utils import data


class LSTMWithDropout(nn.Module):
    def __init__(self, dropout=None, *args, **kwargs):
        """
        Modified LSTM layer with dropout.

        Parameters
        ----------
        dropout : float
            The probability of an element to be zeroed or dropped out.
        args
        kwargs
        """
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x, out_state = self.lstm(x)
        x = self.dropout(x)
        return x, out_state


class LSTMWithDropoutReLU(nn.Module):
    def __init__(self, dropout=None, *args, **kwargs):
        """
        Modified LSTM layer with dropout.

        Parameters
        ----------
        dropout : float
            The drop probability for applying dropout.
        args
        kwargs
        """
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.relu(self.dropout(x))
        return x


class ecogDataset(data.Dataset):
    def __init__(self, data_dict):
        """
        ECoG dataset with other attributes, inheriting from
        torch.utils.data.Dataset. To be used with torch.utils.data.DataLoader.

        Parameters
        ----------
        data_dict : dict
            Dictionary with keys corresponding to all aspects of data that
            wish to be used. 'ecog' must be a field of shape (samples, ?, ?).
        """
        # Assign data field for every kwarg.
        for data_key in data_dict.keys():
            setattr(self, data_key, data_dict[data_key])

        self.total_samples, _, _ = self.ecog.shape

        self.ecog = torch.tensor(self.ecog).float()
        self.labels = torch.tensor(self.labels).long()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'ecog'  : self.ecog[idx, :, :],
            'labels': self.labels[idx]
        }

        return sample


class fileDataset(data.Dataset):
    def __init__(self, trial_list):
        """
        Filename dataset with other attributes, inheriting from
        torch.utils.data.Dataset. To be used with
        torch.utils.data.DataLoader. This is meant to be used to get batches
        of filenames which are then loaded just prior to compute.

        Parameters
        ----------
        data_dict : dict
            Dictionary with keys corresponding to all aspects of data that
            wish to be used. 'ecog' must be a field of shape (samples, ?, ?).
        """
        self.total_samples = len(trial_list)
        self.filenames = trial_list

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.filenames[idx]
