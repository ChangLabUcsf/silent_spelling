"""
Dataloaders and the augmentations used for training models.
"""

import random
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import random
import copy


class WeightedBravoWindowDataset(Dataset):
    def __init__(self, X, Y, weights, transform=None, y_transforms=None):
        
        self.X = X
        self.Y = Y
        self.transform = transform
        self.weights = weights
        self.y_transform = y_transforms
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = copy.deepcopy(self.X[idx])
        if self.transform:
            if not self.y_transform: 
                return (self.transform(sample), self.Y[idx], self.weights[idx])
            else: 
                return (self.transform(sample), self.y_transform(self.Y[idx]), self.weights[idx])
        else:
            return (sample, self.Y[idx], self.weights[idx])

class BravoWindowDataset(Dataset):
    
    def __init__(self, X, Y, transform=None):
        """
        Args: 
            X: the X data
            Y: the Y data
            
            
        """
        self.X = X
        self.Y = Y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = copy.deepcopy(self.X[idx])
        if self.transform:
            return (self.transform(sample), self.Y[idx])
        else:
            return (sample, self.Y[idx])
        
    
"""
This has all of our data augmentations for pytorch, can be added. 

Jitter and blackcout I came up with, the rest is from Willett's handwriting paper
"""
class Jitter(object):
    """
    randomly select the default window from the original window
    scale the amt of jitter by jitter amt
    validation: just return the default window. 
    """
    def __init__(self, original_window, default_window, jitter_amt, sr=200, decimation=6, validate=False):
        self.original_window = original_window
        self.default_window = default_window
        self.jitter_scale = jitter_amt
        
        default_samples = np.asarray(default_window) - self.original_window[0]
        default_samples = np.asarray(default_samples)*sr/decimation
        
        default_samples[0] = int(default_samples[0])
        default_samples[1] = int(default_samples[1])
        
        self.default_samples = default_samples
        self.validate = validate
        
        self.winsize = int(default_samples[1] - default_samples[0])+1
        self.max_start = int(int((original_window[1] - original_window[0])*sr/decimation) - self.winsize)
        
        
    def __call__(self, sample):
        if self.validate: 
            return sample[int(self.default_samples[0]):int(self.default_samples[1])+1, :]
        else: 
            start = np.random.randint(0, self.max_start)
            scaled_start = np.abs(start-self.default_samples[0])
            scaled_start = int(scaled_start*self.jitter_scale)
            scaled_start = int(scaled_start*np.sign(start-self.default_samples[0]) + self.default_samples[0])
            return (sample[scaled_start:scaled_start+self.winsize])
        


        
class Blackout(object):
    """
    The blackout augmentation.
    """
    def __init__(self, blackout_max_length=0.3, blackout_prob=0.5):
        
        self.bomax = blackout_max_length
        self.bprob = blackout_prob
        
        
    def __call__(self, sample):
      
        blackout_times = int(np.random.uniform(0, 1)*sample.shape[0]*self.bomax)
        start = np.random.randint(0, sample.shape[0]-sample.shape[0]*self.bomax)
        if random.uniform(0, 1) < self.bprob: 
            sample[start:(start+blackout_times), :] = 0
        return sample
        
    
class AdditiveNoise(object):
    def __init__(self, sigma):
        """
        Just adds white noise.
        """
        self.sigma = sigma
        
    def __call__(self, sample):
        sample_ = sample + self.sigma*np.random.randn(*sample.shape)
        return sample_
        
class ScaleAugment(object):
    def __init__(self, low_range, up_range):
        self.up_range = up_range # e.g. .8
        self.low_range = low_range
        # print('scale', self.low_range, self.up_range)
#         assert self.up_range >= self.low_range
    def __call__(self, sample):
        multiplier = np.random.uniform(self.low_range, self.up_range)
        return sample*multiplier

class LevelChannelNoise(object):
    def __init__(self, sigma, channels=128):
        """
        Sigma: the noise std. 
        """
        self.sigma= sigma
        self.channels = 128
        
    def __call__(self, sample):
        sample += self.sigma*np.random.randn(1,sample.shape[-1]) # Add uniform noise across the whole channel. 
        return sample
    
def normalize(x, axis=-1, order=2):
    """
    This is from the keras source code https://github.com/keras-team/keras/blob/v2.7.0/keras/utils/np_utils.py#L77-L91
    
    Normalizes a Numpy array.
    Args:
      x: Numpy array to normalize.
      axis: axis along which to normalize.
      order: Normalization order (e.g. `order=2` for L2 norm).
    Returns:
      A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)


class Normalize(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        return normalize(sample)
    
class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)
    
    
class FlexibleBravoWindowDataset(Dataset): 
    def __init__(self, X, Y, lengths, transform=None):
        """
        Args: 
            X: the X data
            Y: the Y data
        """
        self.X = X
        self.Y = Y
        self.lens = lengths
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx]
        length = self.lens[idx]
        if self.transform:
            s,l = self.transform((sample, length))
            return (s, l, self.Y[idx])
        else:
            return (sample, length, self.Y[idx])