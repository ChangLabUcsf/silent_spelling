# import keras
import os
# from .keras_dataloader import DataGenerator


def split_data(X, Y, split):
    
    """This is useful for doing 10-fold CV"""
    sss = StratifiedShuffleSplit(n_splits=10, test_size=.1, random_state=0)
    
    for i, (train_index, test_index) in enumerate(sss.split(X, Y)):
        
        for t in train_index: 
            if t in test_index: 
                assert False
        if i == split: 
            X_tr = X[train_index]
            Y_tr = Y[train_index]
            X_te = X[test_index]
            Y_te = Y[test_index]
            
            return X_tr, Y_tr, X_te, Y_te
