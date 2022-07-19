"""
Functions to split the data for bravo patients.
"""

import numpy as np

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

def validate_split(X, Y, data_amt, k, dates=None):
    skf = StratifiedKFold(n_splits=int(1/data_amt), random_state=0, shuffle=True)
    # just take the first one.
    
    if dates is None:
        for n, (train_index_, test_index_) in enumerate(skf.split(X, Y)): 
            if n == k:
    #             print('k', k)
                for ind in train_index_: 
                    assert not ind in test_index_
    #             print(test_index_)
                return X[train_index_], Y[train_index_], X[test_index_], Y[test_index_]
    
    else: 
        for n, (train_index_, test_index_) in enumerate(skf.split(X, Y)): 
            if n == k:
    #             print('k', k)
                for ind in train_index_: 
                    assert not ind in test_index_
    #             print(test_index_)
                return X[train_index_], Y[train_index_], [dates[t] for t in train_index_], X[test_index_], Y[test_index_], [dates[t] for t in test_index_]
        
        
def get_LC_split(X, Y, split, total_splits):
    """
    Given an X and Y array, evenly split it, for making learning curves. 
    
    """
    pct = (split+1)/total_splits
    np.random.seed(1337)
    end_inds = []
    for yy in set(Y): 
        inds = np.random.permutation(np.where(Y == yy)[0])
        end_inds.extend(inds[:int(len(inds)*pct)])
    end_inds = sorted(end_inds)
    return X[end_inds], Y[end_inds]
        
        

def realtime_split(X, Y, blocks, n_blocks=None): 
    
    """
    This simulates the scenario that we have in realtime: 
    eg. leaving out just blocks at the end for testing.
    
    TODO: in reality this should just give us the latest dates
    """
    
    all_blocks=sorted(list(set(blocks)))
    
    if not n_blocks == None: 
        pass
        
    else: 
        n_blocks = int(len(all_blocks)*.2)
        
    n_blocks = max(n_blocks, 1)

    print('n blocks', n_blocks)
        
    tr_blocks = all_blocks[:-n_blocks]
    te_blocks = all_blocks[-n_blocks:]
    print('trblocks', tr_blocks, 'teblocks', te_blocks)
    
    X_tr, Y_tr, blocks_tr = [], [], []
    X_te, Y_te, blocks_te = [], [], []
    for x, y, bl in zip(X, Y, blocks): 
        if bl in tr_blocks: 
            X_tr.append(x)
            Y_tr.append(y)
            blocks_tr.append(bl)
        else: 
            X_te.append(x)
            Y_te.append(y)
            blocks_te.append(bl)
        
        
    X_tr = np.asarray(X_tr)
    Y_tr = np.asarray(Y_tr)
    X_te = np.asarray(X_te)
    Y_te = np.asarray(Y_te)
    
    return X_tr, X_te, Y_tr, Y_te, blocks_tr, blocks_te
    
    
