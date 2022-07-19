import numpy as np
import datetime
import bravo_ml
# from bravo_ml.data_loading.load_bravo_utils import get_date
from sklearn.model_selection import StratifiedKFold
import pickle

def remove_motor(x, y, b, d):
    """
    Remove any motor trials (for alphabet1_1 and alphabet1_2 comparisons)
    """
    newx = np.zeros_like(x)
    newy, newb, newd = [],[], []
    ct = 0
    for k, (xx, yy, bb, dd) in enumerate(zip(x, y, b, d)):
        if yy != 26: 
            newx[ct] = xx
            newy.append(yy)
            newb.append(bb)
            newd.append(dd)
            ct+=1
            
    newx = newx[:ct]
    return newx, np.array(newy), newb, newd
    
    
from collections import Counter, defaultdict
import numpy as np
def balance_classes(x, y, b, d): 
    newx = np.zeros_like(x)
    newy, newb, newd = [],[], []
    ct = 0
    c = Counter(y)
    minv = np.inf
    
    for k, v in c.items(): 
        minv= min(v, minv)
        
    classct = defaultdict(lambda:0)
    for ind in range(1, y.shape[0]): 
        ind = -ind
        yind = y[ind]
        classct[yind] +=1 
        if classct[yind] < minv: 
            newx[ct] = x[ind]
            newy.append(yind)
            newb.append(b[ind])
            newd.append(d[ind])
            ct +=1
            
    newx = newx[:ct]
    return newx, np.array(newy), newb, newd
            
def get_split(inds,x,y,b):
    """
    Go through the process of getting the split
    """
    newx = np.zeros_like(x)
    newy, newb = [],[]
    ct = 0
    for k, (xx, yy, bb) in enumerate(zip(x, y, b)):
        if k in inds: 
            newx[ct] = xx
            newy.append(yy)
            newb.append(bb)
            ct+=1
        
    newx = newx[:ct]
    return newx, np.array(newy), newb

from collections import Counter
def split_data(xyb, xyb_secondary, split_scheme, num_cvs, cv, num_samples):
    """
    split data into train and test sets
    
    """
        
    skf= StratifiedKFold(n_splits=num_cvs, shuffle=True, random_state=0)
    x,y, b = xyb
    
    if not num_samples is None:
        x = x[-num_samples:]
        y = y[-num_samples:]
        b = b[-num_samples:]
        
    skf.get_n_splits(x, y)
    
    ct = 0
    for tr_ind, te_ind in skf.split(x, y):
        if ct == cv: 
            break
        ct+=1
    
    X_tr, Y_tr, blocks_tr = get_split(tr_ind, x, y, b)
    X_te, Y_te, blocks_te = get_split(te_ind, x, y, b)
    
    if not (xyb_secondary is None):
        x,y, b = xyb_secondary
        if not num_samples is None:
            x = x[-num_samples:]
            y = y[-num_samples:]
            b = b[-num_samples:]
        if len(y) < 260:
            c = Counter(y)
            minc = 10000
            for k, v in c.items(): 
                if v < minc: 
                    minc = v
            if minc < num_cvs: 
                num_cvs = minc
                print('new num cvs', minc)
                skf= StratifiedKFold(n_splits=num_cvs, shuffle=True, random_state=0)
        skf.get_n_splits(x, y)
        ct = 0
        for tr_ind, te_ind in skf.split(x, y):
            if ct == cv: 
                break
            ct+=1
        X_te, Y_te, blocks_te = get_split(te_ind, x, y, b)
        
    return X_tr, Y_tr, blocks_tr, X_te, Y_te, blocks_te

def get_dates_from_blocks(block_list_mimed, date_dict_fp):
    """
    get the date that each block occured on
    """
    date = []

    with open(date_dict_fp, 'rb') as handle:
        date_dict = pickle.load(handle)
        print('loaded date dictionary')
    for b in block_list_mimed: 
     
        
        date.append(date_dict[b])
        
        
    return date

def load_data(filepath, start_date, end_date, date_dict_fp=None): 
    """
    Inputs: 
    filepath: the filepath to load stuff from
    start_date: The day that the data should start
    end_date: the day that the data should end.
    
    outputs: the data arrays, x, y, and blocks. between start and end date (if applicable)
    """
    X = np.load(filepath %'X')
    Y = np.load(filepath %'Y')
    blocks = np.load(filepath %'blx')
    
    dates = get_dates_from_blocks(blocks, date_dict_fp)
    print(start_date, end_date)
    if start_date is None and end_date is None: 
        return X, Y, blocks, dates
    
    if not start_date is None: 
        X_, y_, blocks_, dates_= np.zeros_like(X), [], [], []
        
        good_ct = 0
        for xx, yy, bb, dd in zip(X, Y, blocks, dates): 
            if dd > start_date: 
                X_[good_ct] = xx
                y_.append(yy)
                blocks_.append(bb)
                dates_.append(dd)
                good_ct +=1
        X = X_[:good_ct]
        Y= y_
        blocks = blocks_
        dates = dates_
    
    if not end_date is None: 
        X_, y_, blocks_, dates_ = np.zeros_like(X), [], [], []
        good_ct = 0
        for xx, yy, bb, dd in zip(X, Y, blocks, dates): 
            if dd < end_date: 
                X_[good_ct] = xx
                y_.append(yy)
                blocks_.append(bb)
                dates_.append(dd)
                good_ct +=1
        X = X_[:good_ct]
        Y= y_
        blocks = blocks_
        dates = dates_
    
    return X, np.array(Y), blocks, dates