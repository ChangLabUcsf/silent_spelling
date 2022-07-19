import numpy as np

def get_date_train_test_data(test_date, X_alphabet, Y_alphabet, date): 
    """
    This function basically gets the data corresponding to any date before the test date

    """
    train_X, train_Y, train_dates = [], [], []
    test_X, test_Y, test_dates = [], [], []
    
    
    for x, y, d in zip(X_alphabet, Y_alphabet, date):
        if d < test_date: 
            train_X.append(x)
            train_Y.append(y)
            train_dates.append(d)
            
        elif d == test_date: 
            test_X.append(x)
            test_Y.append(y)
            test_dates.append(d)
            
        
    return np.array(train_X), np.array(train_Y), train_dates, np.array(test_X), np.array(test_Y), test_dates


            
def extract_reps_from_test_data(X_te, Y_te, n_reps=0): 
    """
    Extracts a certain number of repetitions.
    """
    tr_inds, te_inds = [],[]
    
    for y in list(set(Y_te)):
        inds = np.where(Y_te == y)[0]
        tr_inds.extend(inds[:n_reps])
        te_inds.extend(inds[n_reps:])
        

    return X_te[tr_inds], Y_te[tr_inds], X_te[te_inds], Y_te[te_inds]
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

def standard_date_split(test_day, X_alphabet, Y_alphabet, date):
    X_alphabet= normalize(X_alphabet)
    X_tr, Y_tr, train_dates, X_te, Y_te, test_dates = get_date_train_test_data(test_day, X_alphabet, Y_alphabet, date)
    X_te_tune, Y_te_tune, X_te, Y_te = extract_reps_from_test_data(X_te, Y_te)
    X_te_dates = test_dates[:X_te.shape[0]]
    tune_dates = test_dates[:X_te_tune.shape[0]]
    
    return X_tr, Y_tr, train_dates, X_te, Y_te, X_te_dates, X_te_tune, Y_te_tune, tune_dates

def get_exp_name(args, english_words=False):
    """
    Makes an experiment name given a dictionary of args.
    
    input: args
    returns: string. 
    
    
    """
    if args['method'] == 'standard':
        if not args['not_from_pretrained']:
            res = args['method']
        else: 
            res =  'standard_not_from_pretrained'
    elif args['method'] == 'session_embedding':
        res = args['method'] + f"_one_hot_{args['session_one_hot']}_loss_weight_{args['date_loss_alpha']}_embed_dim_{args['embedding_dim']}"
    elif args['method'] == 'pruning':
        res = 'pruning_top_k_%d' %args['topk']
    
    elif args['method'] == 'norm' or (args['method'] == 'tent' and english_words):
        res =  'norm' + '_LN_%s' %(str(args['layernorm']))  
    
    elif args['method'] == 'tent' and not english_words: 
        res = 'tent' + '_LN_%s' %(str(args['layernorm'])) + '_labels_' + str(args['labels'])
    else: 
        res =  args['method']
        
    if args['normalize'] == False:
        res += '_no_keras_norm'
        
    if args['only_2021']: 
        res += '_2021_only'
    if args['charweight']:
        res += '_charweight'
    if args['ensemble']: 
        res += '_ensemble'
    if args.get('combo', False):
        res += '_HG_and_raw'
    if args.get('realtime_sim', False):
        res += '_realtime_raw_processing_sim'
    if args.get('zscore_raw', False):
        res += '_zscored_raw'
    elif args.get('car_raw', False):
        res += '_car_raw'
    if args.get('realtime_sim', False):
        if not args.get('car_raw', False) and not (args.get('zscore_raw', False)):
            res += 'noz_nocar'
    return res


def encode_dates(date_list): 
    """
    Takes in lists of dates 
    and encodes them
    
    Returns the lists of dates, but encoded. 
    """
    ct = 0
    enc_dict = {}
    for d in date_list: 
        for dd in d: 
            if not dd in enc_dict: 
                enc_dict[dd] = ct
                ct +=1
    
    res = []
    for d in date_list: 
        subres = []
        for dd in d: 
            subres.append(enc_dict[dd])
            
        subres = np.array(subres)
        res.append(subres)
        
    return res