from .load_bravo_utils import *

def load_paradigm_blocks(tr_paradigms,
                         te_paradigms,
                         id=1,
                         window=(-1, 3),
                        preprocess_args={'f':0},
                        detection_args={'f':0}, 
                        neural_data_type='zscore_hg'): 
    """
    Input: train and test paradigm_stim_tuples - tuples as follows ('overt', ['english_words2_1', 'english_words2_2']) etc.
    id - int: which bravo we are loading things for. (e.g. 1)
    window_size - tuple the window we want to load (e.g. (-1, 3))
    preprocess args - dictionary to pass into the preprocessing script. {'decimate_flag':True, 'decimate':6}
    
    Returns: x, y the data for that paradigm for that patient. 
    """
    
    tr_blocks = []
    for tups in tr_paradigms: 
        print(tups)
        paradigms = tups[0]
        stimset = tups[1]
        print(paradigms, stimset)
        tr_blocks_, _ = get_block_numbers(paradigms, stimset, id)
        tr_blocks.extend(tr_blocks_)
        
    # Do the same for the test paradigms
    te_blocks = []
    for tups in te_paradigms: 
        paradigms = tups[0]
        stimset = tups[1]
        _, te_blocks_ = get_block_numbers(paradigms, stimset, id)
        te_blocks.extend(te_blocks_)
        
    X_tr, Y_tr = load_blocks(tr_blocks, window, neural_data_type=neural_data_type, **detection_args)
        
    X_te, Y_te = load_blocks(te_blocks,window, neural_data_type=neural_data_type, **detection_args)
    
    X_tr, Y_tr = basic_preprocess(X_tr, Y_tr, **preprocess_args)
    X_te, Y_te = basic_preprocess(X_te, Y_te, **preprocess_args)
    
    return X_tr, Y_tr, X_te, Y_te