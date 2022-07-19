"""
This is my basic dataloading script for the BRAVO patients.


Main functions are load_blocks - useful for gather which blocks are ready to be used. 
"""

import pandas as pd
import numpy as np

import os
from sklearn.preprocessing import LabelEncoder
import numpy
import pandas
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d
import json
import pickle
import scipy
grid_size = 128
exclude_dates = None

def get_certain_blocks(json_filename, id=1, paradigm=None, stim_set=['words'], default_dataset='training',
                       exclude_dates=[]):
    
    """
    Inputs: This takes in the json filename, 
    id = which bravo patient
    paradigm = 'overt, covert, etc'
    stim_set = 'english_words, english_aac1' etc.
    default_dataset = training.
    exclude_dates = list of dates to exclude. 
    
    Outputs: A list of block numbers corresponding to those dates. 
    """
    
    print('Loading blocks for patient %d' %id)
    with open(json_filename, 'r') as f:
        block_breakdown = json.load(f)
    block_breakdown = block_breakdown['bravo' + str(id)]

    print('Loading blocks with paradigm = ', paradigm)
    assert len(paradigm) == 1
    blocks = []
    if paradigm is None:
        print('no paradigm')
        assert False
        
        
    if stim_set is None:
        print('No stim set was listed')
        assert False

    if exclude_dates is None:
        exclude_dates = ['']

    if default_dataset is None:
        default_dataset = ['training', 'validation', 'testing']

    for key in block_breakdown.keys():
        if block_breakdown[key]['paradigm'] in paradigm:
            if block_breakdown[key]['utterance_set'] in stim_set:
                if block_breakdown[key]['date'] not in exclude_dates:
                    if block_breakdown[key]['default_dataset'] in default_dataset:
                        blocks.append(int(key))

    return sorted(blocks)

def get_date(block, id):
    """
    This function gets the date for each block for patient #id. 
    """
    json_filename = block_breakdown_path
    with open(json_filename, 'r') as f:
        block_breakdown = json.load(f)
    block_breakdown = block_breakdown['bravo' + str(id)]
#     print(block_breakdown.keys())
    if id == 2:
        blk = int(block)
    if True: 
        blk = str(int(block))
    return block_breakdown[blk]['date']

def get_paradigm(block, id): 
    """This function automatically gets the paradigm"""
    json_filename = block_breakdown_path
    with open(json_filename, 'r') as f:
        block_breakdown = json.load(f)
    block_breakdown = block_breakdown['bravo' + str(id)]
    return block_breakdown[str(block)]['paradigm']


# get fold blocks from block_breadowns.json
# block_breakdown_path = # get fold blocks from block_breadowns.json
block_breakdown_path = None # Fill in with your JSON_PATH '.../block_breakdowns.json'


def get_block_numbers(paradigms, stim_set, id=1, get_all_blx=False): 
    """
    This is a wrapper for the get_certain_blocks function. 
    
    You just need to input the paradigms and stimset, as well as the ID.
    
    Returns: the list of blocks for training - this comes from the 'training' and 'testing' subsets. 
    
    As well as the blocks for testing 'validtion'
    
    """
    blocks_for_training = get_certain_blocks(block_breakdown_path,
                                         id=id,
                                         paradigm=paradigms,
                                         stim_set=stim_set,
                                         default_dataset=['training', 'testing'],
                                         exclude_dates=exclude_dates)
    blocks_for_testing = get_certain_blocks(block_breakdown_path,
                                            id=id,
                                            paradigm=paradigms,
                                            stim_set=stim_set,
                                            default_dataset='validation',
                                            exclude_dates=exclude_dates)
    if get_all_blx: 
            blocks_for_training = get_certain_blocks(block_breakdown_path,
                                         id=id,
                                         paradigm=paradigms,
                                         stim_set=stim_set,
                                         default_dataset=None,
                                         exclude_dates=exclude_dates)

      
    # this just ascertains that we are not getting blocks that were meant for training. 
    for bl in blocks_for_training: 
        assert not bl in blocks_for_testing
        
    return blocks_for_training, blocks_for_testing

def zero_bad_chans_function(trial_data, block_num, id=1): 
    
    
    """
    Inputs: 
    trial data - the data for the trial you're using 
    block_num - the block number being used
    id - the patient we are working with. 
    
    Returns: 
    
    Trial data, but with data from the bad channel for that block zero'd out. 
    """
    json_filename = block_breakdown_path
    with open(json_filename, 'r') as f:
        block_breakdown = json.load(f)

    block_breakdown = block_breakdown['bravo' + str(id)]

    for key in block_breakdown.keys():
        if key == block_num: 
            bad_ch = block_breakdown[key]['bad_channels']
            bad_ch_0 = bad_ch
            trial_data[:, :, bad_ch_0] = np.zeros_like(trial_data[:, :, 0])
    return trial_data

def running_zscore(matrix, sr=200, win_size=30):
    """
    Applies a 30 second running zscore to the input data.

    Parameters
    ----------
    matrix : 2d array of float/int
        The data to z-score with shape (time, features)
    sr : int
        The temporal sampling rate of the data.
    win_size : int
        The size of the rolling window in seconds. Default is 30.

    Returns
    -------
    zscored_data : 2d array (same shape as incoming matrix)
        The zscored data.
    """
    # Defines the rolling window
    window = int(win_size * sr)

    EPSILON = 1e-15

    # Calculate rolling mean and standard deviation
    df = pd.DataFrame(data=matrix)
    running_mean = df.rolling(window, min_periods=1).mean()
    running_std = df.rolling(window, min_periods=1).std() + EPSILON

    # Calculate the z-scored data, replaces any NaN's with zeros
    zscored_data = ((df - running_mean) / running_std).fillna(0).values

    return zscored_data


def load_blocks(train_blocks, event_time_window, 
    neural_data_type='zscore_hg',
    zero_bad_chans=False, 
    detected_times=False,
    mic_flag=False, 
    exact_window=False,
    jrl_filepath =None,
    use_offset=False,
    offset_buff=0.0,
    additional_buffer=0.0,
    id=1,
    verbose=False, 
    paradigm=None,
    load_special_filepath=None,
    car_raw=False,
    decimate_raw=False,
    zscore_raw=False,
    return_full_block=False,
    return_rawcar=False,
    **kwargs): 
    
    """
    MAIN dataloader function for the bravo patients.
    
    Inputs: 
    
    train_blocks - the training blocks that we are using for the bravo patients. 
    event_time_window - the time window we're using for dataloading (e.g. -1, 3)
    neural_data_type - 'zscore_hg' what kind of neural data we want. Other options: raw, and just hg (no zscore). 
    zero_bad_chans - boolean telling us if we want to zero out the bad channels or not. 
    paradigm - the paradigm being used. This is important because this helps us extract the 
                'go cue' onset. Quite useful. 
    detected_times - boolean, whether or not to use detected times. 
    jrl_filepath - Filepath that jessie has to her data
    id = 1 or 2 (or ...), which patient's data we are loading. 
    verbose = if we want printed updates of all of the blocks loading processes. Useful for debugging. 
    use_offset  = boolean of if we want to use the detected offsets or not
    offset_buff  = float, how many seconds long the offset buffer should be. 
    additional_buffer = how much extra time to keep on either side of the 'detected' on/off. 
    
    
    Returns. A tensor of windowed_data, and corresponding labels.
    """
    
    X = None
   
    sessions = []
    blx_used = []
    trials_per_block = []
    
    # This will load the hdf5 files, as well as detected times files. 
    print('loading')
    print('BLX should be a thing...')
    
    if not paradigm == 'ganguly_motor':
        gimlet_dir = GIM_DIR #you will need to set this
    else: 
        gimlet_dir = GIM_DIR # will need to be set by user
    for idx, block in enumerate(train_blocks):
        print(block)
        if idx%10 ==0: 
            print('.', end = '')
        blx_used.append(block)
        rawcar=False # flag to check if we have rawcar
        prefixed = [filename for filename in os.listdir(gimlet_dir) if filename.startswith(str(block) + '-')]
        file_path = os.path.join(gimlet_dir, prefixed[0])
        with pd.HDFStore(file_path, 'r') as f:

            
            if not detected_times: 
                events   = f['behavior/events']
            raw1k               = f['neural/raw1k']
            
            
            if load_special_filepath is None:
                try:
                    hgacar200           = f['neural/hgacar200']
                    hgacar200_running30 = f['neural/hgacar200_running30']
                except Exception:
                    rawcar = True
                    hgacar200 = f['neural/hgarawcar200']
                    hgacar200_running30 = f['neural/hgarawcar200_running30']
                    
            #####
            # TODO: clean this up :D
            ####
            elif load_special_filepath.startswith('renorm'):
                try:
                    path = os.path.join(RENORM_FP + f"{block}_{load_special_filepath.split('_')[-1]}.npy")
                    print('path', path)
                    hgacar200 = np.load(path)
                    hgacar200_running30 = np.load(path)
                except Exception: 
                    print('no renormed found for this block %d' %block)
                    continue
            else: 
#                 hgacar200 = np.load(os.path.join(load_special_filepath, '%d-FIR_8-band_high_gamma_(1000_Hz)NO_CAR-hgacar200.npy' %block))
#                 hgacar200_running30 = np.load(os.path.join(load_special_filepath, '%d-FIR_8-band_high_gamma_(1000_Hz)NO_CAR-hgacar200_running30.npy' %block))
#           
#                     print(rawz.shape)
                print('loading raw from special filepath')
                try:
                    hgacar200 = np.load(os.path.join(load_special_filepath, f"{block}-filter1-rawcar200.npy"))
                    hgacar200_running30 = np.load((os.path.join(load_special_filepath, f"{block}-filter1-rawcar200_running30.npy")))
                except Exception: 
                    print('File not found, for custom numpy dataset')
                    print(os.path.join(load_special_filepath, f"{block}-filter1-rawcar200.npy"))
                    
                    print((os.path.join(load_special_filepath, f"{block}-filter1-rawcar200_running30.npy")))
                
            if paradigm =='ganguly_motor' and np.max(np.max(raw1k)) > 100:
                print('bad block', block)
                continue
            if detected_times: 
                jrl_basepath = jrl_filepath
                prefixed = [filename for filename in os.listdir(jrl_basepath) if filename.startswith(str(block) + '-')]
                try:
                    jrl_path = os.path.join(jrl_basepath, prefixed[0])
                    with open(jrl_path, 'rb') as f: 
                        data = pickle.load(f)
                except Exception:
                    print('BLOCK MISSING')
                    continue

            if mic_flag: 
                mic = f['analog']
            
        # Get the go cue corresponding to each paradigm. 
        if paradigm is None: 
            paradigm=get_paradigm(block, id)
        if paradigm == 'shadow_listen': 
            go_cue = 1.0
    
        if paradigm == 'overt' or paradigm =='covert_noshadow' or paradigm == 'mimed' or paradigm == 'attempted' or paradigm == 'attempted_motor': 
            go_cue = 5.0
        elif paradigm == 'mimed_rapid_spelling':
            go_cue = 5.0
        elif paradigm == 'shadow' or paradigm == 'shadow_mimed' or paradigm == 'shadow_then_mimed' or paradigm =='covert': 
            go_cue = 6.0
        elif paradigm == 'ganguly_motor':
            go_cue = 3.0
            print(go_cue)
        elif paradigm == 'rest':
            return hgacar200.as_matrix(), 0
        else: 
            print('Paradigm is not supported yet. Please make changes to the loading code to support this paradigm.')
            assert False

        if not detected_times: 
            onset_times = []
            labels = []
            for index, row in events.iterrows():
#                 print(row['event_label'])
                if row['phase_num'] == go_cue and row['state_num'] == 2.0: 
                    onset_times.append(row['elapsed_time'])
                    labels.append(row['event_label'])

        else:
            if paradigm != 'attempted_motor':
                onset_times = data['speech']['predicted_start']
                end_times = data['speech']['predicted_end']
                labels = data['speech']['event_label']
            else: 
                onset_times = data['motor']['predicted_start']
                end_times = data['motor']['predicted_end']
                labels = data['motor']['event_label']
                
            print('using det times')
            if verbose: 
                print('using detected times')
#                 print(onset_times, end_times)
#                 print(np.mean(np.asarray(end_times-onset_times)))
            
        if neural_data_type == 'hg': 
            if not load_special_filepath:
                neural = hgacar200.as_matrix()
            else: 
                neural = hgacar200
            sampling_rate_neural = 200
        elif 'raw' in neural_data_type: 
            neural = raw1k.as_matrix()
            
            sampling_rate_neural = 1000
            if car_raw:
                print('carring raw')
                neural -= np.mean(neural, axis=1, keepdims=True) # Car
            if decimate_raw:
                print('decimating raw')
                neural = scipy.signal.decimate(neural, 5, axis=0) # decimate
                sampling_rate_neural = 200
            if zscore_raw:
                print('zscoring raw')
                neural = running_zscore(neural) # zscore
        elif mic_flag and 'mic' in neural_data_type: 
            neural = mic.as_matrix()
            sampling_rate_neural = 30000
        else: 
            print(neural_data_type)
            if not load_special_filepath:
                neural = hgacar200_running30.as_matrix()
            else: 
                neural = hgacar200_running30
            if rawcar and not return_rawcar:
                print('using HG only from raw and hg stream')
                neural = neural[:, :128]
            sampling_rate_neural = 200


#         if verbose: 
#             print('time raw', neural.shape[0]/sampling_rate_neural)
#             print('time hga', hgacar200.as_matrix().shape[0]/sampling_rate_neural)
            
        print('sr', sampling_rate_neural)
        if type(onset_times) == int:
            onset_times = list(onset_times)
            

        # This will be the onset times in seconds e.g. [1.23, 4.55, 7.11, 12.01]
        onset_times = np.squeeze(np.asarray(onset_times))
        if len(onset_times.shape) == 0:
            onset_times = np.expand_dims(onset_times, axis=0)
        # If there are offset times (e.g. just the case for detected times, use them. )
        if detected_times: 
            offset_times = np.squeeze(np.asarray(end_times))
            if len(offset_times.shape) == 0:
                offset_times = np.expand_dims(offset_times, axis=0)
        print('n onsets', np.shape(onset_times))
        num_events = len(onset_times)
        
        event_time_window = event_time_window
        event_time_window_arg = event_time_window

        event_index_window = np.around(np.array(event_time_window) * sampling_rate_neural).astype(int)
        onset_inds = np.around(onset_times * sampling_rate_neural).astype(int)
        
        
        if use_offset or exact_window: 
            offset_inds = np.around(offset_times * sampling_rate_neural).astype(int)

        if not (exact_window or use_offset): 
            
            trial_data = np.empty(shape=(num_events, event_index_window[1] - event_index_window[0], neural.shape[1]), dtype=neural.dtype)
            
            for i, cur_onset in enumerate(onset_inds):
                if paradigm=='ganguly_motor' and (i == len(onset_inds)-1):
                    continue
                if detected_times == True and (use_offset or exact_window): 
                    init = np.zeros((event_index_window[1]-event_index_window[0], neural.shape[1]))

                    trial_data[i, :, :] = np.zeros((event_index_window[1]-event_index_window[0], neural.shape[1]))
                    offset_ind = offset_inds[i]
                    offset = offset_ind + offset_buff
                    if offset > cur_onset + event_index_window[1]: 
                        print('offset was beyond the edge', offset, cur_onset+event_index_window[1])
                        offset = min(offset, cur_onset + event_index_window[1])
                    data = neural[(cur_onset + event_index_window[0]):offset, :]
                    trial_data[i, :data.shape[0], :] = dat
                else: 
                    try:
                        trial_data[i, :, :] = neural[cur_onset + event_index_window[0] : cur_onset + event_index_window[1], :]
                    except Exception: 
                        print('end of data exception, this should only happen during ganguly data')
                        print('this means that the final block was too short...')

        else: 
            trial_data = []
            for i, (cur_onset, cur_offset) in enumerate(zip(list(onset_inds), list(offset_inds))):
                buf_ind = int(additional_buffer*sampling_rate_neural)
                data = neural[(cur_onset-buf_ind):(cur_offset+buf_ind), :]
                trial_data.append(data)
            
        labels = np.asarray(labels)
        labels = np.reshape(labels, (labels.shape[0], 1))
        
        if return_full_block: 
            return neural, onset_inds, labels

        if zero_bad_chans: 
            if verbose:
                print('zeroing bad channels')
            trial_data = zero_bad_chans_function(trial_data, block)
        
        if (idx ==0 or X is None):
            if not(exact_window or use_offset): 
                X = [trial_data]
            else: 
                X = trial_data
            Y = [np.asarray(labels)]
            blx = list(np.ones(len(labels))*block)
        else: 
            if not(exact_window or use_offset): 
                X.append(trial_data)
            else: 
                X.extend(trial_data)
            Y.append(np.asarray(labels))
            blx.extend(np.ones(len(labels))*block)
            
    print('done loading')
    if not (exact_window or use_offset):
        return np.asarray(np.concatenate(X, axis=0)), np.asarray(np.concatenate(Y, axis=0)), np.array(blx)
    else: 
        if X is None:
            return None, None, None
        return X, np.asarray(np.concatenate(Y, axis=0)), np.array(blx)

def balance_samples(X, Y): 
    """
    This is a function which balances the samples across a bunch of trials. 
    
    This is useful if we want to even out the data
    
    I would recommend weighting the loss function instead though - since then we are able to use all the data collected. 
    
    Inputs: X and Y data & label tensors.     
    Returns: X and Y but where X has all the same samples as Y. 
    """
    assert X.shape[0] == Y.shape[0]
    
    # Go through each sample and build up a thing of counts. 
    sample_counts = np.zeros(Y.shape[1])
    
    for sample in range(X.shape[0]):
        sample_counts[np.argmax(Y[sample])] +=1
    
    print('samples counts', list(sample_counts))
    
    
    balance_min = np.min(sample_counts)
    print('balance min', balance_min)
    
    X_f  = np.zeros((int(Y.shape[1]*balance_min), X.shape[1], X.shape[2])) # samples*min_samples. 
    Y_f = np.zeros((int(Y.shape[1]*balance_min), Y.shape[1])) # min_samples*samples
    
                                  
    sample_counts = np.zeros(Y.shape[1]) # reset sample counts to zero. 
    total_sample_ind = 0
    for sample in range(X.shape[0]):
        if sample_counts[np.argmax(Y[sample])] < balance_min: 
            sample_counts[np.argmax(Y[sample])] += 1 
            X_f[total_sample_ind] = X[sample]
            Y_f[total_sample_ind] = Y[sample]
            total_sample_ind +=1

    print('post balancing sample counts', list(sample_counts))
    
    return X_f, Y_f


def basic_preprocess(X, Y, sigma=20, 
                     decimate=5, 
                     sigma_flag=False, 
                     decimate_flag=False, 
                     encode_Y = False,
                     one_hot_Y = False, 
                     balance_samples_flag=False,
                    **kwargs):
                                  
    """
    This is a function which takes in the blocks you want and does the following. 
    
    It can decimate the blocks, as well as 
    smoothing them. 
    
    """

    print(Y.shape, 'Y shape basic preprocess ')
    print(X.shape, "X shape basic preprocess")
    encoder = LabelEncoder()
    encoder.fit(Y.ravel())
    encoded_Y =encoder.transform(Y.ravel())
    if one_hot_Y: 
        dummy_y = np_utils.to_categorical(encoded_Y)
    elif encode_Y: 
        dummy_y = encoded_Y
    else: 
        dummy_y = Y

    # apply smoothing and downsmapling. 
    if decimate_flag: 
        print('downsample factor:', decimate)
        X = signal.decimate(X, decimate, axis=1)
    if sigma_flag: 
        print('smoothing is happening')
        X = gaussian_filter1d(X, sigma, axis = 1)

    if balance_samples_flag: 
        print('WE BALANCING SAMPLES')
        X, dummy_y = balance_samples(X, dummy_y, single_run_flag)
    return X, dummy_y

def flatten_X_arrays(X): 
    """
    This basically just flattens arrays. 
    """
    X = np.reshape(X,(X.shape[0], X.shape[1]*X.shape[2]))
    return X


def append_session_id(X, block_ids, scale): 

    # One hot encode the sessions
    sessions = block_ids
    for k in range (number_of_sessions):
        sessions.append(k)
    Y = np.asarray(sessions)
    encoder = LabelEncoder()
    encoder.fit(Y.ravel())
    encoded_Y =encoder.transform(Y.ravel())
    dummy_y = np_utils.to_categorical(encoded_Y)
    dummy_y = dummy_y[:-number_of_sessions]
    dummy_y = dummy_y*scale
    X = np.hstack((X, dummy_y))
    return X

def get_bands(X, frequency_band_array, sampling_rate, keep_raw = False, zscore_flag = True): 
    X_original = X
    for idx, frequency_band in enumerate(frequency_band_array): 
        bands = [frequency_band[0], frequency_band[1]]
        print(bands)
        new = return_freq_power(X_original, sampling_rate, bands)
        if idx == 0: 
            if keep_raw: 
                X_f = np.concatenate((X_original, new), axis = -1)
            else: 
                X_f = new
            print(X_f.shape)
        else: 
            X_f = np.concatenate((X_f, new), axis = -1)
    print(X_f.shape)
    if zscore_flag: 
        X_f = zscore(X_f, axis = 1)
    return X_f

def return_freq_power(X, samplingrate, bands): 
    # Input: X. Raw data, s.r, 
    low = bands[0]
    high = bands[1]
    sos = signal.butter(3, [low, high], btype='bandpass', output='sos', fs = samplingrate)
    filtered = signal.sosfilt(sos, X, axis = 1)
    filtered = np.abs(signal.hilbert(filtered))
    return filtered


import numpy as np

def window_encode_Y(y_tr, y_te=None): 
    """
    This is a helper function that takes in y_tr and y_te,
    it then gets everything together, makes an encoding dictionary (and reverse encoding dictionary)
    
    Then
    """
    
    import h5py
    f = h5py.File('/path/to/metadata.h5', 'r')
    
    if not y_te is None:
        all_ys =np.concatenate((y_tr, y_te), axis=0)
    else: 
        all_ys = y_tr

    all_ys = [y[0] for y in all_ys]
    all_ys = list(all_ys)
    all_ys = sorted(list(set(all_ys)))
    
    
    all_utt = [f['utterance_data']['all_utterances'][y] for y in all_ys]
    
    all_utt = sorted(list(set(all_utt)))
    
    
    utt_enc = {v:k for k, v in enumerate(all_utt)} # Go from utterance to ind
#     enc_utt = {v:k for k, v in utt_enc.items()}
    import copy
    y_og = copy.deepcopy(y_tr)
    y_tr = [f['utterance_data']['all_utterances'][y[0]] for y in y_tr]
    enc_utt= {}
    for y, y_nu in zip(y_og, y_tr):
        enc_utt[y[0]] = y_nu
    
    if not y_te is None:
        y_te = [f['utterance_data']['all_utterances'][y[0]] for y in y_te]
    
    
    y_tr = [utt_enc[y] for y in y_tr]
    if not y_te is None:
        y_te = [utt_enc[y] for y in y_te]
    
    return np.array(y_tr), np.array(y_te), utt_enc, enc_utt



def car(array, median=False): 
    """
    input: array: array you want to CAR (comman average reference)
        median, wether to use median or mean. 
        
    returns: array, the Common average referenced array :D. 
    """
    for k, xx in enumerate(array):
        if not median:
            array[k] -= np.mean(xx, axis=-1, keepdims=True)
        else: 
            array[k] -= np.median(xx, axis=-1, keepdims=True)
            
    return array
        
        
        
