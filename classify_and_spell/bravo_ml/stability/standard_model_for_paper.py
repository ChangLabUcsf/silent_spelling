import bravo_ml
from bravo_ml.modelling import gpu_utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_gpu_lowest_memory())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
# !nvidia-smi

# %load_ext autoreload
# %autoreload 2
# import bravo_ml
from bravo_ml.modelling import gpu_utils
import os
from bravo_ml.data_loading.load_bravo_windows import load_paradigm_blocks, load_blocks
from bravo_ml.data_loading.load_bravo_utils import get_block_numbers
from bravo_ml.data_loading.splitters import validate_split
from bravo_ml.modelling.models import mod_input_shape, get_bravo_models
from bravo_ml.data_loading.load_bravo_utils import window_encode_Y
from bravo_ml.stability.utils import get_exp_name
# from bravo_ml.stability 
import wandb
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import argparse
import numpy as np
from bravo_ml.stability.utils import standard_date_split
from bravo_ml.modelling.torch_models import CnnRnnClassifier 
from bravo_ml.modelling.torch_dataloader import Jitter, Blackout, AdditiveNoise, LevelChannelNoise, ToTensor, ScaleAugment, Normalize
from bravo_ml.modelling.torch_trainers import model_predict, train_classifier
from bravo_ml.data_loading.splitters import validate_split
from torchvision import transforms
import torch
from bravo_ml.modelling.torch_dataloader import BravoWindowDataset, WeightedBravoWindowDataset
import datetime
import os
from bravo_ml.data_loading.load_bravo_windows import load_paradigm_blocks, load_blocks
from bravo_ml.data_loading.load_bravo_utils import get_block_numbers
from bravo_ml.data_loading.splitters import validate_split
from sklearn.model_selection import StratifiedKFold
from datetime import datetime


def load_alpha_data(test_day, X_alphabet, Y_alphabet, blocks_alphabet, args=None):
    
    from bravo_ml.data_loading.load_bravo_utils import get_date
    from datetime import datetime
    test_day = datetime.strptime(test_day, "%Y_%m_%d")

    date = []
    date_dict = {}
    for b in blocks_alphabet: 
        if not b in date_dict: 
            date_dict[b] = get_date(b, id=args.get('id', 1))

        date.append(date_dict[b])


    # Experiment one: Run the benchmark (e.g. no additional training)
    dates = sorted(list(set(date)))
    import datetime

    if type(date[0]) == str:
        date = [datetime.datetime.strptime(d, "%Y_%m_%d") for d in date]
        
    X_tr, Y_tr, dates_tr = [], [], []
    X_te, Y_te, dates_te = [], [], []
    
    for x, y, d in zip(X_alphabet, Y_alphabet, date):
        
        if d < test_day:
            if (not args['only_2021']) or (d > datetime.datetime(2021, 3, 1, 1, 1,1)):
                X_tr.append(x)
                Y_tr.append(y)
                dates_tr.append(d)
        elif d == test_day:
            X_te.append(x)
            Y_te.append(y)
            dates_te.append(d)
            
    return np.array(X_tr), np.array(Y_tr), dates_tr, np.array(X_te), np.array(Y_te), dates_te

### Step 1. Load up a model with english 
# words and pretrain it using those 
# new augmenetations? Is this really necessary? 

# from ray import tune

# _= load_eng_data()
# _ =load_alpha_data()

from bravo_ml.modelling.torch_models import CnnRnnClassifier

def train_english_words(args):
    
    ####
    #
    ####
    
    if args['from_pretrained']:
        n_targ = 50
    else: 
        n_targ = args['n_targ']
    
    model = CnnRnnClassifier(rnn_dim=int(args['decode_nodes']),
                            KS=int(args['ks']),
                            num_layers=int(args['decode_layers']),
                            dropout=args['dropout'],
                            n_targ=n_targ,
                            bidirectional=True,
                            in_channels=args.get('in_channels', 128))
        
    if args['from_pretrained']:
        train_loader, test_loader = get_alpha_loaders(args, english_words=True)
        best_acc = 0.0
        best_model=None
        patience = 0
        train_flag = True
        model = model.to(args['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(40):
            if train_flag:
                model, acc, loss = model_step(model, optimizer, train_loader, test_loader, best_acc, epoch)
            best_acc = max(acc, best_acc)
            if best_acc == acc:
                patience = 0
                best_model = copy.deepcopy(model)
            else: 
                patience += 1
            if patience == 5:
                train_flag = False
                
#             tune.report(mean_accuracy = best_acc/4)
            
        best_model.dense = torch.nn.Linear(int(2*args['decode_nodes']), int(30))
            
        print('best english words acc', best_acc)
        return best_model, best_acc
            
        
    return model, 0.0
    
    
def assign_weights(X, args):
    
    """
    Inputs: X: the array that you'll be using.
    group_amts: The amt of data in each group (random numbers, to be summed up.)
    group_weights: the weight of each group [0-10]
    
    
    """
    
    total_samples = X.shape[0]
    return np.ones(total_samples)
    
#     group_amts= []
#     group_weights =[]
#     for k in range(args['n_groups']):
#         group_amts.append(args['group_%d_pct' %k])
#         group_weights.append(args['group_%d_weight' %k])
#     pcts = group_amts/np.sum(group_amts)
#     weights = np.ones(total_samples)
    
#     start_ind = 0
#     for k, pct in enumerate(pcts): 
#         curr_samps = int(pct*total_samples)
#         end_ind = start_ind + curr_samps
        
#         weights[start_ind:end_ind] = group_weights[k]
#         start_ind = end_ind
        
#     return weights
    
def get_alpha_loaders(args, X_tr, Y_tr, blocks_tr, 
                      X_te, Y_te, blocks_te, english_words=False):
        
    
    train_jitter = Jitter((-2.0, 4.0), (args['winstart'], args['winend']), jitter_amt=args['jitter_amt'])
    test_jitter = Jitter((-2.0, 4.0), (args['winstart'], args['winend']), jitter_amt=0.0)
    blackout = Blackout(args['blackout_len'], args['blackout_prob'])
    noise = AdditiveNoise(args['additive_noise_level'])
    chan_noise = LevelChannelNoise(args['random_channel_noise_sigma'])
    scale = ScaleAugment(args['scale_augment_low'], args['scale_augment_high'])
    normalize = Normalize()
    
    composed = transforms.Compose([
        normalize, train_jitter,  blackout , noise, chan_noise, scale
    ])
    
    test_augs = transforms.Compose([
        normalize, test_jitter
    ])
    
    train_weights = assign_weights(X_tr, args)
    test_weights = np.ones(X_te.shape[0])
    
    train_dset = WeightedBravoWindowDataset(X_tr, Y_tr, train_weights, composed)
    test_dset = WeightedBravoWindowDataset(X_te, Y_te, test_weights, test_augs)
    
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=64, 
                                         shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=64,
                                             shuffle=False)
    return train_loader, test_loader
    
def model_step(model, optimizer, train_loader, test_loader, goat_acc, step, args):
    
    
    if True:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    else: 
        loss_fn = torch.nn.CrossEntropyLoss(weight=args['charweight_values'], reduction='none')
    total_loss = 0
    total_correct = 0
    total = 0
    model.train()
    for x, y, w in train_loader:
        
        w = w.float().to(args['device'])
        x = x.float().to(args['device'])
        y = y.long().to(args['device'])
        w.requires_grad = False
        preds = model(x)
        
        optimizer.zero_grad()
        loss = w*loss_fn(preds, y)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(preds, dim=-1)
        total_loss += loss.item()*x.shape[0]
        total_correct+= (predicted == y).sum().item()
        total += x.shape[0]
    print('Epoch %d tr loss' %step, '%.3f' %(total_loss/total), 'tr acc %.3f' %(total_correct/total))
    
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y, w in test_loader:
           
            x = x.float().to(args['device'])
            w = w.float().to(args['device'])
            y = y.long().to(args['device'])
            preds = model(x)
            loss = loss_fn(preds, y)
            loss = w*loss
            loss = loss.mean()
            _, predicted = torch.max(preds, dim=-1)
            total_loss += loss.item()*x.shape[0]
            total_correct+= (predicted == y).sum().item()
            total += x.shape[0]
    print('Epoch %d te loss' %step,'%.3f' %(total_loss/total), 'te acc %.3f' %(total_correct/total))
    return model, total_correct/total, total_loss/total

        
import copy


def get_mimed_loader(args, X_te, Y_te, blocks_te):
    train_jitter = Jitter((-2.0, 4.0), (args['winstart'], args['winend']), jitter_amt=args['jitter_amt'])
    test_jitter = Jitter((-2.0, 4.0), (args['winstart'], args['winend']), jitter_amt=0.0)
    blackout = Blackout(args['blackout_len'], args['blackout_prob'])
    noise = AdditiveNoise(args['additive_noise_level'])
    chan_noise = LevelChannelNoise(args['random_channel_noise_sigma'])
    scale = ScaleAugment(args['scale_augment_low'], args['scale_augment_high'])
    normalize = Normalize()
    
    
    test_augs = transforms.Compose([
        normalize, test_jitter
    ])
    
    test_weights = np.ones(X_te.shape[0])
    
    test_dset = WeightedBravoWindowDataset(X_te, Y_te, test_weights, test_augs)
    
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=64,
                                             shuffle=False)
    return test_loader

def eval_mimed_loader(mimed_loader, model, blocks, args):
    preds = []
    corrs = []
    resd = {}
    with torch.no_grad():
        for x, y, w in mimed_loader:
            x = x.float().to(args['device'])
            y= y.long().to(args['device'])

            pred = model(x)
            _, pred_labels = torch.max(pred, -1)
            preds.extend(pred_labels.cpu().numpy())
            corrs.extend(y.cpu().numpy())
    from bravo_ml.data_loading.load_bravo_utils import get_date
    import datetime
    b_to_date = {}
    for b in blocks:
        if not b in b_to_date: 
            b_to_date[b] = datetime.datetime.strptime(get_date(b, id=1), "%Y_%m_%d")

    dates = [b_to_date[b] for b in blocks]
    df = pd.DataFrame({
        'preds':preds,
        'corrs':corrs,
        'dates':dates
    })
    for d in set(df['dates'].values):
        subd = df.loc[df['dates']==d]
        print('d', d)
        print(np.mean(subd['preds']==subd['corrs']))
        
        
        


def training_function(config, X_alphabet, Y_alphabet, blocks_alphabet, 
                      X_te, Y_te, blocks_te, 
                      X_v, Y_v, blocks_v, 
                      lr=1e-3, run=True, pretr_fp=None, cv=0):
    args=config
        
    model, engl_acc = train_english_words(args)
    
    if not pretr_fp is None: 
        model.dense = torch.nn.Linear(int(args['decode_nodes']*2), 27)
        print('loading pretrained model from', pretr_fp)
        model.load_state_dict(torch.load(pretr_fp %cv))
        model.dense = torch.nn.Linear(int(args['decode_nodes']*2), len(set(Y_alphabet)))
    model = model.to(args['device'])
    goat_acc = 0
    patience = 0
    train_flag= True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # for _ in range(10): 
    #     print('training winend', args['winend'])
    train_loader, val_loader= get_alpha_loaders(args, X_alphabet, Y_alphabet, blocks_alphabet, X_v, Y_v, blocks_v)
    args['winend'] = 2.5
    print('Test window ends at 2.5s')
    _, test_loader= get_alpha_loaders(args, X_alphabet, Y_alphabet, blocks_alphabet,
                                                X_te, Y_te, blocks_te)
    if not run:
        return model, test_loader
    
    steps = 100
    if lr == 1e-4: 
        steps = 200
    for step in range(steps): #TODO change back.
        if train_flag:
            model, acc, loss = model_step(model, optimizer, train_loader, val_loader, goat_acc, step, args)
        goat_acc = max(acc, goat_acc)
        if not acc == goat_acc: 
            patience +=1
        else:
            patience = 0
            print('storing model params')
            best_model = copy.deepcopy(model)
        if patience == 5:
            train_flag=False
#         tune.report(mean_accuracy=goat_acc)

    return best_model, test_loader


def get_combo_args(args, n_class, chans):
    """
    This function loads in the arguments from a hyperparameter optimization that enables us to 
    combine the argtuments. 
    """

    orig_args=args
    print('using device', orig_args['device'])
    
    args = {'new_pretrain': False, 'from_pretrained': False, 
            'pretrained_dir': '/path/to/english_words_model', 
            'device': orig_args['device'], 'force_eng_retrain': False, 
            'savedir': 'path_to/tuning/results/', 
            'decode_layers': 2, 'decode_nodes': 274.4173302276509, 
            'dropout': 0.5451680587415845, 'ks': 4, 'winstart': 0.0, 'winend': 2.69, 
            'jitter_amt': 0.23718210382239427, 'additive_noise_level': 0.0027354917297051813, 
            'scale_augment_low': 0.9551356218945801, 'scale_augment_high': 1.0713824626558794, 
            'blackout_len': 0.30682868940865543, 'blackout_prob': 0.04787032280216536,
            'random_channel_noise_sigma': 0.028305685438945617
            }


    
    args['test_day'] = None
    args['decode_nodes'] = int(args['decode_nodes'])
    args['n_targ'] = n_class
    args['in_channels'] = chans
    combined_args= args
    for k, v in orig_args.items(): 
        if not k in args: 
            combined_args[k] = v
            
    return combined_args, args


def get_model_predictions(model, test_loader, args): 
    model.to(args['device'])
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():

        correct = 0
        preds = []
        labels = []
        te_total_samples = 0
        te_total_loss = 0

        for batch in test_loader:
            x = batch[0]
            y = batch[1]
            
            x = x.float().to(args['device'])
            y= y.long().to(args['device'])

            pred = model(x)
            loss = loss_fn(pred, y) 

            te_total_samples += x.shape[0]

            te_total_loss += x.shape[0]*loss.item()
            _, pred_labels = torch.max(pred, -1)
            correct += (pred_labels == y).sum().item()
            labels.extend(y.detach().cpu().numpy())
            preds.extend(F.softmax(pred, dim=-1).detach().cpu().numpy())

        te_loss = te_total_loss/te_total_samples
        te_acc = correct/te_total_samples
        print('acc', te_acc)
    return np.array(preds), np.array(labels)


def train_net(X_alphabet, Y_alphabet, blocks_alphabet,
              X_te, Y_te, blocks_te,
              X_v, Y_v, blocks_v,
              lr, args, cv, seed=0, 
             run =True, pretr_model=None):
    
    
    # CHECK
    combined_args, args = get_combo_args(args, max(27, len(set(Y_alphabet))), X_alphabet.shape[-1])
            
        
    best_model, test_loader = training_function(args, X_alphabet, Y_alphabet,
                                                blocks_alphabet,
                                                X_te, Y_te, blocks_te,
                                                X_v, Y_v, blocks_v,
                                                lr,
                                                run=run, 
                                               pretr_fp=pretr_model, 
                                               cv=cv)
    if not run: 
        return best_model, test_loader
    model = best_model

    model.eval()
    model.to('cpu')
    torch.save(model.state_dict(), args['model_saving_fp'] %cv)
    if not args['device'] == 'cpu':
        model.to(args['device']) 
    preds, labels  = get_model_predictions(best_model, test_loader, args)    
            
    return np.array(labels), np.array(preds), test_loader, best_model

    
