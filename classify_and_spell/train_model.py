import os
import argparse
import wandb
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_fp', default=None, type=str, 
                    help='where the data is stored. Should be in format X, Y, labels')
parser.add_argument('--end_date', default=None, type=str, 
                   help='last day of data to use, YYYY_MM_DD format')
parser.add_argument('--start_date', default=None, type=str, 
                   help='first day of data to use, YYYY_MM_DD format')
parser.add_argument('--pred_storage_fp', default=None, type=str, 
                   help='where to store the predictions and labels')

parser.add_argument('--data_split_scheme', default='random', type=str,
                   help='how to split up the data')

parser.add_argument('--save_saliencies', action='store_true', help='save and calculate saliences')

parser.add_argument('--model_saving_fp', default=None, help='where to store the models + the base name')

parser.add_argument('--ensemble_amt', default=1, type=int, help='how many models to use in ens.')

parser.add_argument('--load_pretrained_model', default=None, type=str, help='path of pretr. models')

parser.add_argument('--feats_to_use', default='hga_and_raw', type=str, help='which band')

parser.add_argument('--test_dataset_fp', default=None, type=str, help='test fp, if its different.')

parser.add_argument('--test_both', action='store_true', help='test both the original test set and the new one.')

parser.add_argument('--test_start_date', default=None, type=str, 
                   help='when to start the test set dates')

parser.add_argument('--test_end_date', default=None, type=str,
                   help='when to end the test set dates')

parser.add_argument('--num_cvs', default=10, type=int, help='how many CVs to run')

parser.add_argument('--experiment_name', default='experiment', type=str, help='what to name the runs/models/preds')

parser.add_argument('--num_samples', default=None, type=int, help='max no. of samples to use')

parser.add_argument('--dont_store_saliences', action='store_true')

# parser.add_argument('--dont_train', action='store_true', help='just use the old models and test on the data')

args= vars(parser.parse_args())


from data_wrangling import load_data , split_data
from machine_learning import run_standard_model, prediction_only
from bravo_ml.model_eval.torch_salience import get_saliences
from saving_functions import save_saliences, make_dataframe

args['test_end_date'] = args['end_date']

# Step 1: Load in the dataset.
X, Y, blocks, dates = load_data(args['dataset_fp'], args['start_date'], args['end_date'])

if args['feats_to_use'] == 'raw':
    X = X[:, :, :128]
elif args['feats_to_use'] == 'hga': 
    X = X[:, :, 128:]

### If there's a different test dataset, specify it and then make the new test tuple of datasets.
if args['test_dataset_fp'] != None: 
    Xt, Yt, blockst, _ = load_data(args['test_dataset_fp'], args['test_start_date'], args['test_end_date'])
    test_tuple = (Xt, Yt, blockst)
else: 
    test_tuple = None

# Initialize lists to save all the meaningful metrics we want
pred_list = []
label_list = []
cvs_list = []
blocks_list = []
prev_sal_start = 0
if args['test_both']:
    preds2, labels2, cvs2, blocks2 = [], [], [], []    
all_saliences = np.zeros_like(X)


# Go through the cvs necessary. 
for cv in range(args['num_cvs']): 
    # Split the data as u wish.
    X_tr, Y_tr, blocks_tr, X_te, Y_te, blocks_te = split_data((X,Y, blocks), None,
                                                              args['data_split_scheme'], 
                                                              args['num_cvs'], cv, 
                                                              args['num_samples'])

    X_tr, Y_tr, blocks_tr, X_v, Y_v, blocks_v = split_data((X_tr, Y_tr, blocks_tr),
                                                          None,  
                                                           args['data_split_scheme'], 
                                                            args['num_cvs'], cv,
                                                          None)
    
    print('shapes', X_tr.shape, X_v.shape, X_te.shape)
        
    # Load the pretrained model if necessary
    if not args['load_pretrained_model'] is None: 
        pretr_model = args['load_pretrained_model']
       
    else: 
        pretr_model = None
    
    
    if pretr_model is None: 
        lr = 1e-3
    else: 
        lr = 1e-4
    # Run the training loop.
    labels, preds, te_loader, model = run_standard_model(X_tr, Y_tr, blocks_tr, 
                                                         X_te, Y_te, blocks_te,
                                                         X_v, Y_v, blocks_v, pretr_model, 
                                                         lr, cv,
                                                         args)
    
    pred_list.extend(preds)
    label_list.extend(labels)
    cvs_list.extend([cv]*len(labels))
    blocks_list.extend(blocks_te)
    

      ### Get saliences and put them into the array
    saliences = get_saliences(model, te_loader)
    all_saliences[prev_sal_start:prev_sal_start+X_te.shape[0], :saliences.shape[1]] = saliences
    prev_sal_start += X_te.shape[0]

    ### Test on the second test set if we want.
    if args['test_both']:
        # get the new test stuff. 
        _, _,_,X_te, Y_te, blocks_te = split_data((X,Y, blocks), test_tuple,
                                                              args['data_split_scheme'], args['num_cvs'], cv, 
                                                              args['num_samples'])
        preds, labels= prediction_only(model, X_te, Y_te, blocks_te, cv, args)
        preds2.extend(preds)
        labels2.extend(labels)
        cvs2.extend([cv]*len(preds))
        blocks2.extend(blocks_te)
        ### END OF FUNCTION

# Make and save the dataframe from the trial
df = make_dataframe(pred_list, label_list, cvs_list, blocks_list, args)

if args['test_both']:
    df2 = make_dataframe(preds2, labels2, cvs2, blocks2, {
        'experiment_name':args['experiment_name'] + '_second_test_set'
    })
# save the saliences
if not (args['dont_store_saliences']):
    save_saliences(all_saliences[:, :saliences.shape[1], :], args)