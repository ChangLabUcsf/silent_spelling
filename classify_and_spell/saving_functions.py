import pandas as pd
import numpy as np

def make_dataframe(preds, labs, cvs, blx, args):
  
    df = pd.DataFrame({
        'pred_vec':preds,
        'label':labs,
        'cv':cvs,
        'blocks':blx,
    })
    
    df.to_pickle(f"./result_dfs/{args['experiment_name']}.pkl")
    return df

def save_saliences(salz, args):
    np.save('./saliences/' + args['experiment_name'] + '.npy',
           salz)