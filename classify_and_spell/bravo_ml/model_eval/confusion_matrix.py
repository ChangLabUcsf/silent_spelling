import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix
import scipy
import matplotlib


def even_lists(preds_list, test_list):
    
    for k, (p, t) in enumerate(zip(preds_list, test_list)):
        if len(p) != len(t):
            print(len(p), len(t))
            test_list[k] = t[:len(p[k])]
        
    print(np.array(test_list).shape)
    return preds_list, test_list

def plot_confusions(preds_list, test_list, text_decoder, fontsize=18): 
    
    # Step 1.
    # Get the confusions
    
    # Step 2. Plot it using the confusion matrix code.
#     preds_list, test_list = even_lists(preds_list, test_list) # for when batch size is off. 
    
    # Step 3. Make this plot again, but with clustering (TODO)
    plt.rcParams.update({'font.size': fontsize})
    
    n_class = len(text_decoder.keys())

    preds_list = np.array(preds_list)
    preds_list.shape

    test_list = np.array(test_list)

    test_list.shape

    text_decoder

    targets = np.reshape(test_list, (-1, 1))
    all_preds = np.reshape(preds_list, (-1, n_class))



    targets_ = targets

    preds_ = np.argmax(all_preds, axis=-1)

    cm = confusion_matrix(targets_, preds_)
    cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]

    for i in range(cm.shape[0]): 
        for j in range(cm.shape[1]): 
            cm[i, j] = np.round(cm[i, j], 2)
            
    z = scipy.cluster.hierarchy.ward(1-cm)
    labels = [text_decoder[v] for v in range(len(text_decoder.keys()))]
    pdf = pd.DataFrame(data=cm, index=labels, columns=labels)
    sns.clustermap(pdf, cmap= 'Greys', row_linkage=z, col_linkage=z, 
                   figsize=(13, 13), linewidths=.5, annot = False)
    
    plt.show()