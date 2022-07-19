
from bravo_ml.stability.standard_model_for_paper import training_function
import numpy as np
def run_standard_model(X_tr, Y_tr, blocks_tr, 
                       X_te, Y_te, blocks_te,
                       X_v, Y_v, blocks_v, 
                       pretr_model, lr,
                      cv, args):   
    # Step 1. Get the loaders NOTE: you should make a validation set :(
    
    from bravo_ml.stability.standard_model_for_paper import train_net
    
    labels, preds, test_loader, model = train_net(X_tr, Y_tr, blocks_tr, 
                                                 X_te, Y_te, blocks_te, 
                                                  X_v, Y_v, blocks_v,
                                                  lr, args, cv, pretr_model=pretr_model)
    
    return labels, preds, test_loader, model


def prediction_only(model, X_te, Y_te, blocks_te, cv, args):
    from bravo_ml.stability.standard_model_for_paper import train_net
    from bravo_ml.stability.standard_model_for_paper import get_model_predictions
    _, test_loader =  train_net(X_te, Y_te, blocks_te,
                                       X_te, Y_te, blocks_te,
                                       X_te, Y_te, blocks_te,
                                     0, args,cv,
                                       run=False)
    
    
    preds, labels= get_model_predictions(model, test_loader, {'device':'cuda'})
    return preds, labels


def get_model_shell(X_te, Y_te, blocks_te, cv, args): 
    from bravo_ml.stability.standard_model_for_paper import train_net
    from bravo_ml.stability.standard_model_for_paper import get_model_predictions
    model_shell, test_loader =  train_net(X_te, Y_te, blocks_te,
                                       X_te, Y_te, blocks_te,
                                       X_te, Y_te, blocks_te,
                                     0, args,cv,
                                       run=False)

    return model_shell 

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