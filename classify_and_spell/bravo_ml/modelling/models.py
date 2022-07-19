# import keras

# from keras.layers import *
# from keras import Model


def mod_input_shape(input_shape, default_window, original_window, decimation, sr=200):
    """
    This function is used to get the real input shapes AFTER augmentations have occured. 
    It is necessary to call this function PRIOR to having the neural network initialized
    so it knows what shapes to expect...
    
    Inputs: 
    input_shape: original input shape
    default_window: window we are going to use
    original_window: window used during load
    decimation: how much decimation was used
    sr: sampling rate, in Hz.
    
    returns modified input_shape, to reflect the augmentation window sizes.
    
    
    """
    default_samples = np.asarray(default_window) - original_window[0]

    # original window is say -2, 4
    # will shift it, so now its 1s in, 5s relative to 0 where 0 is the onset

    default_samples = np.asarray(default_samples)*sr/decimation

    default_samples[0] = int(default_samples[0])
    default_samples[1] = int(default_samples[1])
    
    win_size = (default_samples[1]-default_samples[0])+1
    input_shape = list(input_shape)
    input_shape[0] = int(win_size)
    return tuple(input_shape)

def cnn_rnn_model_maker(n_class, input_shape, hyperparameters):
    
    """
    This funciton will give us the cnn-rnn model that I used for the 50-word set for our publication.
    
    input: 
    n_class - number of classes
    input_shape - the input shape (excluding batch size)
    hyperaparameters - the hyperarameters which are a dict with...
        nodes, num_layers, dropout, l2, ks, strides
        Name says it all for these. 
        
        
    Returns: A model ready to be trained. 
    
    """
    config=hyperparameters
    
    nodes = config['nodes']
    num_layers = config['num_layers']
    dropout = config['dropout']
    l2 = config['l2']
    ks= config['ks']
    strides=config['ks']
    try:
        seed = config['seed']
    except Exception: 
        seed = None
    
    if seed==None: 
        seed = np.random.randint(0, 1000)
        
    np.random.seed(seed) # reproducibility
    
    
    x = Input(shape = input_shape) 
    f = Conv1D(filters=nodes, kernel_size=ks, strides=strides, padding='valid', kernel_regularizer=regularizers.l2(l2))(x)
    f = Dropout(dropout)(f)
    for k in range(num_layers): 
        return_seq = bool(not(k==num_layers-1))
        f = Bidirectional(CuDNNGRU(nodes, kernel_regularizer=regularizers.l2(l2), return_sequences=return_seq))(f)
        f = Dropout(dropout)(f)
    out = Dense(n_class, activation='softmax', kernel_regularizer=regularizers.l2(l2))(f)
    model = Model(x, out)
    return model



def deep_cnn_rnn_model_maker(n_class, input_shape, hyperparameters):     
    """
    This model enables you to use large convolutions to hugely downsample the input
    before running the LSTM on it...
    
    Hopefully, this helps us learn better representations. 
    
    Input: 
        n_class: the number of classes you are trying to decode. 
        input_shape: The exepected input shape. 
        
        Hyperparameters: a dict with hyperparameters.
        
        l2, dropout, num_layers, as per usual. 
        
        NEW for this model: 
            conv_layer_specs: 
                A dictionary which has 
    
    """
    nodes = hyperparameters.get('nodes', 128)
    l2 = hyperparameters.get('l2', 0.0)
    dropout = hyperparameters.get('dropout', 0.0)
    num_layers = hyperparameters.get('num_layers', 1)
    conv_layer_specs = hyperparameters['conv_layer_specs']
    
    
    x = Input(shape = input_shape) 
    f = x
    for k, layer in enumerate(conv_layer_specs.keys()): 
        f = Conv1D(filters=nodes, 
                   kernel_size = conv_layer_specs[layer]['ks'], 
                   strides = conv_layer_specs[layer]['stride'],
                   padding = 'valid',
                   kernel_regularizer=regularizers.l2(l2))(f)
        f = Dropout(dropout)(f)
    for k in range(num_layers): 
        return_seq = bool(not(k==num_layers-1))
        f = Bidirectional(CuDNNGRU(nodes, kernel_regularizer=regularizers.l2(l2), return_sequences=return_seq))(f)
        f = Dropout(dropout)(f)
    out = Dense(n_class, activation='softmax', kernel_regularizer=regularizers.l2(l2))(f)
    model = Model(x, out)
    return model
        


def mlp_maker(n_class, input_shape, hyperparameters): 
    
    """
    This function returns an MLP given your hyperparameters, n_classes, and inputshape. 
    
    n_class=number of classes
    input shape = input shape, for keras .
    
    hyperparameters = 
        nodes: number of nodes
        num_layers = number of layers. 
        dropout - droput amount
        l2 - 2 norm penalty. 
        
    returns: model - the fully assembled model.
    
    """
    
    config = hyperparameters
    nodes = config['nodes']
    num_layers = config['num_layers']
    dropout = config['dropout']
    l2 = config['l2']
    
    try: 
        seed = config['seed']
    except Exception: 
        seed = None
    if seed == None:
        seed = np.random.randint(0, 1000)
    np.random.seed(seed)
    
    x = Input(shape= input_shape)
    f= Flatten()(x)
    for k in range(num_layers):
        f = Dense(nodes, activation='relu', kernel_regularizer=regularizers.l2(l2))(f)
        f = Dropout(dropout)(f)
    out = Dense(n_class, kernel_regularizer=regularizers.l2(l2))(f)
    model = Model(x, out)
    return model
    

def get_bravo_models(model_name, n_class, input_shape, hyperparameters):
    """
    This is the overall model call.
    
    You can call the following models: 
    
    CNN-RNN - the typical model being used. (see cnn_rnn_model_maker for the hyperparameters required.)
    
    RNN - Just an RNN based thing TODO
    
    CNN - Just a CNN TODO
    
    MLP - Just an MLP  TODO
    
    Transformer - TODO
    
    """
    if model_name == 'cnn-rnn': 
        return cnn_rnn_model_maker(n_class, input_shape, hyperparameters)
    
    if model_name == 'mlp': 
        return mlp_maker(n_class, input_shape, hyperparameters)
    
    
    if model_name == 'deep_cnn-rnn':
        return deep_cnn_rnn_model_maker(n_class, input_shape, hyperparameters)
    
    
    else:
        print('Model not available')
        assert False