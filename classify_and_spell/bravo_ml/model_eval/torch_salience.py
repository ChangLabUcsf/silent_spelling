import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

def get_saliences(model, test_loader, device='cuda'):
    """
    this is a function that takes in a pytorch model and a data loader and returns
    the saliences.
    
    For the sake of deeper analyses the loader has two options: 
      
    """
    
    res_dict = defaultdict(list)
    
    
    
    loss_fn = nn.CrossEntropyLoss()
#     model.eval()
    model.train()
    model.dropout.eval()
    
    all_x = []
    for batch in test_loader:
        x = batch[0]
        y = batch[1]
        x = x.float().to(device)
        x.requires_grad_()
        y= y.long().to(device)
        pred = model(x)
        loss = loss_fn(pred, y) 
        loss.backward()
        # print(x.grad.data.shape)
        
        all_x.extend(x.grad.data.cpu().numpy())
            
    return np.array(all_x)
        
        
     