"""
training routines for pytorch models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

def train_classifier(model, train_loader, 
                        test_loader,
                        device,
                        optimizer=None,
                        epochs=100,
                        early_stop = True,
                         patience=4,
                        lr=1e-3,
                         min_epochs=11,
                        checkpoint = False,
                        checkpoint_fp = '/path_to/results/tmp.pth',
                        wandb=False,
                    es_metric='acc',
                    file_checkpoint=True,
                    weight=None,
                    clip_grad=True):
    """
    pytorch training script. 
    
    
    inputs: 
    
    model: the model. Shoudl take in the input and return 
    Add LR schedule

    """
    
    history = {}
    history['acc'] = []
    history['val_acc'] = []
    history['val_loss'] = []
    history['loss'] = []
    
    
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight).to(device)
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=lr)
    patience_ct = 0
    max_te_acc = 0
    min_te_loss = 1000
    for epoch in range(epochs):
        
        total_loss = 0
        total_samples = 0
        correct= 0
        # training routine. 
        model.train()
        for x, y in train_loader: 
            x = x.float().to(device)
            y= y.long().to(device).squeeze()
            pred = model(x)
            loss = loss_fn(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            total_samples += x.shape[0]
            
            total_loss += x.shape[0]*loss.item()
            
            _, pred_labels = torch.max(pred, -1)
            correct += (pred_labels == y).sum().item()
            
        tr_loss = total_loss/total_samples
        tr_acc = correct/total_samples
        
        
        model.eval()
        te_total_loss, te_total_samples = 0, 0
        correct = 0
        for x, y in test_loader:
            x = x.float().to(device)
            y= y.long().to(device)
            
            pred = model(x)
            loss = loss_fn(pred, y) 
            
            te_total_samples += x.shape[0]
            
            te_total_loss += x.shape[0]*loss.item()
            _, pred_labels = torch.max(pred, -1)
            correct += (pred_labels == y).sum().item()
            
        te_loss = te_total_loss/te_total_samples
        te_acc = correct/te_total_samples
        
        
        # Logging
        if wandb:
            wandb.log({
                "tr_loss":tr_loss,
                "te_loss":te_loss,
                "tr_acc": tr_acc,
                "te_acc": te_acc
            })
            
        history['acc'].append(tr_acc)
        history['val_acc'].append(te_acc)
        history['loss'].append(tr_loss)
        history['val_loss'].append(te_loss)
        
        print('epoch %d, loss: %.3f, te_loss: %.3f, acc: %.3f, te acc: %.3f' %(epoch, 
                                                                              tr_loss,
                                                                              te_loss, 
                                                                              tr_acc,
                                                                              te_acc))
        
        # End of logging.
        
        """
        Early stopping.
        """
        if early_stop: 
            if es_metric == 'acc':
                if te_acc > max_te_acc:
                    max_te_acc = te_acc
                    patience_ct = 0
                    if checkpoint and file_checkpoint:
                        torch.save(model.state_dict(), checkpoint_fp)
                    elif not file_checkpoint: 
                        best_model = copy.deepcopy(model)


                else: 
                    if epochs < min_epochs-patience: 
                        patience_ct =0
                    patience_ct += 1
                    if patience_ct == patience:
                        break
            elif es_metric == 'loss':
                if te_loss < min_te_loss:
                    min_te_loss = te_loss
                    patience_ct = 0
                    if checkpoint and file_checkpoint: 
                        torch.save(model.state_dict(), checkpoint_fp)
                    elif not file_checkpoint:
                        best_model = copy.deepcopy(model)

                else: 
                    if epochs < min_epochs-patience: 
                        patience_ct =0
                    patience_ct += 1
                    if patience_ct == patience:
                        break
                
    if checkpoint and file_checkpoint:
        print('restoring from %s' %(checkpoint_fp))
        model.load_state_dict(torch.load(checkpoint_fp))
        model.eval()
    elif not file_checkpoint: 
        print('using the best model, which was copied')
        model = best_model
        model.eval()
    return model, history



import numpy as np
            
def model_predict(model, test_loader, val_loader=None, device='cpu', softmax=False):
    """
    This just returns labels and predictions
    
    Input: model, test loader, device
    Output: predictions, and labels, as numpy arrays.
    """
    
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    te_total_loss, te_total_samples = 0, 0
    correct = 0
    preds = []
    labels = []
    
    model.eval()
    te_total_loss, te_total_samples = 0, 0
    correct = 0
    if not val_loader is None: 
        for x, y in val_loader:
            x = x.float().to(device)
            y= y.long().to(device)

            pred = model(x)
            loss = loss_fn(pred, y) 

            te_total_samples += x.shape[0]

            te_total_loss += x.shape[0]*loss.item()
            _, pred_labels = torch.max(pred, -1)
            correct += (pred_labels == y).sum().item()

        te_loss = te_total_loss/te_total_samples
        te_acc = correct/te_total_samples
        print('val acc', te_acc)
    
    """ THE OTHER ONE"""
    correct = 0
    preds = []
    labels = []
    te_total_samples = 0
    for x, y in test_loader:
        x = x.float().to(device)
        y= y.long().to(device)
        
        

        pred = model(x)
        loss = loss_fn(pred, y) 

        te_total_samples += x.shape[0]

        te_total_loss += x.shape[0]*loss.item()
        _, pred_labels = torch.max(pred, -1)
        correct += (pred_labels == y).sum().item()
        labels.extend(y.detach().cpu().numpy())
        if softmax:
            pred = torch.softmax(pred, dim=-1)
        preds.extend(pred.detach().cpu().numpy())

    te_loss = te_total_loss/te_total_samples
    te_acc = correct/te_total_samples
    
    print('te acc', te_acc)
    return np.array(labels), np.array(preds), te_acc



def train_flex_classifier(model, train_loader, 
                        test_loader,
                        device,
                        optimizer=None,
                        epochs=100,
                        early_stop = True,
                         patience=4,
                        lr=1e-3,
                         min_epochs=11,
                        checkpoint = False,
                        checkpoint_fp = '/path/to/results/tmp.pth',
                        wandb=False,
                         es_metric='acc'):
    """
    pytorch training script. 
    
    
    inputs: 
    
    model: the model. Should take in the input and return 
    Add LR schedule

    """
    
    history = {}
    history['acc'] = []
    history['val_acc'] = []
    history['val_loss'] = []
    history['loss'] = []
    
    
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=lr)
    patience_ct = 0
    max_te_acc = 0
    for epoch in range(epochs):
        
        total_loss = 0
        total_samples = 0
        correct= 0
        # training routine. 
        model.train()
        for x, l, y in train_loader: 
            x = x.float().to(device)
            y= y.long().to(device).squeeze()
            l = l.long().cpu()
            pred = model(x, l)
            
            loss = loss_fn(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_samples += x.shape[0]
            
            total_loss += x.shape[0]*loss.item()
            
            _, pred_labels = torch.max(pred, -1)
            correct += (pred_labels == y).sum().item()
            
        tr_loss = total_loss/total_samples
        tr_acc = correct/total_samples
        
        
        model.eval()
        te_total_loss, te_total_samples = 0, 0
        correct = 0
        for x, l, y in test_loader:
            x = x.float().to(device)
            y= y.long().to(device).squeeze()
            l = l.long().cpu()
            pred = model(x, l)
            loss = loss_fn(pred, y) 
            
            te_total_samples += x.shape[0]
            
            te_total_loss += x.shape[0]*loss.item()
            _, pred_labels = torch.max(pred, -1)
            correct += (pred_labels == y).sum().item()
            
        te_loss = te_total_loss/te_total_samples
        te_acc = correct/te_total_samples
        
        
        # Logging
        if wandb:
            wandb.log({
                "tr_loss":tr_loss,
                "te_loss":te_loss,
                "tr_acc": tr_acc,
                "te_acc": te_acc
            })
            
        history['acc'].append(tr_acc)
        history['val_acc'].append(te_acc)
        history['loss'].append(tr_loss)
        history['val_loss'].append(te_loss)
        
        print('epoch %d, loss: %.3f, te_loss: %.3f, acc: %.3f, te acc: %.3f' %(epoch, 
                                                                              tr_loss,
                                                                              te_loss, 
                                                                              tr_acc,
                                                                              te_acc))
        
        # End of logging.
        
        """
        Early stopping.
        """
        if early_stop: 
            if 'acc' in es_metric:
                if te_acc > max_te_acc:
                    max_te_acc = te_acc
                    patience_ct = 0
                    if checkpoint:
                        torch.save(model.state_dict(), checkpoint_fp)


                else: 
                    if epochs < min_epochs-patience: 
                        patience_ct =0
                    patience_ct += 1
                    if patience_ct == patience:
                        break
            elif 'loss' in es_metric: 
                if te_loss < min_te_loss:
                    min_te_loss = te_loss
                    pateince_ct = 0
                    if checkpoint: 
                        torch.save(model.state_dict(), checkpoint_fp)
                else: 
                    if epochs < min_epochs-patience: 
                        patience_ct=0
                    patience_ct+=1
                    if patience_ct == patience: 
                        break
                
    if checkpoint:
        print('restoring from %s' %(checkpoint_fp))
        model.load_state_dict(torch.load(checkpoint_fp))
        model.eval()
    return model, history


def model_flex_predict(model, test_loader, val_loader=None, device='cpu'):
    """
    This just returns labels and predictions
    
    Input: model, test loader, device
    Output: predictions, and labels, as numpy arrays.
    """
    with torch.no_grad():
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        te_total_loss, te_total_samples = 0, 0
        correct = 0
        preds = []
        labels = []

        model.eval()
        te_total_loss, te_total_samples = 0, 0
        correct = 0
        if not val_loader is None: 
            for x, l, y in val_loader:
                x = x.float().to(device)
                y= y.long().to(device).squeeze()
                l = l.long().cpu()
                print('cpu')

                pred = model(x, l)
                loss = loss_fn(pred, y) 

                te_total_samples += x.shape[0]

                te_total_loss += x.shape[0]*loss.item()
                _, pred_labels = torch.max(pred, -1)
                correct += (pred_labels == y).sum().item()

            te_loss = te_total_loss/te_total_samples
            te_acc = correct/te_total_samples
            print('val acc', te_acc)

        """ THE OTHER ONE"""
        correct = 0
        preds = []
        labels = []
        te_total_samples = 0
        for x, l, y in test_loader:
            x = x.float().to(device)
            y= y.long().to(device).squeeze()
            l = l.long().cpu()



            pred = model(x, l)
            loss = loss_fn(pred, y) 

            te_total_samples += x.shape[0]

            te_total_loss += x.shape[0]*loss.item()
            _, pred_labels = torch.max(pred, -1)
            correct += (pred_labels == y).sum().item()
            labels.extend(y.detach().cpu().numpy())
            preds.extend(pred.detach().cpu().numpy())

        te_loss = te_total_loss/te_total_samples
        te_acc = correct/te_total_samples

        print('te acc', te_acc)
        return np.array(labels), np.array(preds), te_acc
