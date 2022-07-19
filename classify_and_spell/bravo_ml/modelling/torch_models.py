import torch
import torch.nn as nn
import torch.nn.functional as F
    
class CnnRnnClassifier(torch.nn.Module): 
    """
    The CNN RNN classifier that I used for the bravo1 decoding, in pytorch. 
    """
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels=128):
        super().__init__()
        
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=KS)
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =num_layers,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim*mult, n_targ)
        
    def forward(self, x): 
        # x comes in bs, t, c
        x = x.contiguous().permute(0, 2, 1)
        # now bs, c, t
        x = self.preprocessing_conv(x)
#         x = F.relu(x)
        x = self.dropout(x)
        x = x.contiguous().permute(2, 0, 1)
        # now t, bs, c
        _ , x = self.BiGRU(x)
        x = x.contiguous().view(self.num_layers, self.mult, -1, self.rnn_dim)
        x = x[-1] # Only care about the output at the final layer.
        # (2, bs, rnn_dim)
        
        x= x.contiguous().permute(1, 0, 2)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dropout(x)
        out = self.dense(x)
        return out
    
    
class Continuous_CnnRnnClassifier(torch.nn.Module): 
    """
    Modified model which allows us to take in the prev. b1 results.
    """
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels=128):
        super().__init__()
        
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=KS)
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =num_layers,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim*mult, n_targ)
        
    def forward(self, x): 
        # x comes in bs, t, c
        x = x.contiguous().permute(0, 2, 1)
        # now bs, c, t
        x = self.preprocessing_conv(x)
#         x = F.relu(x)
        x = self.dropout(x)
        x = x.contiguous().permute(2, 0, 1)
        # now t, bs, c
        output , x = self.BiGRU(x)
#         x = x.contiguous().view(self.num_layers, self.mult, -1, self.rnn_dim)
#         x = x[-1] # Only care about the output at the final layer.
#         # (2, bs, rnn_dim)
        
#         x= x.contiguous().permute(1, 0, 2)
#         x = x.contiguous().view(x.shape[0], -1)
        # t, bs,  2*rnn_dim
    
        output = output.contiguous().permute(1, 0, 2) # now will be (bs, time, chans)
        
        out = self.dropout(output)
        out = self.dense(out)
        # bs, t, c
        out = out.contiguous().permute(0, 2, 1) # Now we have (bs, c, t, good for the output.)
        
        return out

    
class NormGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, 
                bidirectional, 
                dropout, layernorm, first):
        super().__init__()
        
        rnn_dim = input_size
        self.first = first
        self.inmult = 1
        if (not self.first) and bidirectional:
            self.inmult = 2
        self.GRU = nn.GRU(input_size=rnn_dim*self.inmult, hidden_size=rnn_dim, 
                           num_layers =1,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        if bidirectional: 
            self.mult =2
        else: 
            self.mult = 1
            
        if layernorm: 
            self.norm = nn.LayerNorm(rnn_dim*self.mult)
        else: 
            self.norm = nn.BatchNorm1d(rnn_dim*self.mult)
        self.layernorm = layernorm
            
    def forward(self, x):
        # T, BS, C
        output, h_n = self.GRU(x)
        #output = T, BS ,ND*HS
#         output = output.contiguous().permute()
        
        # FOR BATCHNORM, we want N, C,L -> batchnorm
        if not self.layernorm: 
            output = output.contiguous().permute(1, 2, 0)
            output = self.norm(output) #NCL
            output = output.contiguous().permute(2, 0, 1) #L, N, C
            
        else: 
            output = output.contiguous().permute(1, 0, 2)
            output = self.norm(output) # NLC
            output = output.contiguous().permute(1, 0, 2) #L, N, C
        return output, h_n
        
        
        
    
class Norm_CnnRnnClassifier(torch.nn.Module): 
    """
    The CNN RNN classifier that I used for the bravo1 decoding, in pytorch. 
    """
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels=128, layernorm=False,
                no_norm=False):
        super().__init__()
        
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=KS)
        if layernorm:
            self.bn1 = nn.LayerNorm(rnn_dim)
        else:
            self.bn1 = nn.BatchNorm1d(rnn_dim)
        
        self.BiGRU1 = NormGRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =1,
                            bidirectional=bidirectional, 
                            dropout=dropout,
                            layernorm=layernorm,
                                           first=True)
        
        self.BiGRU2 = NormGRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                   num_layers =1,
                    bidirectional=bidirectional, 
                    dropout=dropout,
                    layernorm=layernorm,
                                   first=False)
            
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim*mult, n_targ)
        if layernorm: 
            self.bn2 = nn.LayerNorm(rnn_dim*mult)
        else: 
            self.bn2 = torch.nn.BatchNorm1d(rnn_dim*mult)
        self.layernorm = layernorm
        
        
    def forward(self, x): 
        # x comes in bs, t, c
        x = x.contiguous().permute(0, 2, 1)
        # now bs, c, t
        x = self.preprocessing_conv(x)
#         x = F.relu(x)
        if not self.layernorm:
            x = self.bn1(x)
        
        x = x.contiguous().permute(2, 0, 1)
        # now t, bs, c
#         if self.layernorm: 
#             x = self.bn1(x) # Layernorm digs this.
        
        x = self.dropout(x)
   
        out1, x = self.BiGRU1(x)
        out1 = self.dropout(out1)
        out2, x = self.BiGRU2(out1)
        
        x = x.contiguous().view(1, self.mult, -1, self.rnn_dim)
        
        x = x[-1] # Only care about the output at the final layer.
        # (2, bs, rnn_dim)
        
        x= x.contiguous().permute(1, 0, 2)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.bn2(x)
        x = self.dropout(x)
        out = self.dense(x)
        
        return out
    
#######
#
#######
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class FlexibleCnnRnnClassifier(torch.nn.Module): 
    """
    The CNN RNN classifier that I used for the bravo1 decoding, in pytorch. 
    """
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels=128):
        super().__init__()
        
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=KS)
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =num_layers,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.ks = KS
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim*mult, n_targ)
        
    def forward(self, x, lens): 
        # x comes in bs, t, c
        lens = lens//self.ks
        x = x.contiguous().permute(0, 2, 1)
        # now bs, c, t
        x = self.preprocessing_conv(x)
#         x = F.relu(x)
        x = self.dropout(x)
        x = x.contiguous().permute(2, 0, 1)
        # now t, bs, c
#         print(lens.max())
#         print(lens.max(), x.shape)
        packed = pack_padded_sequence(x, lens.long(), enforce_sorted=False)
        emissions, hiddens = self.BiGRU(packed)
        unpacked_emissions, _ = pad_packed_sequence(emissions)
        x = hiddens[-self.mult:]
        x= x.contiguous().permute(1, 0, 2)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dropout(x)
        out = self.dense(x)
        return out
