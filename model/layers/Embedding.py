import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#pasted away
class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day(288, one day -> per 5 minutes)
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D):
        super(STEmbedding, self).__init__()
        self.mlp_spatial1 = nn.Linear(D,D)
        self.mlp_spatial2 = nn.Linear(D,D)
        # input_dims = 7 + T(=288)
        self.mlp_temporal1 = nn.Linear(295,D)
        self.mlp_temporal2 = nn.Linear(D,D)
        self.activation = nn.ReLU()
    
    
    def forward(self, SE, TE, T=288):
        # Spatial Embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.mlp_spatial1(SE)
        SE = self.activation(SE)
        SE = self.mlp_spatial2(SE)
        # SE.shape[1,1,V,D]

        # Temporal Embedding
        batch_size, num_his_pred, _ = TE.shape
        # one-hot encoding
        dayofweek = F.one_hot(TE[..., 0].to(torch.int64) % 7, num_classes=7).float()
        timeofday = F.one_hot(TE[..., 1].to(torch.int64) % T, num_classes=T).float()
        # Concatenate
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        
        # [B,num_his_pred,7+T] -> [B,num_his_pred,1,7+T]
        TE = TE.view(batch_size, num_his_pred, 1, -1)
        TE = self.mlp_temporal1(TE)
        TE = self.activation(TE)
        TE = self.mlp_temporal2(TE)

        return SE + TE
    
#new
class XTEembedding(nn.Module):
    def __init__(self, d_model) -> None:
        super(XTEembedding, self).__init__()
        self.d_model = d_model
        self.T = 288
        self.linear_te = nn.Linear(7+self.T, d_model)
        self.linear_out1 = nn.Linear(d_model + d_model, d_model)
        self.linear_out2 = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()
    
    
    def forward(self, X, TE):
        # [B,num_his_pred,V,D]
        X = X.repeat(1, 2, 1, 1)
        # Temporal Embedding
        num_vertex = X.shape[2]
        # one-hot encoding
        dayofweek = F.one_hot(TE[..., 0].to(torch.int64) % 7, num_classes=7).float()
        timeofday = F.one_hot(TE[..., 1].to(torch.int64) % self.T, num_classes=self.T).float()
        # Concatenate
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        
        # [B,num_his_pred,7+T] -> [B,num_his_pred,V,7+T]
        TE = TE.unsqueeze(2).repeat(1, 1, num_vertex, 1)
        TE = self.linear_te(TE)

        res = torch.cat((X, TE), dim=-1)
        res = self.linear_out1(res)
        res = self.activation(res)
        res = self.linear_out2(res)
        return res

#new
class XSEembedding(nn.Module):
    def __init__(self, d_model) -> None:
        super(XSEembedding, self).__init__()
        self.d_model = d_model
        self.T = 288
        self.linear_se = nn.Linear(d_model, d_model)
        self.linear_out1 = nn.Linear(d_model + d_model, d_model)
        self.linear_out2 = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()
    
    
    def forward(self, X, SE):
        B = X.shape[0]
        L = X.shape[1]
        # Spatial Embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        # SE.shape[1,1,V,D] -> [B,L,V,D]
        SE = SE.repeat(B, L, 1, 1)
        SE = self.linear_se(SE)
        res = torch.cat((X, SE), dim=-1)
        res = self.linear_out1(res)
        res = self.activation(res)
        res = self.linear_out2(res)
        return res
