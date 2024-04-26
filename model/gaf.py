import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from model.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import ipdb

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


class SpatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    num_heads:        number of attention heads
    dim_heads:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, num_heads, dim_heads):
        super(SpatialAttention, self).__init__()
        D = num_heads * dim_heads
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.mlp_Q = nn.Linear(2*D, D)
        self.mlp_K = nn.Linear(2*D, D)
        self.mlp_V = nn.Linear(2*D, D)
        self.mlp_output = nn.Linear(D, D)
        self.activation = nn.ReLU()

    def forward(self, X, STE):
        batch_size = X.shape[0]

        # 拼接X和STE -> 2D
        X = torch.cat((X, STE), dim=-1)

        # 对Q,K,V进行线性变换
        Q = self.mlp_Q(X)
        Q = self.activation(Q)
        K = self.mlp_K(X)
        K = self.activation(K)
        V = self.mlp_V(X)
        V = self.activation(V)
        # D = num_heads * dim_heads
        # 对Q,K,V进行多头注意力机制的切分 [batch_size, num_step, num_vertex, num_heads * dim_heads]
        # -> [num_heads * batch_size, num_step, num_vertex, dim_heads]
        Q = torch.cat(torch.split(Q, self.num_heads, dim=-1), dim=0)
        K = torch.cat(torch.split(K, self.num_heads, dim=-1), dim=0)
        V = torch.cat(torch.split(V, self.num_heads, dim=-1), dim=0)

        # 计算attention score
        # [num_heads * batch_size, num_step, num_vertex, num_vertex]
        attention_score = torch.matmul(Q, K.permute(0,1,3,2)) / (self.dim_heads ** 0.5)
        attention_score = F.softmax(attention_score, dim=-1)

        # 计算attention output
        attention_output = torch.matmul(attention_score, V)

        # 合并多头注意力机制的输出 -> [batch_size, num_step, num_vertex, D]
        X = torch.cat(torch.split(attention_output, batch_size, dim=0), dim=-1)

        # 输出层
        X = self.mlp_output(X)
        X = self.activation(X)
        return X


class GatedFusion(nn.Module):
    '''
    gated fusion
    H_spatial:      [batch_size, num_step, num_vertex, D]
    H_temporal:     [batch_size, num_step, num_vertex, D]
    return:         [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.mlp_Xs = nn.Linear(D, D, bias=False)
        self.mlp_Xt = nn.Linear(D, D)
        self.mlp_output1 = nn.Linear(D, 2*D)
        self.activations = nn.ReLU()
        self.mlp_output2 = nn.Linear(2*D, D)

    def forward(self, H_spatial, H_temporal):
        Xs = self.mlp_Xs(H_spatial)
        Xt = self.mlp_Xt(H_temporal)
        gate = torch.sigmoid(torch.add(Xs, Xt))
        output = torch.add(torch.mul(gate, H_spatial), torch.mul(1-gate, H_temporal))
        output = self.mlp_output1(output)
        output = self.activations(output)
        output = self.mlp_output2(output)
        
        return output


class GAF(nn.Module):
    '''
        INPUT:
            X:      [batch_size, num_his, num_vertex]
            TE:     [batch_size, num_his + num_pred, 2](time-of-day, day-of-week)
            SE:     [num_vertex, num_heads * dim_heads]
            configs:
                - num_heads
                - dim_heads
                - num_his
                - factor=3
                - dropout=0.1
                - d_ff=128
                - moving_avg=16
                - encoder_layers=1
                - decoder_layers=1
        OUTPUT:
            Y_hat:  [batch_size, num_pred, num_vertex]
    '''

    def __init__(self, SE, configs):
        super(GAF, self).__init__()
        # Embedding
        D = configs['num_heads'] * configs['dim_heads']
        self.num_his = configs['num_his']
        self.SE = SE
        self.STEmbedding = STEmbedding(D)
        
        # linear layer
        self.linear_x = nn.Linear(1, D)
        self.linear_decoder_xste = nn.Linear(2*D, D)
        # Decomp
        kernel_size = configs['moving_avg']
        self.decomp = series_decomp(kernel_size)

        # Spatial Attention
        self.spatial_attention = SpatialAttention(configs['num_heads'], configs['dim_heads'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            configs['factor'],
                            attention_dropout=configs['dropout'],
                        ),
                        d_model=D, 
                        n_heads=configs['num_heads']
                    ),
                    d_model=D,
                    d_ff=configs['d_ff'],
                    moving_avg=configs['moving_avg'],
                    dropout=configs['dropout'],
                ) for l in range(configs['encoder_layers'])
            ],
            norm_layer=my_Layernorm(D)
        )

        # Gated Fusion
        self.gate = GatedFusion(D)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            configs['factor'],
                            attention_dropout=configs['dropout'],
                        ),
                        D,
                        configs['num_heads']
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            configs['factor'],
                            attention_dropout=configs['dropout'],
                        ),
                        D,
                        configs['num_heads']
                    ),
                    d_model=D,
                    d_ff=configs['d_ff'],
                    moving_avg=configs['moving_avg'],
                    dropout=configs['dropout'],
                )
                for l in range(configs['decoder_layers'])
            ],
            norm_layer=my_Layernorm(D),
        )
    
        # output layer
        self.linear_out1 = nn.Linear(D,D)
        self.activation = nn.ReLU()
        self.linear_out2 = nn.Linear(D,1)
    
    
    def forward(self, X, TE):
        # X -> [batch_size, num_his, num_vertex, 1]
        X = X.unsqueeze(-1)
        X = self.linear_x(X)
        # STEmbedding
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]

        # Encoder
        enc_out1 = self.spatial_attention(X, STE_his)
        enc_out2 = self.encoder(X, STE_his)

        # ipdb.set_trace()
        # gated fusion
        enc_out = self.gate(enc_out1, enc_out2)
        
        # Decoder
        dec_out1, dec_out2 = self.decoder(X, enc_out, STE_pred)
        dec_out = dec_out1 + dec_out2
        
        out = self.linear_out1(dec_out)
        out = self.activation(out)
        out = self.linear_out2(out)
        return out.squeeze(-1)