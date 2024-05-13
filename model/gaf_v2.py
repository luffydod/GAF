import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.Embedding import XSEembedding, XTEembedding
from model.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from model.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import ipdb


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
        self.linear_Q = nn.Linear(D, D)
        self.linear_K = nn.Linear(D, D)
        self.linear_V = nn.Linear(D, D)
        self.linear_output1 = nn.Linear(D, D)
        self.linear_output2 = nn.Linear(D, D)
        self.activation = nn.ReLU()

    def forward(self, xse):
        batch_size = xse.shape[0]

        # 对Q,K,V进行线性变换
        Q = self.linear_Q(xse)
        K = self.linear_K(xse)
        V = self.linear_V(xse)
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
        res = torch.cat(torch.split(attention_output, batch_size, dim=0), dim=-1)

        # 输出层
        res = self.linear_output1(res)
        res = self.activation(res)
        res = self.linear_output2(res)
        return res


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
        self.xte_embedding = XTEembedding(D)
        self.xse_embedding = XSEembedding(D)
        
        # linear layer
        self.input_linear1 = nn.Linear(1, D)
        self.input_linear2 = nn.Linear(D, D)
        self.output_linear1 = nn.Linear(D, D)
        self.output_linear2 = nn.Linear(D,1)
        self.activation = F.relu

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
        self.gate1 = GatedFusion(D)
        self.gate2 = GatedFusion(D)
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
    
    
    
    def forward(self, X, TE):
        # X -> [batch_size, num_his, num_vertex, D]
        X = X.unsqueeze(-1)
        X = self.input_linear1(X)
        X = self.activation(X)
        X = self.input_linear2(X)

        # Embedding
        xse = self.xse_embedding(X, self.SE)
        xte = self.xte_embedding(X, TE)
        xte_his  = xte[:, :self.num_his, ...]
        xte_pred = xte[:, self.num_his:, ...]

        # Encoder
        enc_out1 = self.spatial_attention(xse)
        enc_out2 = self.encoder(xte_his)

        # ipdb.set_trace()
        # gated fusion
        enc_out = self.gate1(enc_out1, enc_out2)
        
        # Decoder
        dec_out1, dec_out2 = self.decoder(xte_his, enc_out, xte_pred)
        # dec_out = dec_out1 + dec_out2
        dec_out = self.gate2(dec_out1, dec_out2)
        out = self.output_linear1(dec_out)
        out = self.activation(out)
        out = self.output_linear2(out)

        # [B,L,V,1] -> [B,L,V]
        return out.squeeze(-1)