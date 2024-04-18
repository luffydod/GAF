import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dims, output_dims, activations, use_bias=True):
        super(MLP, self).__init__()
        if isinstance(output_dims, int):
            output_dims = [output_dims]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(output_dims, tuple):
            output_dims = list(output_dims)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(output_dims) == list
        assert len(input_dims) == len(output_dims) == len(activations)
        layers = []
        for input_dim, output_dim, activation in zip(input_dims, output_dims, activations):
            # Add Linear layer
            fc_layer = nn.Linear(input_dim, output_dim, bias=use_bias)
            layers.append(fc_layer)

            # Activation function
            if activation is None:
                pass
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f'Wrong activation function: {activation}')

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # [B, T, V, in_D]
        res = self.mlp(x)
        # [B, T, V, out_D]
        return res


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
        self.mlp_spatial = MLP(
            input_dims=[D,D],
            output_dims=[D,D],
            activations=['relu', None]
        )
        self.mlp_temporal = MLP(
            input_dims=[295,D],
            output_dims=[D,D],
            activations=['relu', None]
        )   # input_dims = 7 + T(=288)
    
    def forward(self, SE, TE, T=288):
        # Spatial Embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.mlp_spatial(SE)
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
        TE = self.mlp_temporal(TE)

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
        self.mlp_Q = MLP(
            input_dims=2*D,
            output_dims=D,
            activations='relu'
        )
        self.mlp_K = MLP(
            input_dims=2*D,
            output_dims=D,
            activations='relu'
        )
        self.mlp_V = MLP(
            input_dims=2*D,
            output_dims=D,
            activations='relu'
        )
        self.mlp_output = MLP(
            input_dims=D,
            output_dims=D,
            activations='relu'
        )
    
    def forward(self, X, STE):
        batch_size = X.shape[0]

        # 拼接X和STE -> 2D
        X = torch.cat((X, STE), dim=-1)

        # 对Q,K,V进行线性变换
        Q = self.mlp_Q(X)
        K = self.mlp_K(X)
        V = self.mlp_V(X)
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

        return X

class TemporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    num_heads:      number of attention heads
    dim_heads:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, num_heads, dim_heads, mask=True):
        super(TemporalAttention, self).__init__()
        D = num_heads * dim_heads
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.mask = mask
        self.mlp_Q = MLP(
            input_dims=2*D,
            output_dims=D,
            activations='relu'
        )
        self.mlp_K = MLP(
            input_dims=2*D,
            output_dims=D,
            activations='relu'
        )
        self.mlp_V = MLP(
            input_dims=2*D,
            output_dims=D,
            activations='relu'
        )
        self.mlp_output = MLP(
            input_dims=D,
            output_dims=D,
            activations='relu'
        )

    def forward(self, X, STE):
        batch_size, num_step, num_vertex, _ = X.shape
        X = torch.cat((X, STE), dim=-1)
        # 2*D -> D [batch_size, num_step, num_vertex, num_heads * dim_heads]
        Q = self.mlp_Q(X)
        K = self.mlp_K(X)
        V = self.mlp_V(X)
        # [num_heads * batch_size, num_step, num_vertex, dim_heads]
        Q = torch.cat(torch.split(Q, self.num_heads, dim=-1), dim=0)
        K = torch.cat(torch.split(K, self.num_heads, dim=-1), dim=0)
        V = torch.cat(torch.split(V, self.num_heads, dim=-1), dim=0)
        # Q: [num_heads * batch_size, num_vertex, num_step, dim_heads]
        # K: [num_heads * batch_size, num_vertex, dim_heads, num_step]
        # V: [num_heads * batch_size, num_vertex, num_step, dim_heads]
        Q = Q.permute(0,2,1,3)
        K = K.permute(0,2,3,1)
        V = V.permute(0,2,1,3)

        # 计算attention score
        attention_score = torch.matmul(Q, K) / (self.dim_heads ** 0.5)

        # mask attention score
        if self.mask==True:
            mask = torch.ones(num_step, num_step)
            # [1,1,num_step,num_step]
            mask = torch.tril(mask).unsqueeze(0).unsqueeze(0)
            # [num_heads*batch_size, num_vertex, num_step, num_step]
            mask = mask.repeat(self.num_heads * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention_score = torch.where(mask, attention_score,-float('inf'))
        # softmax
        attention = F.softmax(attention_score, dim=-1)
        # [num_heads * batch_size, num_vertex, num_step, dim_heads]
        X = torch.matmul(attention, V)
        X = X.permute(0,2,1,3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.mlp_output(X)
        
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
        self.mlp_Xs = MLP(
            input_dims=D,
            output_dims=D,
            activations=None,
            use_bias=False
        )
        self.mlp_Xt = MLP(
            input_dims=D,
            output_dims=D,
            activations=None,
            use_bias=True
        )
        self.mlp_output = MLP(
            input_dims=[D,D],
            output_dims=[D,D],
            activations=['relu', None]
        )

    def forward(self, H_spatial, H_temporal):
        Xs = self.mlp_Xs(H_spatial)
        Xt = self.mlp_Xt(H_temporal)
        gate = torch.sigmoid(torch.add(Xs, Xt))
        output = torch.add(torch.mul(gate, H_spatial), torch.mul(1-gate, H_temporal))
        output = self.mlp_output(output)
        
        return output

class piggyBlock(nn.Module):
    def __init__(self, num_heads, dim_heads, mask=False):
        super(piggyBlock, self).__init__()
        self.SpatialAttention = SpatialAttention(num_heads, dim_heads)
        self.TemporalAttention = TemporalAttention(num_heads, dim_heads, mask)
        self.GatedFusion = GatedFusion(num_heads * dim_heads)

    def forward(self, X, STE):
        H_spatial = self.SpatialAttention(X, STE)
        H_temporal = self.TemporalAttention(X, STE)
        H = self.GatedFusion(H_spatial, H_temporal)
        
        return torch.add(X, H)

class TransformAttention(nn.Module):
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    num_heads:      number of attention heads
    dim_heads:      dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    '''

    def __init__(self, num_heads, dim_heads):
        super(TransformAttention, self).__init__()
        D = num_heads * dim_heads
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.mlp_Q = MLP(
            input_dims=D,
            output_dims=D,
            activations='relu'
        )
        self.mlp_K = MLP(
            input_dims=D,
            output_dims=D,
            activations='relu'
        )
        self.mlp_V = MLP(
            input_dims=D,
            output_dims=D,
            activations='relu'
        )
        self.mlp_output = MLP(
            input_dims=D,
            output_dims=D,
            activations='relu'
        )

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        # what is the meaning of this?
        Q = self.mlp_Q(STE_pred)
        K = self.mlp_K(STE_his)
        V = self.mlp_V(X)
        Q = torch.cat(torch.split(Q, self.num_heads, dim=-1), dim=0)
        K = torch.cat(torch.split(K, self.num_heads, dim=-1), dim=0)
        V = torch.cat(torch.split(V, self.num_heads, dim=-1), dim=0)
        # Q:    [num_heads * batch_size, num_vertex, num_pred, dim_heads]
        # K:    [num_heads * batch_size, num_vertex, dim_heads, num_his]
        # V:    [num_heads * batch_size, num_vertex, num_his, dim_heads]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 3, 1)
        V = V.permute(0, 2, 1, 3)

        attention_score = torch.matmul(Q, K) / (self.dim_heads ** 0.5)
        attention_score = F.softmax(attention_score, dim=-1)

        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention_score, V)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.mlp_output(X)
        
        return X

class GGBond(nn.Module):
    '''
        INPUT:
            X:      [batch_size, num_his, num_vertex]
            TE:     [batch_size, num_his + num_pred, 2](time-of-day, day-of-week)
            SE:     [num_vertex, num_heads * dim_heads]
            num_his:    number of history steps
            num_pred:   number of prediction steps
            T:          one day is divided into T steps
            num_block:  number of piggyBlocks in the encoder/decoder
            num_heads:  number of attention heads
            dim_heads:  dimension of each attention head outputs
        OUTPUT:
            Y_hat:  [batch_size, num_pred, num_vertex]
    '''

    def __init__(self, SE, num_his, num_heads, dim_heads, num_block):
        super(GGBond, self).__init__()
        D = num_heads * dim_heads
        self.num_his = num_his
        self.SE = SE
        self.STEmbedding = STEmbedding(D)
        self.piggyBlockEncoder = nn.ModuleList([
            piggyBlock(num_heads, dim_heads) for _ in range(num_block)
        ])
        self.piggyBlockDecoder = nn.ModuleList([
            piggyBlock(num_heads, dim_heads) for _ in range(num_block)
        ])
        self.TransformAttention = TransformAttention(num_heads, dim_heads)
        self.mlp_input = MLP(
            input_dims=[1,D],
            output_dims=[D,D],
            activations=['relu',None]
        )
        self.mlp_output = MLP(
            input_dims=[D,D],
            output_dims=[D,1],
            activations=['relu', None]
        )

    def forward(self, X, TE):
        # input -> [batch_size, num_his, num_vertex, D]
        X = self.mlp_input(X.unsqueeze(-1))

        # STEmbedding
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:,self.num_his:]

        # Encoder
        for net in self.piggyBlockEncoder:
            X = net(X, STE_his)
        
        # TransformAttention
        X = self.TransformAttention(X, STE_his, STE_pred)

        # Decoder
        for net in self.piggyBlockDecoder:
            X = net(X, STE_pred)
        
        # output
        X = self.mlp_output(X)
        return X.squeeze(-1)