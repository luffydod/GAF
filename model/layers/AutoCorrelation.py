import torch
import torch.nn as nn
import math
import ipdb

class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, factor=1, scale=None, attention_dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)


    def time_delay_agg(self, values, corr):
        """
            - training
                SpeedUp version of Autocorrelation (a batch-normalization style design)
                This is for the training phase.
            - inference
                Standard version of Autocorrelation
            - values: [B, H, d, V, L]
            - corr: [B, H, d, V, L]
        """
        batch, head, channel, num_vertex, length = values.shape

        # find top k
        top_k = int(self.factor * math.log(length))
        # mean_value [B,V,L]
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        
        if self.training:
            # 计算到batch mean
            index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
            # index [V,top_k] -> [B,V,top_k]
            index_expanded = index.unsqueeze(0).expand(batch, -1, -1)
            # weights [B,V,top_k]
            weights = torch.gather(mean_value, dim=-1, index=index_expanded)
        else:
            weights, index_expanded = torch.topk(mean_value, top_k, dim=-1)
        
        # update corr [B,V,top_k]
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        tmp_values = values # [B,H,d,V,L]
        # index_expanded [B,H,d,V,L,top_k]
        index_expanded = index_expanded.unsqueeze(1).unsqueeze(1).unsqueeze(4).expand(-1,head,channel,-1,length,-1)
        # seq_index [B,H,d,V,L,top_k]
        seq_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(values.shape[0],head,channel,num_vertex,-1,top_k).to(values.device)
        res_index = (seq_index + index_expanded) % length
        # expand tmp_values
        tmp_values = tmp_values.unsqueeze(-1).expand(-1,-1,-1,-1,-1,top_k)
        rolled_values = torch.gather(tmp_values, dim=-2, index=res_index)

        # expand tmp_corr
        tmp_corr = tmp_corr.unsqueeze(1).unsqueeze(1).unsqueeze(4).expand(-1,head,channel,-1,length,-1)
        delays_agg = (rolled_values * tmp_corr).sum(-1)
        # [B,H,d,V,L]
        return delays_agg


    def forward(self, queries, keys, values):
        '''
            Q: [B,L,V,n_heads,d_keys]
            K: [B,S,V,n_heads,d_keys]
            V: [B,S,V,n_heads,d_values]
        '''
        L = queries.shape[1]
        S = values.shape[1]
        # 对齐序列长度
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), ...]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, ...]
            keys = keys[:, :L, ...]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 3, 4, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 3, 4, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        # corr: [B,H,d_keys,V,L]
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # time delay agg
        res = self.time_delay_agg(values.permute(0, 3, 4, 2, 1).contiguous(), corr).permute(0, 4, 3, 2, 1).contiguous()
        # V: [B,H,d,V,L]->permute[B,L,V,H,d]
        return res


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        '''
            V: num_vertex
            B: batch_size
            L: Q->num_steps
            S: K->num_steps
            H: num_heads
        '''
        B, L, V, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads

        # ipdb.set_trace()
        # d_keys = d_values = d_model // n_heads
        # Q: [B,L,V,d_model]->[B,L,V,n_heads,d_keys]
        # K: [B,S,V,d_model]->[B,S,V,n_heads,d_keys]
        # V: [B,S,V,d_model]->[B,S,V,n_heads,d_values]
        queries = self.query_projection(queries).view(B, L, V, H, -1)
        keys = self.key_projection(keys).view(B, S, V, H, -1)
        values = self.value_projection(values).view(B, S, V, H, -1)

        out = self.inner_correlation(queries, keys, values)
        # out: [B,L,V,H,d] -> [B,L,V,H*d] -> [B,L,V,d_model]
        out = out.view(B, L, V, -1)
        out = self.out_projection(out)

        return out
