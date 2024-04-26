import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool2d(kernel_size=[1,kernel_size], stride=[1,stride], padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, ...].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        end = x[:, -1:, ...].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        x = torch.cat([front, x, end], dim=1)
        # [b,l,v,c] -> permute -> [b,v,c,l] -> pooling -> permute -> [b,l,v,c]
        x = self.avg(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=13, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.conv1 = nn.Conv2d(
                        in_channels=d_model,
                        out_channels=d_ff,
                        kernel_size=[1,1],
                        stride=[1,1],
                        bias=False)
        self.conv2 = nn.Conv2d(
                        in_channels=d_ff,
                        out_channels=d_model,
                        kernel_size=[1,1],
                        stride=[1,1],
                        bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        # [b, l, N, c] -> permute -> [b,c,N,l] -> conv2d -> [b,c1,N'=N,l'=l]
        y = self.dropout(self.activation(self.conv1(y.permute(0,3,2,1))))
        y = self.dropout(self.conv2(y).permute(0,3,2,1))
        res, _ = self.decomp2(x + y)
        return res


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(2*64, 64)
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, X, STE):
        # 拼接X和STE -> 2D
        X = torch.cat((X, STE), dim=-1)
        X = self.linear(X)
        x = X 
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        res = X + x
        if self.norm is not None:
            res = self.norm(res)

        return res


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear = nn.Linear(2*d_model, d_model)
        self.conv1 = nn.Conv2d(
                        in_channels=d_model,
                        out_channels=d_ff,
                        kernel_size=[1,1],
                        stride=[1,1],
                        bias=False)
        self.conv2 = nn.Conv2d(
                        in_channels=d_ff,
                        out_channels=d_model,
                        kernel_size=[1,1],
                        stride=[1,1],
                        bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv2d(
                            in_channels=d_model,
                            out_channels=d_model,
                            kernel_size=[1,3],
                            stride=[1,1],
                            padding=[0,1],
                            padding_mode='circular',
                            bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        x = x + self.dropout(self.self_attention(x, x, x))
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross))
        x, trend2 = self.decomp2(x)
        y = x
        # [B,L,N,C] -> permute -> [B,C,N,L] -> conv2d -> [B,C',N'=N,L'=L]
        y = self.dropout(self.activation(self.conv1(y.permute(0, 3, 2, 1))))
        # [B,L,N,d]
        y = self.dropout(self.conv2(y).permute(0,3,2,1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3

        residual_trend = self.projection(residual_trend.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, trend):
        # 拼接X和STE -> 2D
        X = torch.cat((X, trend), dim=-1)
        X = self.linear(X)
        x = X

        for layer in self.layers:
            x, residual_trend = layer(x, cross)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        return x, trend
