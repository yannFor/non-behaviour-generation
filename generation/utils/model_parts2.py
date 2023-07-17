import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import constants


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        #self.ln = nn.LayerNorm([channels]).to('cuda:0')

        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x,c=None):
        #x = x.swapaxes(1, 2).to('cuda:0')
        #x_ln = self.ln(x).to('cuda:0')
        x = x.swapaxes(1, 2)
        x_ln = self.ln(x)
        if c is None:
            attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        else:
            #c_ln = nn.LayerNorm([self.channels])(c)
            attention_value,_ = self.mha(x_ln, c, c)

        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel,mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel, padding="same", bias=True),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel, padding="same", bias=True),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.silu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels,kernel,emb_dim=256):
        super().__init__()
        #self.maxpool_conv = nn.Sequential(
        #    nn.MaxPool1d(2),
        #    DoubleConv(in_channels, in_channels,kernel=kernel, residual=True),
        #    DoubleConv(in_channels, out_channels,kernel=kernel,residual=False),
        #)
        self.maxpool_conv = self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, in_channels,kernel=kernel, residual=True),
            DoubleConv(in_channels, in_channels,kernel=kernel,residual=True),
            nn.Conv1d(in_channels,out_channels,4,2,1)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                3496,
                out_channels
            ))

    def forward(self, x, t=None):
        x = self.maxpool_conv(x)
        #print(t.shape)
        #emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        if t is None:
            return x
        else:
            emb = self.emb_layer(t)[:,:, None].repeat(1, 1, x.shape[-1])
            return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels,kernel, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels,kernel=kernel,residual=True),
            DoubleConv(in_channels, in_channels,kernel=kernel, residual=True),
        )
        self.convfinal = nn.Conv1d(in_channels,out_channels,3,padding="same")

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                3496,
                out_channels
            ),
        )

    def forward(self, x, skip_x=None, t=None):
        x = self.up(x)
        x = torch.cat([skip_x,x],1)
        x = self.conv(x)
        x = self.convfinal(x)
        

        if t is None:
            return x
        else:
            # emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            #print(self.emb_layer)
            emb = self.emb_layer(t)[:,:, None].repeat(1, 1, x.shape[-1])
            #print(emb.shape)
        return x + emb


def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))