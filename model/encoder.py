import torch
import torch.nn.functional as F
from torch import nn

from model.attention import ProbAttention

from model.embed import DataEmbedding


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=(3,),
            padding=(1,),
            stride=(1,),
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, c, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = ProbAttention(d_k, d_v, d_model, n_heads, c, dropout, mix=False)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, attn_mask=None):
        x = self.attention(x, x, x, attn_mask=attn_mask)

        residual = x.clone()
        x = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        y = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm(residual + y)


class Encoder(nn.Module):
    def __init__(
        self,
        d_k,
        d_v,
        d_model,
        d_ff,
        n_heads,
        n_layer,
        n_stack,
        d_feature,
        d_mark,
        dropout,
        c,
    ):
        super(Encoder, self).__init__()

        self.embedding = DataEmbedding(d_feature, d_mark, d_model, dropout)

        self.stacks = nn.ModuleList()
        for i in range(n_stack):
            stack = nn.Sequential()
            stack.add_module(
                "elayer" + str(i) + "0",
                EncoderLayer(d_k, d_v, d_model, d_ff, n_heads, c, dropout),
            )

            for j in range(n_layer - i - 1):
                stack.add_module("clayer" + str(i) + str(j + 1), ConvLayer(d_model))
                stack.add_module(
                    "elayer" + str(i) + str(j + 1),
                    EncoderLayer(d_k, d_v, d_model, d_ff, n_heads, c, dropout),
                )
            stack.add_module("nlayer", nn.LayerNorm(d_model))
            self.stacks.append(stack)

    def forward(self, enc_x, enc_mark):
        x = self.embedding(enc_x, enc_mark)

        out = []
        for i, stack in enumerate(self.stacks):
            inp_len = x.shape[1] // (2 ** i)
            y = x[:, -inp_len:, :]
            y = stack(y)
            out.append(y)
        out = torch.cat(out, -2)

        return out
