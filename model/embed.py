import math

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # 不需要训练更新

    def forward(self, x):
        x = self.pe[: x.size(1), :]
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_mark, d_model):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(d_mark, d_model)

    def forward(self, x):
        return self.embed(x)


class TokenEmbedding(nn.Module):
    def __init__(self, d_feature, d_model):
        super(TokenEmbedding, self).__init__()
        self.Conv = nn.Conv1d(
            in_channels=d_feature,
            out_channels=d_model,
            kernel_size=(3,),
            padding=(1,),
            stride=(1,),
            padding_mode="circular",
        )

        nn.init.kaiming_normal_(
            self.Conv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, x):
        x = self.Conv(x.permute(0, 2, 1)).transpose(1, 2)  # 卷积是在最后一个维度进行的
        return x


class DataEmbedding(nn.Module):
    def __init__(self, d_feature, d_mark, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(d_feature=d_feature, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.time_embedding = TimeFeatureEmbedding(d_mark=d_mark, d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = (
            self.value_embedding(x)
            + self.position_embedding(x)
            + self.time_embedding(x_mark)
        )
        return self.dropout(x)
