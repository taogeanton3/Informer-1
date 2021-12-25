import torch.nn.functional as F
from torch import nn

from model.attention import ProbAttention, FullAttention

from model.embed import DataEmbedding
from utils.mask import get_attn_subsequence_mask


class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, c, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = ProbAttention(d_k, d_v, d_model, n_heads, c, dropout, mix=True)
        self.cross_attention = FullAttention(d_k, d_v, d_model, n_heads, dropout, mix=False)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, x, enc_outputs, self_mask=None, cross_mask=None):
        x = self.self_attention(x, x, x, attn_mask=self_mask)
        x = self.cross_attention(x, enc_outputs, enc_outputs, attn_mask=cross_mask)

        residual = x.clone()
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(x).transpose(-1, 1))
        return self.norm(residual + y)


class Decoder(nn.Module):
    def __init__(
        self, d_k, d_v, d_model, d_ff, n_heads, n_layer, d_feature, d_mark, dropout, c
    ):
        super(Decoder, self).__init__()

        self.embedding = DataEmbedding(d_feature, d_mark, d_model, dropout)

        self.decoder = nn.ModuleList()
        for _ in range(n_layer):
            self.decoder.append(
                DecoderLayer(d_k, d_v, d_model, d_ff, n_heads, c, dropout)
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dec_in, dec_mark, enc_outputs):
        y = self.embedding(dec_in, dec_mark)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(y)

        for layer in self.decoder:
            y = layer(y, enc_outputs, self_mask=dec_self_attn_subsequence_mask)

        y = self.norm(y)
        return y
