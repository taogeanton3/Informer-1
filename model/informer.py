from torch import nn

from model.decoder import Decoder
from model.encoder import Encoder


class Informer(nn.Module):
    def __init__(
        self,
        d_k=64,
        d_v=64,
        d_model=512,
        d_ff=512,
        n_heads=8,
        e_layer=3,
        d_layer=2,
        e_stack=3,
        d_feature=7,
        d_mark=4,
        dropout=0.1,
        c=5,
    ):
        super(Informer, self).__init__()

        self.encoder = Encoder(
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layer=e_layer,
            n_stack=e_stack,
            d_feature=d_feature,
            d_mark=d_mark,
            dropout=dropout,
            c=c,
        )
        self.decoder = Decoder(
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layer=d_layer,
            d_feature=d_feature,
            d_mark=d_mark,
            dropout=dropout,
            c=c,
        )

        self.projection = nn.Linear(d_model, d_feature, bias=True)

    def forward(self, enc_x, enc_mark, dec_in, dec_mark):
        enc_outputs = self.encoder(enc_x, enc_mark)
        dec_outputs = self.decoder(dec_in, dec_mark, enc_outputs)
        dec_outputs = self.projection(dec_outputs)

        return dec_outputs
