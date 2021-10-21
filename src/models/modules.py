import torch
import torch.nn as nn


class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight[self.padding_idx], 0)


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)


class LSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, param in self.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)


class TransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout=0.1,
        activation="relu",
        *args,
        **kwargs
    ):
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, activation=activation
        )
        norm = nn.LayerNorm(d_model)
        super().__init__(encoder_layers, num_layers, norm=norm)
        # initrange = 0.1
        # self.weight.data.uniform_(-initrange, initrange)
