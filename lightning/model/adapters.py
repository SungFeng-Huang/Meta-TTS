import os
import json
import torch
import pytorch_lightning as pl

from torch import nn
from resemblyzer import VoiceEncoder


class SpeakerAdaLNAdapter(nn.Module):
    def __init__(self, d_emb, d_hidden):
        super().__init__()
        self.d_hidden = d_hidden
        self.linear = nn.Linear(d_emb, 2*d_hidden)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.ones_(self.linear.bias.reshape(2, d_hidden)[0])

    def forward(self, spk_emb):
        # spk_emb: (B, d_emb)
        output = self.linear(spk_emb)
        output = output.reshape(-1, 2, self.d_hidden)
        # (B, mean/std, d_hidden)
        return output


class ProsodyEncoder(nn.Module):
    def __init__(self, d_emb, n_layers, log_f0=True, energy=False):
        super().__init__()
        n_input_channels = int(log_f0) + int(energy)
        n_AdaLN = 2 * n_layers
        self.d_emb = d_emb
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_input_channels, d_emb, n_layers, batch_first=True)
        self.linear = nn.Linear(d_emb, d_emb)
        self.relu = nn.ReLU()

    def forward(self, prosody):
        self.lstm.flatten_parameters()
        # prosody: (B, 1/2)
        _, (hidden, _) = self.lstm(prosody)
        # hidden[-1]: (B, d_emb)
        output = self.relu(self.linear(hidden[-1]))
        return output / torch.norm(output, dim=1, keepdim=True)


class ProsodyAdaLNAdapter(nn.Module):
    def __init__(self, d_emb, d_hidden):
        super().__init__()
        self.d_hidden = d_hidden
        self.linear = nn.Linear(d_emb, 2*d_hidden)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.ones_(self.linear.bias.reshape(2, d_hidden)[0])

    def forward(self, psd_emb):
        # psd_emb: (B, d_emb)
        output = self.linear(psd_emb)
        output = output.reshape(-1, 2, self.d_hidden)
        # (B, mean/std, d_hidden)
        return output
