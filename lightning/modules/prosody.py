import torch
from torch import nn


class ProsodyEncoder(nn.Module):
    def __init__(self, d_emb, n_layers, log_f0=True, energy=False):
        super().__init__()
        n_input_channels = int(log_f0) + int(energy)
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
