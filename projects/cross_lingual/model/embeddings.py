import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import Optional


class MultilingualEmbedding(nn.Module):
    def __init__(self, id2symbols, dim: int, padding_idx: int=0):
        super().__init__()
        self.id2symbols = id2symbols
        self.dim = dim
        self.padding_idx = padding_idx

        self.tables = nn.ParameterDict()
        for symbol_id, v in id2symbols.items():
            if len(v) > 0:
                w_init = torch.randn(len(v), dim)
                std = sqrt(2.0 / (len(v) + dim))
                val = sqrt(3.0) * std  # uniform bounds for std
                w_init.uniform_(-val, val)
                w_init[padding_idx].fill_(0)
                self.tables[f"table-{symbol_id}"] = nn.Parameter(w_init)

    def forward(self, x, symbol_id: Optional[str]=None):
        if symbol_id is None:
            # for k, p in self.tables.items():
            #     print(k, p.shape)
            concat_tables = torch.cat([p for p in self.tables.values()], dim=0)
            return F.embedding(x, concat_tables, padding_idx=self.padding_idx)
        return F.embedding(x, self.tables[f"table-{symbol_id}"], padding_idx=self.padding_idx)
