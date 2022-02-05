import numpy as np
import torch.nn.functional as F
from functools import partial

import torch
from torch import nn

import pytorch_lightning as pl

from ..utils import MatchingGraphInfo
import transformer.Constants as Constants
from text.define import LANG_ID2SYMBOLS


class Codebook(pl.LightningModule):
    def __init__(self, size, d_in, dim, num_heads=4, temperature=None):
        super().__init__()
        self.size = size
        self.d_in = d_in
        self.dim = dim
        self.num_heads = num_heads

        self.banks = nn.Parameter(torch.randn(size, dim))
        self.q_linear = nn.Linear(d_in, dim)
        self.softmax = nn.Softmax(dim=3)
        assert dim % num_heads == 0

        self.temperature = temperature if temperature is not None else (dim // num_heads) ** 0.5
    
    def forward(self, ref, mask=None):
        """
        Input:
            ref: Reference tensor (query) with shape (B, L, input_dim).
            mask: Attention mask with same shape as the output (B, nH, L, size).
        Output: 
            Attention map with size (B, nH, L, size).
        """
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)
        ref[ref != ref] = 0

        B, L, _ = ref.shape
        q = self.q_linear(ref).view(B, L, self.num_heads, self.dim // self.num_heads)
        q = q.transpose(1, 2).contiguous()  # B, nH, L, dim // nH
        k = self.banks.view(1, self.size, self.num_heads, self.dim // self.num_heads)
        k = k.transpose(1, 2).contiguous()  # 1 x nH x size x dim // nH
        attn = q @ k.transpose(2, 3) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        return self.softmax(attn)

    def get_matching(self, ref, lang_id):
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)
        ref[ref != ref] = 0

        mask = torch.nonzero(ref.sum(dim=1), as_tuple=True)
        ref = ref.unsqueeze(0)  # 1, N, d_feat
        attn = self.forward(ref).squeeze(0)  # nH, L, size
        infos = []
        for i in range(self.num_heads):
            info = MatchingGraphInfo({
                "title": f"Head-{i}",
                "y_labels": [LANG_ID2SYMBOLS[lang_id][int(m)] for m in mask[0]],
                "x_labels": [str(i) for i in range(1, self.size + 1)],
                "attn": attn[i][mask].detach().cpu().numpy(),
                "quantized": False,
            })
            infos.append(info)
        return infos


# class HardBank(pl.LightningModule):
#     """
#     Input an attention map A, combine banks according to quantized A.
#     """
#     def __init__(self, model_config, algorithm_config):
#         super().__init__()
#         self.num_heads = 4

#     def forward(self, attn):
#         with torch.no_grad():
#             weighting_matrix = torch.zeros(
#                 ref.shape[0], self.att_banks.shape[0],
#                 device=self.device
#             )
#             # NOTE: Can't directly assign = 1 since need to ensure tensors on the same device, use ones_like() instead. 
#             weighting_matrix[ref_mask, attn.argmax(1)] = torch.ones_like(ref_mask).float()
#             # padding_idx
#             weighting_matrix[Constants.PAD].fill_(0)
#         weighted_embedding = weighting_matrix @ self.emb_banks



class SoftBank(pl.LightningModule):
    """
    Input an attention map A, combine banks according to A, same with the last step in normal attention.
    """
    def __init__(self, size, d_out, num_heads=4):
        super().__init__()
        self.size = size
        self.d_out = d_out
        self.num_heads = num_heads
        self.banks = nn.Parameter(torch.randn(size, d_out))
        assert d_out % num_heads == 0

    def forward(self, attn, pad=False):
        """
        Input:
            attn: Tensor with shape (B, nH, L, size).
            pad: If true, pad embedding is set to 0.
        Output: 
            Tensor with shape (B, L, d_out).
        """
        B, nH, L, _ = attn.shape
        v = self.banks.view(1, self.size, nH, self.d_out // nH)
        v = v.transpose(1, 2).contiguous()  # 1, nH, size, d_out // nH
        weighted_embedding = attn @ v  # B, nH, L, d_out // nH
        weighted_embedding = weighted_embedding.transpose(1, 2).contiguous().view(B, L, self.d_out)
        
        if pad:
            weighted_embedding[:, Constants.PAD] = 0
        return weighted_embedding


class ASRRefHead(pl.LightningModule):
    """
    Use ref to generate embedding, related to embedding table quality.
    """
    def __init__(self, model_config, algorithm_config):
        super().__init__()
        self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
        self.codebook_size = self.codebook_config["size"]
        self.d_feat = self.codebook_config["representation_dim"]
        self.d_word_vec = model_config["transformer"]["encoder_hidden"]

        self.num_heads = 4
        self.codebook = Codebook(self.codebook_size, self.d_feat, 
                                self.d_word_vec, num_heads=self.num_heads)
        self.banks = SoftBank(self.codebook_size, self.d_word_vec, num_heads=self.num_heads)
        self.tables = nn.ParameterDict({
            f"table-{lang_id}": nn.Parameter(torch.randn(len(v), self.d_word_vec))
        for lang_id, v in LANG_ID2SYMBOLS.items() if len(v) > 0})

    def forward(self, batch, ref, lang_id, mask=None):
        """
        Input:
            batch: Containing groundtruth text.
            ref: Embedding table with shape (N, d_feat), where N is number of phonemes.
            lang_id: Language ID.
        """
        ref = ref.unsqueeze(0)  # 1, N, d_feat
        attn = self.codebook(ref, mask)
        embedding = self.banks(attn, pad=True)  # 1, N, d_word_vec
        emb_texts = F.embedding(batch[3], embedding.squeeze(0), padding_idx=Constants.PAD)  # B, L, d_word_vec
        diff = (emb_texts.unsqueeze(2) - self.tables[f"table-{lang_id}"].unsqueeze(0).unsqueeze(0)) ** 2  # B, L, N, d_word_vec
        mse = diff.mean(dim=3)  # B, L, N
        # print("Score shape: ", mse.shape)
        return -mse

    def get_new_embedding(self, ref, lang_id, *args, **kwargs):
        ref = ref.unsqueeze(0)  # 1, N, d_feat
        attn = self.codebook(ref)
        embedding = self.banks(attn, pad=True)  # 1, N, d_word_vec
        return embedding.squeeze(0)

    def get_matching(self, ref_phn_feats, lang_id):
        mask = torch.nonzero(ref_phn_feats.sum(dim=1), as_tuple=True)
        ref = ref_phn_feats.unsqueeze(0)  # 1, N, d_feat
        attn = self.codebook(ref).squeeze(0)  # nH, L, size
        infos = []
        for i in range(self.num_heads):
            info = MatchingGraphInfo({
                "title": f"Head-{i}",
                "y_labels": [LANG_ID2SYMBOLS[lang_id][int(m)] for m in mask[0]],
                "x_labels": [str(i) for i in range(1, self.codebook_size + 1)],
                "attn": attn[i][mask].detach().cpu().numpy(),
                "quantized": False,
            })
            infos.append(info)
        return infos


class ASRCenterHead(pl.LightningModule):
    """
    Simple monolingual/multilingual ASR head (w/o ref), related to input quality.
    """
    def __init__(self, dim, multilingual=True):
        super().__init__()
        self.dim = dim
        self.multilingual = multilingual
        self.tables = nn.ParameterDict({
            f"table-{lang_id}": nn.Parameter(torch.randn(len(v), self.dim))
        for lang_id, v in LANG_ID2SYMBOLS.items() if len(v) > 0})

        # Create mask table (example below) only once
        # [[1 1 1 0 0 0]
        #  [0 0 0 1 0 0]
        #  [0 0 0 0 1 1]]
        # calculate re-id increment
        increment = 0
        self.re_id_increment = {}
        for k, v in LANG_ID2SYMBOLS.items():
            self.re_id_increment[k] = increment
            increment += len(v)
        self.n_symbols = increment

        self.register_buffer('phn_mask_table', torch.zeros((len(LANG_ID2SYMBOLS), self.n_symbols)))
        for k, v in LANG_ID2SYMBOLS.items():
            if len(v) > 0:
                self.phn_mask_table[k][self.re_id_increment[k]:self.re_id_increment[k]+len(v)] = 1

    def get_table(self, lang_id):
        return self.tables[f"table-{lang_id}"]

    def get_concat_table(self):
        return torch.cat([self.tables[f"table-{k}"]  
                    for k, v in LANG_ID2SYMBOLS.items() if len(v) > 0], dim=0)
    
    def forward(self, input, lang_ids):
        """
        Input:
            input: Tensor with shape (B, L, dim).
            lang_ids: Language IDs.
        """
        # print("Input shape: ", input.shape)
        if not self.multilingual:  # lang_ids is int
            diff = (input.unsqueeze(2) - self.tables[f"table-{lang_ids}"].unsqueeze(0).unsqueeze(0)) ** 2  # B, L, N, dim
            mse = diff.mean(dim=3)  # B, L, n_symbols
        else:  # lang_ids is a list
            phn_mask = self.phn_mask_table[lang_ids, :].unsqueeze(1)  # B, 1, n_symbols
            concat_table = self.get_concat_table()  # n_symbols, dim
            diff = (input.unsqueeze(2) - concat_table.unsqueeze(0).unsqueeze(0)) ** 2  # B, L, n_symbols, dim
            mse = diff.mean(dim=3)  # B, L, n_symbols
            mse = mse.masked_fill(~(phn_mask.bool()), torch.tensor(float('inf')).to(device=self.device))  # apply mask

        # print("Score shape: ", mse.shape)
        return -mse
