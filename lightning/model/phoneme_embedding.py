import os
import numpy as np
import json
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

import transformer.Constants as Constants
from transformer import Encoder, Decoder, PostNet
from transformer.Modules import ScaledDotProductAttention
from text.symbols import symbols


class PhonemeEmbedding(pl.LightningModule):
    # NOTE:
    #   Tested:
    #       - hard att
    # TODO:
    #   - soft att
    #   - share bank
    #   - embedding mode

    def __init__(self, model_config, algorithm_config):
        """
        All the data members would be moved to cuda together with the whole model
        by pytorch_lightning default settings.
        Directly replacing `model.encoder.src_word_emb` with this class would do
        no harm to the `model.encoder` forwarding.
        """
        super().__init__()
        self.emb_type = algorithm_config["adapt"]["phoneme_emb"]["type"]

        n_src_vocab = len(symbols) + 1
        d_word_vec = model_config["transformer"]["encoder_hidden"]
        self.d_word_vec = d_word_vec

        if self.emb_type == "embedding":
            # TODO
            pass
        
        elif self.emb_type == "codebook":
            self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
            codebook_size = self.codebook_config["size"]

            self.emb_banks = nn.Parameter(torch.randn(codebook_size, d_word_vec))

            att_config = self.codebook_config["attention"]
            d_feat = self.codebook_config["representation_dim"]

            if att_config["type"] == "hard":
                # One-hot similarity

                # feats <-> att_banks -> token_id -> emb_banks
                self.att_banks = nn.Parameter(torch.randn(codebook_size, d_feat))
                # TODO: init from SSL-feature centroids

            elif att_config["type"] == "soft":
                # Attention layer
                #   key: att_banks
                #   value: emb_banks
                #   query: refs

                # att(feats, att_banks) -> token_id weights -> emb_banks
                self.att_banks = nn.Parameter(torch.randn(codebook_size, d_word_vec))
                if att_config["share"]:
                    # TPU shared weights are copied independently
                    # on the XLA device and this line won't have any effect.
                    # However, it works fine for CPU and GPU.
                    self.att_banks.weight = self.emb_banks.weight

                self.w_qs = nn.Linear(d_feat, d_word_vec)
                self.w_ks = nn.Linear(d_word_vec, d_word_vec)
                self.attention = ScaledDotProductAttention(
                    temperature=np.power(d_k, 0.5)
                )


    def on_post_move_to_device(self):
        if (hasattr(self, 'codebook_config')
                and self.codebook_config["attention"]["type"] == "soft"
                and self.codebook_config["attention"]["share"]):
            # Weights shared after the model has been moved to TPU Device
            self.att_banks.weight = self.emb_banks.weight


    def get_new_embedding(self, ref):
        """ Compute binary quantize matrix from reference representations.
        Should run before inner_loop update.

        Args:
            ref: Reference representations with size (vocab_size, codebook_size),
                 where vocab_size <= n_src_vocab.
                 Assert the tensor is already moved to cuda by pytorch_lightning.
        """
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)

        if self.codebook_config["attention"]["type"] == "hard":
            ref_norm = ref.norm(dim=1, keepdim=True)
            ref_mask, _ = torch.nonzero(ref_norm, as_tuple=True)
            normed_ref = ref[ref_mask] / ref_norm[ref_mask]

            bank_norms = self.att_banks.norm(dim=1, keepdim=True)
            normed_banks = self.att_banks / bank_norms

            similarity = normed_ref @ normed_banks.T

            with torch.no_grad():
                weighting_matrix = torch.zeros(
                    ref.shape[0], self.att_banks.shape[0],
                    device=self.device
                )
                # NOTE: I don't know why I can't directly assign = 1
                weighting_matrix[ref_mask, similarity.argmax(1)] = torch.ones_like(ref_mask).float()
                # padding_idx
                weighting_matrix[Constants.PAD].fill_(0)
            weighted_embedding = weighting_matrix @ self.emb_banks
            return weighted_embedding

        elif self.codebook_config["attention"]["type"] == "soft":
            """ Soft attention weight. Better not share_att_banks.
            key: self.att_banks
            value: self.emb_banks
            query: ref
            """
            q = self.w_qs(ref).view(1, -1, 1, self.d_word_vec)
            k = self.w_ks(self.att_banks).view(1, -1, 1, self.d_word_vec)
            q = q.permute(2, 0, 1, 3).contiguous().view(1, -1, d_word_vec)  # 1 x vocab_size x dk
            k = k.permute(2, 0, 1, 3).contiguous().view(1, -1, d_word_vec)  # 1 x codebook_size x dk
            v = self.emb_banks.unsqueeze(0)
            weighted_embedding = self.attention(q, k, v)
            with torch.no_grad():
                weighted_embedding[Constants.PAD].fill_(0)
            return weighted_embedding

