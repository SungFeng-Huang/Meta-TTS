import os
import numpy as np
import json
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F

from text.symbols import symbols
from transformer import Encoder, Decoder, PostNet, Constants


class PhonemeEmbedding(pl.LightningModule):

    # def __new__(cls, emb_type, model_config, default_emb=None):
    #     if emb_type == 'mono-lingual' or emb_type == 'multi-lingual':
    #         # If already constructed, use the old one, which means nothing
    #         # happened.
    #         if default_emb is not None:
    #             return default_emb
    #     return object.__new__(cls, emb_type, model_config, default_emb)

    def __init__(self, emb_type, model_config, default_emb=None):
        """
        All the data members would be moved to cuda together with the whole model
        by pytorch_lightning default settings.
        Directly replacing `model.encoder.src_word_emb` with this class would do
        no harm to the `model.encoder` forwarding.
        """
        super().__init__()

        self.emb_type = emb_type
        n_src_vocab = len(symbols) + 1
        d_word_vec = model_config["transformer"]["encoder_hidden"]

        if emb_type == 'mono-lingual' or emb_type == 'multi-lingual':
            self.src_word_emb = nn.Embedding(
                n_src_vocab, d_word_vec, padding_idx=Constants.PAD
            )

        elif emb_type == 'meta-lingual':
            codebook_size = model_config["codebook_size"]
            d_feat = model_config["representation_dim"]
            self.banks = nn.Parameter(torch.randn(codebook_size, d_feat))
            # self.banks = nn.Embedding(
            #     codebook_size, d_feat, padding_idx=Constants.PAD
            # )
            self.proj = nn.Linear(d_feat, d_word_vec)
            self.weighting_matrix = torch.zeros(
                n_src_vocab, codebook_size, requires_grad=False)

    @torch.no_grad()
    def set_quantize_matrix(self, ref):
        """ Compute binary quantize matrix from reference representations.
        Should run before inner_loop update.

        Args:
            ref: Reference representations with size (vocab_size, codebook_size),
                 where vocab_size <= n_src_vocab.
                 Assert the tensor is already moved to cuda by pytorch_lightning.
        """
        # NOTE: make sure padding_idx of self.weighting_matrix is correct
        ref_norm = ref / ref.norm(dim=1, keepdim=True)
        banks_norm = self.banks / self.banks.norm(dim=1, keepdim=True)
        similarity = ref_norm @ banks_norm.T
        # NOTE: why create tensor again, probably not need to transfer to cuda again
        # similarity = dot_product_similarity(
        # torch.tensor(ref, dtype=torch.float32).cuda(), self.banks.T
        # )  # (vocab_size, codebook_size)

        self.weighting_matrix = torch.zeros_like(similarity)
        self.weighting_matrix[torch.arange(
            len(similarity)), similarity.argmax(1)] = 1

    def forward(self, args):
        src_seq = args  # shape: (batch_size, max_len)

        if self.emb_type == 'mono-lingual' or self.emb_type == 'multi-lingual':
            return self.src_word_emb(src_seq)

        elif self.emb_type == 'meta-lingual':
            weighted_embedding = self.weighting_matrix @ self.banks
            output = F.embedding(src_seq, weighted_embedding,
                                 padding_idx=Constants.PAD)
            output = self.proj(output)
            return output
