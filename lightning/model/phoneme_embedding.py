import os
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


N_SRC_VOCAB = len(symbols) + 1


class PhonemeEmbedding(pl.LightningModule):

    def __new__(cls, model_config, algorithm_config):
        emb_type = algorithm_config["adapt"]["phoneme_emb"]["type"]

        # If not using codebook, return nn.Embedding instead.
        if emb_type == 'embedding':
            # NOTE: currently just use multi-lingual vocab_size
            # TODO: mono-lingual
            d_word_vec = model_config["transformer"]["encoder_hidden"]

            return nn.Embedding(
                N_SRC_VOCAB, d_word_vec, padding_idx=Constants.PAD
            )

        return object.__new__(cls)


    def __init__(self, model_config, algorithm_config):
        """
        All the data members would be moved to cuda together with the whole model
        by pytorch_lightning default settings.
        Directly replacing `model.encoder.src_word_emb` with this class would do
        no harm to the `model.encoder` forwarding.
        """
        super().__init__()
        assert algorithm_config["adapt"]["phoneme_emb"]["type"] == 'codebook'

        # NOTE: currently just use multi-lingual vocab_size
        # TODO: mono-lingual
        d_word_vec = model_config["transformer"]["encoder_hidden"]

        self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]["codebook"]
        codebook_size = self.codebook_config["size"]

        self.emb_banks = nn.Embedding(
            codebook_size, d_word_vec, padding_idx=Constants.PAD
        )

        self.share_att_banks = self.codebook_config["attention"]["share_banks"]
        if self.share_att_banks:
            # feats -> proj -> att(att_banks) -> token_id -> emb_banks
            self.att_banks = nn.Embedding(
                codebook_size, d_word_vec, padding_idx=Constants.PAD
            )
            # TPU shared weights are copied independently
            # on the XLA device and this line won't have any effect.
            # However, it works fine for CPU and GPU.
            self.att_banks.weight = self.emb_banks.weight
        else:
            # feats -> att(att_banks) -> token_id -> emb_banks
            d_feat = self.codebook_config["representation_dim"]
            self.att_banks = nn.Embedding(
                codebook_size, d_feat, padding_idx=Constants.PAD
            )

        if self.codebook_config["attention"]["name"] == "cos_sim_argmax":
            self.weighting_matrix = torch.zeros(N_SRC_VOCAB, codebook_size, requires_grad=False)
        elif self.codebook_config["attention"]["name"] == "scaled_dot_product_att":
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))


    def on_post_move_to_device(self):
        if getattr(self, 'share_att_banks', False):
            # Weights shared after the model has been moved to TPU Device
            self.att_banks.weight = self.emb_banks.weight


    def set_quantize_matrix(self, ref, proj_layer=None):
        """ Compute binary quantize matrix from reference representations.
        Should run before inner_loop update.

        Args:
            ref: Reference representations with size (vocab_size, codebook_size),
                 where vocab_size <= N_SRC_VOCAB.
                 Assert the tensor is already moved to cuda by pytorch_lightning.
        """
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)
        if getattr(self, 'share_att_banks', False) and proj_layer is not None:
            ref = proj_layer(ref)

        if self.codebook_config["attention"]["name"] == "cos_sim_argmax":
            # NOTE: make sure padding_idx of self.weighting_matrix is correct
            ref_norm = ref / ref.norm(dim=1, keepdim=True)
            banks_norm = self.att_banks / self.att_banks.norm(dim=1, keepdim=True)
            similarity = ref_norm @ banks_norm.T

            with torch.no_grad():
                self.weighting_matrix.zero_()
                self.weighting_matrix[torch.arange(len(similarity)), similarity.argmax(1)] = 1

        elif self.codebook_config["attention"]["name"] == "scaled_dot_product_att":
            """ Soft attention weight. Better not share_att_banks.
            key: self.att_banks
            value: self.emb_banks
            query: ref
            """


    def forward(self, args):
        src_seq = args  # shape: (batch_size, max_len)

        weighted_embedding = self.weighting_matrix @ self.banks
        output = F.embedding(src_seq, weighted_embedding, padding_idx=Constants.PAD)
        output = self.proj(output)
        return output

