import torch.nn as nn

import transformer.Constants as Constants
from transformer.Layers import FFTBlock
from text.symbols import symbols
from lightning.modules.utils import get_sinusoid_encoding_table


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self,
                 max_seq_len: int,
                 d_hidden: int,
                 n_layers: int,
                 n_head: int,
                 d_inner: int,
                 kernel_size: int,
                 dropout: float,
                 n_src_vocab: int = len(symbols) + 1,
                 **kwargs):
        """
        Args:
            max_seq_len: model_config["max_seq_len"].
            d_hidden: model_config["transformer"]["encoder_hidden"].
                hidden dimension (pos_enc, model)
            n_layers: model_config["transformer"]["encoder_layer"].
            n_head: model_config["transformer"]["encoder_head"].
            d_inner: model_config["transformer"]["conv_filter_size"].
                FFW filter size in FFTBlock.
            kernel_size: model_config["transformer"]["conv_kernel_size"].
                FFW kernel size in FFTBlock.
            dropout: model_config["transformer"]["encoder_dropout"]
        """
        super().__init__()

        n_position = max_seq_len + 1
        d_word_vec = d_hidden
        d_k = d_v = d_word_vec // n_head
        d_model = d_hidden

        self.max_seq_len = max_seq_len
        self.d_model = d_hidden

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output

