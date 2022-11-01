import torch.nn as nn

from transformer.Layers import FFTBlock
from lightning.modules.utils import get_sinusoid_encoding_table


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 max_seq_len: int,
                 d_hidden: int,
                 n_layers: int,
                 n_head: int,
                 d_inner: int,
                 kernel_size: int,
                 dropout: float,
                 **kwargs):
        """
        Args:
            max_seq_len: model_config["max_seq_len"]
            d_hidden: model_config["transformer"]["decoder_hidden"]
                hidden dimension (pos_enc, model)
            n_layers: model_config["transformer"]["decoder_layer"]
            n_head: model_config["transformer"]["decoder_head"]
            d_inner: model_config["transformer"]["conv_filter_size"]
                FFW filter size in FFTBlock.
            kernel_size: model_config["transformer"]["conv_kernel_size"]
                FFW kernel size in FFTBlock.
            dropout: model_config["transformer"]["decoder_dropout"]
        """
        super().__init__()

        n_position = max_seq_len + 1
        d_word_vec = d_hidden
        d_k = d_v = d_word_vec // n_head
        d_model = d_hidden

        self.max_seq_len = max_seq_len
        self.d_model = d_model

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

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask

