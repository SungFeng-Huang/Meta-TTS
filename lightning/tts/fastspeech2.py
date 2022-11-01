from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning.utilities.cli import instantiate_class

from transformer import PostNet
from lightning.modules import Encoder, Decoder, SpeakerEncoder, VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(pl.LightningModule):
    """ FastSpeech2 """

    # def __init__(self, preprocess_config, model_config, *args, **kwargs):
    def __init__(self,
                 encoder: Encoder,
                 variance_adaptor: VarianceAdaptor,
                 decoder: Decoder,
                 mel_linear: nn.Linear,
                 postnet: PostNet,
                 *args,
                 **kwargs):
        super().__init__()

        self.encoder = encoder
        self.variance_adaptor = variance_adaptor
        self.decoder = decoder
        self.mel_linear = mel_linear
        self.postnet = postnet

    def forward(
        self,
        speaker_args,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        **kwargs,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return {
            "output": output,
            "postnet_output": postnet_output,
            "p_predictions": p_predictions,
            "e_predictions": e_predictions,
            "log_d_predictions": log_d_predictions,
            "d_rounded": d_rounded,
            "src_masks": src_masks,
            "mel_masks": mel_masks,
            "src_lens": src_lens,
            "mel_lens": mel_lens,
        }

