import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformer import Encoder, Decoder, PostNet, Encoder2
from .modules import VarianceAdaptor
from .speaker_encoder import SpeakerEncoder
from .phoneme_embedding import PhonemeEmbedding, HardAttCodebook, SoftAttCodebook, TablePhonemeEmbedding, SoftAttCodebook2
from utils.tools import get_mask_from_lengths


class FastSpeech2(pl.LightningModule):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, algorithm_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.emb_generator = nn.Embedding(1, 1)
        self.encoder = Encoder2(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        # If not using multi-speaker, would return None
        self.speaker_emb = SpeakerEncoder(preprocess_config, model_config, algorithm_config)

        self.phoneme_embedding = PhonemeEmbedding(model_config, algorithm_config)
        print("PhonemeEmbedding", self.phoneme_embedding)

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
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        emb_texts = self.emb_generator(texts)
        output = self.encoder(emb_texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speaker_args).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

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

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speaker_args).unsqueeze(1).expand(
                -1, max(mel_lens), -1
            )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

    def get_new_embedding(self, mode, *args, **kwargs):
        self.emb_generator._parameters['weight'] = \
                    self.phoneme_embedding.get_new_embedding(mode, *args, **kwargs).clone()
