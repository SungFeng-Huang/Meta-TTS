import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformer import Encoder, Decoder, PostNet, Constants
from .modules import VarianceAdaptor
from .speaker_encoder import SpeakerEncoder
from .phoneme_embedding import PhonemeEmbedding
from utils.tools import get_mask_from_lengths


class FastSpeech2(pl.LightningModule):
    """ FastSpeech2 """

    def __init__(self, spk_emb_type, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.speaker_emb = SpeakerEncoder(spk_emb_type, preprocess_config, model_config)

        # Deal with multi-lingual (meta)
        # NOTE: I don't think it's a good choice to use the term "multilingual"
        # to refer to meta-codebook, since there are two conditions:
        #   1. algorithm == "meta": meta-codebook
        #   2. algorithm == "baseline": original nn.Embedding
        # So we have two options:
        #   1. this part should be placed in `System` instead of
        #       `FastSpeech2`.
        #   2. use a general wrapper to deal with it
        # We choose option 2 here:
        self.phn_emb_type = 'mono-lingual'
        if self.model_config.get("multilingual", False):
            self.phn_emb_type = 'meta-lingual'
        self.encoder.src_word_emb = PhonemeEmbedding(
            phn_emb_type, model_config, default_emb=self.encoder.src_word_emb,
        )
        # Then the only thing we need to do is:
        #   1. Add 'encoder' in algorithm_config["adapt"]["modules"]
        #   2. Dataloader stuffs
        #   3. At MetaSystem.on_after_batch_transfer(), run
        #       `self.encoder.set_quantize_matrix()`.

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

        output = self.encoder(texts, src_masks)

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
