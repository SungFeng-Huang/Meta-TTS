import os
import json
import torch
import pytorch_lightning as pl

from torch import nn
from resemblyzer import VoiceEncoder


class SpeakerEncoder(pl.LightningModule):
    """ Could be a NN encoder or an embedding. """
    def __init__(self, emb_type, preprocess_config, model_config):
        super().__init__()
        self.emb_type = emb_type

        if emb_type == "table":
            speaker_file = os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")
            n_speaker = len(json.load(open(speaker_file, "r")))
            self.model = nn.Embedding(n_speaker, model_config["transformer"]["encoder_hidden"])
        elif emb_type == "shared":
            self.model = nn.Embedding(1, model_config["transformer"]["encoder_hidden"])
        elif emb_type == "encoder":
            self.model = VoiceEncoder('cpu')
        elif emb_type == "dvec":
            self.model = VoiceEncoder('cpu')
            self.freeze()

    def forward(self, args):
        if self.emb_type == "table":
            speaker = args
            return self.model(speaker)

        elif self.emb_type == "shared":
            speaker = args
            return self.model(torch.zeros_like(speaker))

        elif self.emb_type == "encoder" or self.emb_type == "dvec":
            ref_mels, ref_slices = args
            partial_embeds = self.model(ref_mels)
            speaker_embeds = [partial_embeds[ref_slice].mean(dim=0) for ref_slice in ref_slices]
            speaker_emb = torch.stack([torch.nn.functional.normalize(spk_emb, dim=0) for spk_emb in speaker_embeds])
            return speaker_emb
