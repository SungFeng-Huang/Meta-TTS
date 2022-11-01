import json
import torch
import pytorch_lightning as pl

from torch import nn
from resemblyzer import VoiceEncoder
from typing import Any

from .utils import SpkEmbType

class GE2E(VoiceEncoder):
    """ VoiceEncoder from scratch """

    def __init__(self,
                 mel_n_channels: int = 40,
                 model_hidden_size: int = 256,
                 model_embedding_size: int = 256,
                 model_num_layers: int = 3,
                 device: str = None):
        super().__init__()

        # Define the network
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device


class SpeakerEncoder(pl.LightningModule):
    """ Could be a NN encoder or an embedding. """

    def __init__(self,
                 d_hidden: int,
                 emb_type: SpkEmbType,
                 speaker_file: Any = None,
                 **kwargs):
        """
        Args:
            d_hidden: model_config["transformer"]["encoder_hidden"]
                Hidden dimension.
            emb_type: algorithm_config["adapt"]["speaker_emb"]
                Speaker embedding type.
            speaker_file: os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")
                `preprocessed_path/speakers.json`.
        """
        super().__init__()
        self.emb_type = emb_type
        self.d_hidden = d_hidden

        if not SpkEmbType.supported_type(self.emb_type):
            raise ValueError("emb_type not in choises")

        if self.emb_type == SpkEmbType.TABLE:
            # speaker_file = 
            n_speaker = len(json.load(open(speaker_file, "r")))
            self.model = nn.Embedding(n_speaker, d_hidden)
        elif self.emb_type == SpkEmbType.SHARED:
            self.model = nn.Embedding(1, d_hidden)
        elif self.emb_type in (SpkEmbType.ENCODER, SpkEmbType.DVEC):
            self.model = VoiceEncoder('cpu')
            if self.emb_type == SpkEmbType.DVEC:
                self.freeze()
        elif self.emb_type == SpkEmbType.SCRATCH:
            self.model = GE2E('cpu')

    def forward(self, args):
        if self.emb_type == SpkEmbType.TABLE:
            speaker = args
            return self.model(speaker)

        elif self.emb_type == SpkEmbType.SHARED:
            speaker = args
            return self.model(torch.zeros_like(speaker))

        elif self.emb_type in (SpkEmbType.ENCODER, SpkEmbType.DVEC,
                               SpkEmbType.SCRATCH):
            ref_mels, ref_slices = args
            partial_embeds = self.model(ref_mels)
            speaker_embeds = [partial_embeds[ref_slice].mean(dim=0)
                              for ref_slice in ref_slices]
            speaker_emb = torch.stack([nn.functional.normalize(spk_emb, dim=0)
                                       for spk_emb in speaker_embeds])
            return speaker_emb
