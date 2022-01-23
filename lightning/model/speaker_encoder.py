import os
import json
import torch
import pytorch_lightning as pl

from torch import nn
from resemblyzer import VoiceEncoder


mel_n_channels = 40
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3
class GE2E(VoiceEncoder):
    """ VoiceEncoder from scratch """

    def __init__(self, device=None):
        super(VoiceEncoder, self).__init__()

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

    def __new__(cls, *args, **kwargs):
        # If not using multi-speaker, do not construct speaker encoder
        if len(args) == 3:
            preprocess_config, model_config, algorithm_config = args
            if not model_config["multi_speaker"]:
                return None
        return super().__new__(cls)

    def __init__(self, preprocess_config, model_config, algorithm_config):
        super().__init__()
        self.emb_type = algorithm_config["adapt"]["speaker_emb"]

        if self.emb_type == "table":
            speaker_file = os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")
            n_speaker = len(json.load(open(speaker_file, "r")))
            self.model = nn.Embedding(n_speaker, model_config["transformer"]["encoder_hidden"])
        elif self.emb_type == "shared":
            self.model = nn.Embedding(1, model_config["transformer"]["encoder_hidden"])
        elif self.emb_type == "encoder":
            self.model = VoiceEncoder('cpu')
        elif self.emb_type == "dvec":
            self.model = VoiceEncoder('cpu')
            self.freeze()
        elif self.emb_type == "scratch_encoder":
            self.model = GE2E('cpu')

    def forward(self, args):
        if self.emb_type == "table":
            speaker = args
            return self.model(speaker)

        elif self.emb_type == "shared":
            speaker = args
            return self.model(torch.zeros_like(speaker))

        elif self.emb_type == "encoder" or self.emb_type == "dvec" or self.emb_type == "scratch_encoder":
            ref_mels, ref_slices = args
            partial_embeds = self.model(ref_mels)
            if torch.isnan(partial_embeds).any():
                print("partial_embeds NaN!")
                print(partial_embeds.shape)
                print("Num of NaN: ", torch.isnan(partial_embeds).sum().item())

            speaker_embeds = [partial_embeds[ref_slice].mean(dim=0) for ref_slice in ref_slices]
            for ref_slice, spk_emb in zip(ref_slices, speaker_embeds):
                if torch.isnan(spk_emb).any():
                    print("0??")
                    print(partial_embeds[ref_slice].shape)
                    rs = partial_embeds[ref_slice].mean(dim=0)
                    if torch.isnan(rs).any():
                        print("ref slice NaN!")
                        print(rs.shape)
                        print("Num of NaN: ", torch.isnan(rs).sum().item())
                    print("speaker_embeds NaN!")
                    print(spk_emb.shape)
                    print("Num of NaN: ", torch.isnan(spk_emb).sum().item())            
            speaker_emb = torch.stack([torch.nn.functional.normalize(spk_emb, dim=0) for spk_emb in speaker_embeds])
            if torch.isnan(speaker_emb).any():
                print("spk normalize NaN!")
                print(speaker_emb.shape)
                print("Num of NaN: ", torch.isnan(speaker_emb).sum().item())
            
            return speaker_emb
