import pytorch_lightning as pl
import numpy as np
import torch
import random
from contextlib import contextmanager
from typing import TypedDict, List


class LightningMelGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        vocoder = torch.hub.load(
            "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
        )
        self.mel2wav = vocoder.mel2wav

    def inverse(self, mel):
        with torch.no_grad():
            return self.mel2wav(mel).squeeze(1)

    def infer(self, mels, max_wav_value, lengths=None):
        """preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        """
        wavs = self.inverse(mels / np.log(10))
        wavs = (wavs.cpu().numpy() * max_wav_value).astype("int16")
        wavs = [wav for wav in wavs]

        for i in range(len(mels)):
            if lengths is not None:
                wavs[i] = wavs[i][: lengths[i]]
        return wavs

@contextmanager
def seed_all(seed=None, devices=None):
    rstate = random.getstate()
    nstate = np.random.get_state()
    with torch.random.fork_rng(devices):
        random.seed(seed)
        np.random.seed(seed)
        if seed is None:
            seed = torch.seed()
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        yield
    random.setstate(rstate)
    np.random.set_state(nstate)

class EpisodicInfiniteWrapper:
    def __init__(self, dataset, epoch_length):
        self.dataset = dataset
        self.epoch_length = epoch_length

    def __getitem__(self, idx):
        # new_idx = random.randrange(len(self.dataset))
        # return self.dataset[new_idx]
        return random.choice(self.dataset)

    def __len__(self):
        return self.epoch_length

def loss2str(loss):
    return dict2str(loss2dict(loss))

def loss2dict(loss):
    tblog_dict = {
        "Total Loss"       : loss[0].item(),
        "Mel Loss"         : loss[1].item(),
        "Mel-Postnet Loss" : loss[2].item(),
        "Pitch Loss"       : loss[3].item(),
        "Energy Loss"      : loss[4].item(),
        "Duration Loss"    : loss[5].item(),
    }
    return tblog_dict

def dict2loss(tblog_dict):
    loss = (
        tblog_dict["Total Loss"],
        tblog_dict["Mel Loss"],
        tblog_dict["Mel-Postnet Loss"],
        tblog_dict["Pitch Loss"],
        tblog_dict["Energy Loss"],
        tblog_dict["Duration Loss"],
    )
    return loss

def dict2str(tblog_dict):
    message = ", ".join([f"{k}: {v:.4f}" for k, v in tblog_dict.items()])
    return message


def asr_loss2dict(loss):
    return {"Total Loss": loss.item()}


class MatchingGraphInfo(TypedDict):
    title: str
    x_labels: List[str]
    y_labels: List[str]
    attn: np.array
    quantized: bool
