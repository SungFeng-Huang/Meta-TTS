import pytorch_lightning as pl
import numpy as np
import torch
import random
from contextlib import contextmanager


class LightningMelGAN(pl.LightningModule):
    def __init__(self, vocoder):
        super().__init__()
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
