import numpy as np
from typing import Literal, Any

import torch
import torchaudio.transforms as T

from .xvec import XvecDataModule
from ..dataset import XvecOnlineDataset as Dataset
from src.utils.tools import pad_2D, pad_1D


class XvecOnlineDataModule(XvecDataModule):

    dataset_cls = Dataset

    def __init__(self,
                 preprocess_config: dict,
                 train_config: dict,
                 *args,
                 dset = "total",
                 target: Literal["speaker", "region", "accent"] = "speaker"):
        """Initialize XvecDataModule.

        Args:
            preprocess_config:
                A dict at least containing paths required by XvecDataset.
            train_config:
                A dict at least containing batch_size.
            target:
                speaker/region/accent.
        """
        # change self.dataset_cls to onlinedataset
        super().__init__(preprocess_config, train_config, target=target)

        self.mel_spec_transform = T.MelSpectrogram(
            sample_rate=self.preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            n_fft=self.preprocess_config["preprocessing"]["stft"]["filter_length"],
            hop_length=self.preprocess_config["preprocessing"]["stft"]["hop_length"],
            win_length=self.preprocess_config["preprocessing"]["stft"]["win_length"],
            center=True,
            pad_mode="reflect",
            power=1,
            n_mels=self.preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            norm="slaney",
            mel_scale="slaney",
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return super().on_before_batch_transfer(batch, dataloader_idx)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        self.mel_spec_transform.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        wavs = batch["wavs"]
        spec_lens = batch["spec_lens"]

        with torch.no_grad():
            spec = self.mel_spec_transform.spectrogram(wavs)
            mels = self.mel_spec_transform.mel_scale(spec)
            energy = torch.norm(spec, dim=-2)
            log_mels = torch.log(torch.clamp(mels, min=1e-5)).transpose(1, 2)
            for i, l in enumerate(spec_lens):
                log_mels[i, l:] = 0

        batch["mels"] = log_mels
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def dict_collate(self, batch):
        ids = [data["id"] for data in batch]
        speakers = [data["speaker"] for data in batch]
        accents = [data["accent"] for data in batch]
        regions = [data["region"] for data in batch]
        wavs = [data["wav"] for data in batch]

        hop_size = self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        # wav_lens = np.array([len(wav) for wav in wavs])
        spec_lens = np.array([len(wav) // hop_size + 1 for wav in wavs])

        speakers = np.array(speakers)
        accents = np.array(accents)
        regions = np.array(regions)
        wavs = pad_1D(wavs, mode="reflect")

        samples = {
            # targets
            "speaker": torch.from_numpy(speakers).long(),
            "accent": torch.from_numpy(accents).long(),
            "region": torch.from_numpy(regions).long(),
            # others
            # "wav_lens": torch.from_numpy(wav_lens).long(),
            "spec_lens": torch.from_numpy(spec_lens).long(),
        }

        return {
            "id": ids,
            "target": samples[self.target],
            "wavs": torch.from_numpy(wavs).float(),
            # "wav_lens": samples["wav_lens"],
            "spec_lens": samples["spec_lens"],
        }
