import numpy as np
from torch.utils.data import Dataset
import random

from dlhlp_lib.utils.numeric import numpy_exist_nan

import Define
from text import text_to_sequence
from projects.cross_lingual.build import build_id2symbols
from src.data.Parsers.parser import DataParser


class FastSpeech2Dataset(Dataset):
    """
    Monolingual, paired dataset for FastSpeech2.
    """
    def __init__(self, filename, data_parser: DataParser, config, spk_refer_wav=False):
        self.data_parser = data_parser
        self.spk_refer_wav = spk_refer_wav
        self.config = config

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.symbol_id = config["symbol_id"]
        self.cleaners = config["text_cleaners"]
        self.id2symbols = build_id2symbols([config])

        self.basename, self.speaker = self.process_meta(filename)

        self.p_noise = 0.0

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        query = {
            "spk": speaker,
            "basename": basename,
        }
        
        duration = self.data_parser.mfa_duration.read_from_query(query)
        mel = self.data_parser.mel.read_from_query(query)
        mel = np.transpose(mel[:, :sum(duration)])
        if self.config["pitch"]["feature"] == "phoneme_level":
            pitch = self.data_parser.mfa_duration_avg_pitch.read_from_query(query)
        else:
            pitch = self.data_parser.interpolate_pitch.read_from_query(query)
            pitch = pitch[:sum(duration)]
        if self.config["energy"]["feature"] == "phoneme_level":
            energy = self.data_parser.mfa_duration_avg_energy.read_from_query(query)
        else:
            energy = self.data_parser.energy.read_from_query(query)
            energy = energy[:sum(duration)]
        phonemes = self.data_parser.phoneme.read_from_query(query)
        phonemes = f"{{{phonemes}}}"
        raw_text = self.data_parser.text.read_from_query(query)

        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        if self.config["pitch"]["normalization"]:
            pitch = (pitch - global_pitch_mu) / global_pitch_std
        if self.config["energy"]["normalization"]:
            energy = (energy - global_energy_mu) / global_energy_std
        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))

        # Experimental
        if self.p_noise > 0:  # add noise to data
            n_symbols = len(self.id2symbols[self.symbol_id])
            for i in range(len(text)):
                if random.random() < self.p_noise:
                    text[i] = random.randint(1, n_symbols) - 1
        
        # Sanity check
        assert not numpy_exist_nan(mel)
        assert not numpy_exist_nan(pitch)
        assert not numpy_exist_nan(energy)
        assert not numpy_exist_nan(duration)
        try:
            assert len(text) == len(duration)
            if self.config["pitch"]["feature"] == "phoneme_level":
                assert len(duration) == len(pitch)
            else:
                assert sum(duration) == len(pitch)
            if self.config["energy"]["feature"] == "phoneme_level":
                assert len(duration) == len(energy)
            else:
                assert sum(duration) == len(pitch)
        except:
            print("Length mismatch: ", query)
            print(len(text), len(phonemes), len(duration), len(pitch), len(energy))
            raise

        sample = {
            "id": basename,
            "speaker": speaker,
            "text": text,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lang_id": self.lang_id,
            "symbol_id": self.symbol_id,
        }

        if self.spk_refer_wav:
            spk_ref_mel_slices = self.data_parser.spk_ref_mel_slices.read_from_query(query)
            sample.update({"spk_ref_mel_slices": spk_ref_mel_slices})

        return sample

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
            return name, speaker


class NoisyFastSpeech2Dataset(FastSpeech2Dataset):
    """
    Experimental corrupted dataset. Test FastSpeech2's robustness with noisy labels.
    """
    def __init__(self, filename, data_parser: DataParser, config, spk_refer_wav=False):
        super().__init__(filename, data_parser, config, spk_refer_wav)
        self.p_noise = 0.1  # Manually change
