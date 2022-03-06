import os

import tgt
import librosa
import numpy as np
import torch
from tqdm import tqdm
# import s3prl.hub as hub

import audio as Audio
from ttt import load_wav2vec2_featurizer


class Preprocessor:
    def __init__(self, config):
        # self.ssl_extractor = getattr(hub, 'wav2vec2_xlsr')().cuda()
        self.ssl_extractor = load_wav2vec2_featurizer('wav2vec2-xls-r-2b', layer=None)
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        if "subsets" in config:
            self.train_set = config["subsets"].get("train", None)
            self.val_set = config["subsets"].get("val", None)
            self.test_set = config["subsets"].get("test", None)

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "xlsr2b-representation")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel-representation")), exist_ok=True)

        print("Extra processing Data ...")

        speakers = {}
        i = 0   # index of total speakers (train + val + test)
        outs = {}
        dsets = []
        for dset in [self.train_set, self.val_set, self.test_set]:
            if dset is None:
                continue
            elif isinstance(dset, list):
                dsets += dset
            elif isinstance(dset, str):
                dsets.append(dset)

        for dset in dsets:
            dset_dir = os.path.join(self.in_dir, dset)
            out = list()
            for speaker in tqdm(os.listdir(dset_dir), desc=dset):
                speakers[speaker] = i
                for wav_name in os.listdir(os.path.join(dset_dir, speaker)):
                    if ".wav" not in wav_name:
                        continue

                    basename = wav_name.split(".")[0]
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                    )
                    if os.path.exists(tg_path):
                        try:
                            self.process_utterance(dset_dir, speaker, basename)
                        except Exception as e:
                            print(e)
                    else:
                        continue
                i += 1

        return outs

    def process_utterance(self, in_dir, speaker, basename):
        wav_path = os.path.join(in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )
        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, ssl_duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # SSL representation
        representation = self.ssl_extractor(wav_path)[24]

        # wav1, _ = librosa.load(wav_path, sr=16000)
        # wav1 = wav1[
        #     int(16000 * start): int(16000 * end)
        # ].astype(np.float32)
        # wav1 = torch.from_numpy(wav1).float()
        # with torch.no_grad():
        #     hidden_states = featurizer(wav_path)
        #     representation = self.ssl_extractor(
        #         [wav1.cuda()])["last_hidden_state"][0]
        # representation = representation.detach().cpu().numpy()
        # hidden_states = featurizer(wav_path)
        # # avg = np.mean(np.stack(hidden_states, axis=0), axis=0)
        # feats_list.append(hidden_states[24])

        # Reload mel-spectrogram
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        if not os.path.isfile(os.path.join(self.out_dir, "mel", mel_filename)):
            print("file not exist!")
            raise ValueError
        # mel_spectrogram = np.load(os.path.join(self.out_dir, "mel", mel_filename))
        # mel_repr = np.zeros(mel_spectrogram.shape)

        # Compute mean mel-spectrogram
        # pos = 0
        # for i, d in enumerate(duration):
        #     if d > 0:
        #         mel_repr[i] = np.mean(mel_spectrogram[pos : pos + d], axis=0)
        #     else:
        #         mel_repr[i] = np.zeros(80)
        #     pos += d
        # mel_repr = mel_repr[: len(duration)]

        # SSL representation Phoneme-level average
        pos = 0
        for i, d in enumerate(ssl_duration):
            if d > 0:
                representation[i] = np.mean(
                    representation[pos: pos + d], axis=0)
            else:
                representation[i] = np.zeros(1920)
            pos += d
        representation = representation[: len(ssl_duration)]

        # Save files
        # mel_repr_filename = "{}-mel-representation-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.out_dir, "mel-representation", mel_repr_filename), mel_repr)
        
        # debug
        # checking = "{}-xlsr53-representation-{}.npy".format(speaker, basename)
        # checking = os.path.join(self.out_dir, "xlsr53-representation", checking)
        # a = np.load(checking)
        # print(a.shape, representation.shape)
        # assert a.shape == representation.shape

        representation_filename = "{}-xlsr2b-representation-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "xlsr2b-representation", representation_filename), representation)

        return None

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations, ssl_durations = [], []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )
            ssl_durations.append(
                int(
                    np.round(e * 50)  # Hubert use 20ms window
                    - np.round(s * 50)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        ssl_durations = ssl_durations[:end_idx]

        return phones, durations, ssl_durations, start_time, end_time
