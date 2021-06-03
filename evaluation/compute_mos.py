import os
import json
from argparse import ArgumentParser
from pathlib import Path
import glob
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate

import numpy as np
import scipy
import torch
from torch.utils.data.dataset import Dataset
import librosa

import speechmetrics

import config


class NeuralMOS:
    def __init__(self, args):
        self.corpus = config.corpus
        self.data_dir_dict = config.data_dir_dict  #value: the dir of test_xxxs, ex: result/LibriTTS/670../audio/Testing
        self.n_sample = config.n_sample # number of samples per speaker ex : 16
        self.n_speaker = config.n_speaker # number of speakers ex: 39
        self.mode_list = config.mode_list
        self.step_list = config.step_list
        with open(os.path.join(self.data_dir_dict['recon'], 'test_SQids.json'), 'r+') as F:
            self.sq_list = json.load(F)
        self.speaker_id_map, self.inv_speaker_id_map = self.get_speaker_id_map()
        self.file_list = self.setup_filelist()

    def setup_filelist(self,):
        file_list = {}
        file_list['real'] = self.get_real_filelist()

        recon_dir = os.path.join(self.data_dir_dict['recon'], 'audio/Testing')
        file_list['recon'] = []
        for _id in range(self.n_speaker * self.n_sample):
            candidate = glob.glob(f"{recon_dir}/test_{_id:03}/*.recon.wav")
            assert len(candidate) == 1
            file_list['recon'].append(candidate[0])

        for mode in self.mode_list:
            mode_dir = os.path.join(self.data_dir_dict[mode], 'audio/Testing')
            for step in self.step_list:
                file_list[f'{mode}_step{step}'] = []
                for _id in range(self.n_speaker * self.n_sample):
                    candidate = glob.glob(f"{mode_dir}/test_{_id:03}/*FTstep_{step}.synth.wav")
                    assert len(candidate) == 1
                    file_list[f'{mode}_step{step}'].append(candidate[0])

        return file_list

    def get_real_filelist(self):
        real_filelist = []
        for speaker_id in range(self.n_speaker):
            real_speaker_id = self.speaker_id_map[speaker_id]
            for sample_id in range(self.n_sample):
                data_id = speaker_id * self.n_sample + sample_id
                current_dict = self.sq_list[data_id]
                query_filename = current_dict['qry_id'][0] + '.wav'
                real_filelist.append(
                    os.path.join(
                        self.data_dir_dict['real'], real_speaker_id, query_filename
                    )
                )
        return real_filelist

    def get_speaker_id_map(self):
        speaker_id_map = dict()  # pseudo id to actual id
        inv_speaker_id_map = dict() # actual id to pseudo id
        for speaker_id in range(self.n_speaker):
            search_dict = self.sq_list[speaker_id * self.n_sample]
            real_speaker_id = str(search_dict['qry_id'][0].split('_')[0])
            speaker_id_map[speaker_id] = real_speaker_id
            inv_speaker_id_map[real_speaker_id] = speaker_id
        return speaker_id_map, inv_speaker_id_map

    def compute_mosnet(self):
        mosnet = speechmetrics.load('mosnet', None)
        for mode in self.file_list:
            if os.path.exists(f'csv/{self.corpus}/mosnet_{mode}.csv'):
                continue
            with open(f'csv/{self.corpus}/mosnet_{mode}.csv', 'w') as f:
                f.write('test_id, mos\n')
                scores = []
                for _id, filepath in tenumerate(self.file_list[mode], desc=mode):
                    basename = Path(filepath).name.split('.')[0]
                    realbase = Path(self.file_list['real'][_id]).name.split('.')[0]
                    assert basename == realbase, f"{_id}, {basename}, {realbase}"

                    score = mosnet(filepath)['mosnet'].item()
                    scores.append(score)

                    test_id = f"test_{_id:03}"
                    f.write(f"{test_id}, {score}\n")
                mean, ci = self.get_mean_confidence_interval(scores)
                print(mode, mean, ci)

    def compute_mbnet(self):
        from Pytorch_MBNet.model import MBNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mbnet = MBNet(num_judges = 5000).to(device)
        mbnet.load_state_dict(torch.load('Pytorch_MBNet/pre_trained/model-50000.pt'))
        mbnet.eval()
        for mode in self.file_list:
            if os.path.exists(f'csv/{self.corpus}/mbnet_{mode}.csv'):
                continue
            dataset = MBNetDataset(self.file_list[mode])
            dataloader = torch.utils.data.DataLoader(
                dataset, collate_fn = dataset.collate_fn, batch_size=1, num_workers=1, shuffle = False
            )
            with open(f'csv/{self.corpus}/mbnet_{mode}.csv', 'w') as f:
                f.write('test_id, mos\n')
                scores = []
                for _id, batch in tenumerate(dataloader, desc=mode):
                    wavs = batch
                    wavs = wavs.to(device)
                    wavs = wavs.unsqueeze(1)
                    mean_scores = mbnet.only_mean_inference(spectrum = wavs)
                    score = mean_scores.cpu().tolist()
                    assert len(score) == 1
                    score = score[0]
                    scores.append(score)

                    test_id = f"test_{_id:03}"
                    f.write(f"{test_id}, {score}\n")
                mean, ci = self.get_mean_confidence_interval(scores)
                print(mode, mean, ci)

    def get_mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, h

    def add_up(self, net='mosnet'):
        with open(f'txt/{self.corpus}/{net}.txt', 'w') as fo:
            for mode in self.file_list:
                scores = []
                with open(f'csv/{self.corpus}/{net}_{mode}.csv', 'r') as f:
                    for line in f.readlines()[1:]:
                        test_id, score = line.strip().split(',')
                        score = float(score.strip())
                        scores.append(score)
                mean, ci = self.get_mean_confidence_interval(scores)
                print(mode, mean, ci)
                fo.write(f"{mode}, {mean}, {ci}\n")

class MBNetDataset(Dataset):
    def __init__(self, filelist):
        self.wav_name = filelist
        self.length = len(self.wav_name)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav, _ = librosa.load(self.wav_name[idx], sr=16000)
        wav = np.abs(librosa.stft(wav, n_fft=512)).T
        return wav

    def collate_fn(self, wavs):
        max_len = max(wavs, key = lambda x: x.shape[0]).shape[0]
        output_wavs = []
        for i, wav in enumerate(wavs):
            wav_len = wav.shape[0]
            dup_times = max_len//wav_len
            remain = max_len - wav_len*dup_times
            to_dup = [wav for t in range(dup_times)]
            to_dup.append(wav[:remain, :])
            output_wavs.append(torch.Tensor(np.concatenate(to_dup, axis = 0)))
        output_wavs = torch.stack(output_wavs, dim = 0)
        return output_wavs

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--net', type=str, choices=['mosnet', 'mbnet'], default=False)
    args = parser.parse_args()
    main = NeuralMOS(args)
    if args.net == 'mosnet':
        main.compute_mosnet()
        main.add_up('mosnet')
    if args.net == 'mbnet':
        main.compute_mbnet()
        main.add_up('mbnet')
