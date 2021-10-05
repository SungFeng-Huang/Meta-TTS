import os
import json
from argparse import ArgumentParser
from pathlib import Path
import glob
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import scipy
import torch
from torch.utils.data.dataset import Dataset
import librosa

import speechmetrics

import config


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
            # if mode in ['scratch_encoder', 'encoder', 'dvec'] and step > 0:
            if mode in ['scratch_encoder', 'encoder', 'dvec']:
                continue
            mode_dir = os.path.join(self.data_dir_dict[mode], 'audio/Testing')
            for step in self.step_list:
                file_list[f'{mode}_step{step}'] = []
                for _id in range(self.n_speaker * self.n_sample):
                    try:
                        candidate = glob.glob(f"{mode_dir}/test_{_id:03}/*FTstep_{step}.synth.wav")
                        if self.corpus == "LibriTTS":
                            candidate = [name for name in candidate if name.split('/')[-1][0].isdigit()]
                        assert len(candidate) == 1, mode_dir + ' / ' + ' - '.join(candidates) + f" / test_{_id:03} / {step}"
                    except:
                        candidate = glob.glob(f"{mode_dir}/*/test_{_id:03}/*FTstep_{step}.synth.wav")
                        if self.corpus == "LibriTTS":
                            candidate = [name for name in candidate if name.split('/')[-1][0].isdigit()]
                        assert len(candidate) == 1, mode_dir + ' / ' + ' - '.join(candidates) + f" / test_{_id:03} / {step}"
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


    def bar_plot(self, mode_name):
        if mode_name == 'base_emb':
            xtitle = 'Baseline (emb table)'
        elif mode_name == 'base_emb1':
            xtitle = 'Baseline (share emb)'
        elif mode_name == 'meta_emb':
            xtitle = 'Meta-TTS (emb table)'
        elif mode_name == 'meta_emb1':
            xtitle = 'Meta-TTS (share emb)'
        models = ['mosnet', 'mbnet', 'wav2vec2', 'tera', 'cpc']
        modes = ['real', 'recon'] + [f'{mode_name}{j}_step{i}' for j in ['_vad','_va','_d',''] for i in [0,5,10,20,50,100]]
        xticks = ['Real', 'Reconstructed'] + [f'{j}, step {i}' for j in ['Emb, VA, D','Emb, VA','Emb, D','Emb'] for i in [0,5,10,20,50,100]]
        data = {}
        dfs = []
        for i, mode in tenumerate(modes, desc='mode'):
            for model in tqdm(models, desc='mos_type', leave=False):
                filename = f'csv/{self.corpus}/{model}_{mode}.csv'
                df = pd.read_csv(filename)
                if model in ['mosnet','mbnet']:
                    df = df.rename(columns={' mos': "MOS"})
                else:
                    df = df.rename(columns={'score': "MOS"})
                df['MOS_type'] = model
                df[xtitle] = xticks[i]
                dfs.append(df)
        dfs = pd.concat(dfs, ignore_index=True)
        ax = sns.barplot(x=xtitle, y='MOS', hue='MOS_type', data=dfs)
        ax.grid()
        plt.ylim((1.5,5))
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(12, 3)

        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(*ax.get_legend_handles_labels(),
                        bbox_to_anchor = (1.01, 0.5),
                        loc            = 'center left',
                        borderaxespad  = 0.)

        plt.tight_layout()
        plt.savefig(f'images/{self.corpus}/MOS_{mode_name}.png', format='png', bbox_extra_artists=(leg, ), bbox_inches='tight')
        # plt.show()


    def plot(self, mode_name):
        title_map = {
            'base_emb': 'Baseline (emb table)',
            'base_emb1': 'Baseline (share emb)',
            'meta_emb': 'Meta-TTS (emb table)',
            'meta_emb1': 'Meta-TTS (share emb)',
        }
        xtitle = title_map[mode_name]

        palette = sns.color_palette(n_colors=8)
        palette_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan']
        fig, ax = plt.subplots(figsize=(4.8, 4.2))

        models = ['mosnet', 'mbnet', 'wav2vec2', 'tera', 'cpc']

        # Horizontal lines with bands
        dfs = []
        for mode, xtick in zip(['real', 'recon'], ['Real', 'Reconstructed']):
            for model in tqdm(models, desc='mos_type', leave=False):
                filename = f'csv/{self.corpus}/{model}_{mode}.csv'
                df = pd.read_csv(filename)
                if model in ['mosnet','mbnet']:
                    df = df.rename(columns={' mos': "MOS"})
                else:
                    df = df.rename(columns={'score': "MOS"})
                df['MOS_type'] = model
                df[xtitle] = xtick
                dfs.append(df)
        dfs = pd.concat(dfs, ignore_index=True)
        dfs = dfs.groupby([xtitle, "test_id"]).mean().groupby(xtitle).agg(self.get_mean_confidence_interval)
        print(dfs)
        for (xtick, row), color in zip(dfs.iterrows(), ['purple', 'grey']):
            mean, ci = row["MOS"]
            rgb_color = palette[palette_color.index(color)]
            ax.axhspan(mean-ci, mean+ci, facecolor=rgb_color, alpha=0.15)
            ax.axhline(mean, linestyle='--', alpha=0.5, color=rgb_color, label=xtick)
        del dfs

        # Curves
        dfs = []
        modes = [f'{mode_name}{j}_step{i}' for j in ['_vad','_va','_d',''] for i in [0,5,10,20,50,100]]
        xticks = [f'{j}, step {i}' for j in ['Emb, VA, D','Emb, VA','Emb, D','Emb'] for i in [0,5,10,20,50,100]]
        for i, mode in tenumerate(modes, desc='mode', leave=False):
            for model in tqdm(models, desc='mos_type', leave=False):
                filename = f'csv/{self.corpus}/{model}_{mode}.csv'
                df = pd.read_csv(filename)
                if model in ['mosnet','mbnet']:
                    df = df.rename(columns={' mos': "MOS"})
                else:
                    df = df.rename(columns={'score': "MOS"})
                df['MOS_type'] = model
                df[xtitle] = xticks[i].rsplit(',', 1)[0]
                df['Adaptation Steps'] = int(mode.rsplit('_', 1)[1][4:])
                dfs.append(df)
        dfs = pd.concat(dfs, ignore_index=True)
        print(dfs.groupby([xtitle, "test_id"]).mean().groupby(xtitle).agg(self.get_mean_confidence_interval))
        ax = sns.lineplot(x='Adaptation Steps', y='MOS', hue=xtitle, data=dfs, ax=ax, err_style='bars')
        del dfs

        h, l = ax.get_legend_handles_labels()
        print(l)
        # Usually seaborn treat hue label as the first legend label with empty
        # artist(handle). In such case, we should remove the first handles/labels.
        # But with unknown reason, the hue label is correctly treated as legent
        # title, so we do not need to remove the first handle/label.
        ax.legend(handles=h, labels=l, ncol=2, title=xtitle, title_fontsize='large')
        plt.ylim((2.6,4.2))
        plt.tight_layout()

        savefile = f"images/{self.corpus}/MOS_{mode_name}.png"
        plt.savefig(savefile, format='png')
        print(savefile)
        plt.close()
        from PIL import Image
        im = Image.open(savefile)
        im.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--net', type=str, choices=['mosnet', 'mbnet', 'wav2vec2', 'tera', 'cpc'], default=False)
    parser.add_argument('--plot', type=str, default=False)
    args = parser.parse_args()
    main = NeuralMOS(args)
    if args.net:
        if args.net == 'mosnet':
            main.compute_mosnet()
        elif args.net == 'mbnet':
            main.compute_mbnet()
        main.add_up(args.net)
    # if args.plot:
        # main.plot(args.plot)
    # for suffix in ['', '_base_emb', '_base_emb1', '_meta_emb', '_meta_emb1']:
    for plot in ['base_emb', 'base_emb1', 'meta_emb', 'meta_emb1']:
        main.plot(plot)
    plt.show()
