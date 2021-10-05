import torch
import torchaudio
import argparse
import os
import torch.nn as nn
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from argparse import ArgumentParser
import json
import random

import matplotlib.pyplot as plt
from pylab import plot, show, savefig, xlim, figure, \
                ylim, legend, boxplot, setp, axes

import config

class SimilarityPlot:
    def __init__(self, args):
        self.corpus = config.corpus
        self.plot_type = config.plot_type
        assert(self.plot_type in ['errorbar', 'box_ver', 'box_hor'])
        self.step_list = config.step_list
        self.sim_plot_mode_list = config.sim_plot_mode_list
        self.sim_plot_color_list = config.sim_plot_color_list
        self.sim_plot_legend_list = config.sim_plot_legend_list
        self.output_path = f"images/{self.corpus}/{args.output_path}"
        
    def load_centroid_similarity(self):
        self.similarity_list_dict = np.load(f'npy/{self.corpus}/centroid_similarity_dict.npy', allow_pickle=True)[()]

    def sim_plot(self, suffix=''):
        if self.plot_type == 'errorbar':
            # self.seaborn_plot(suffix)
            self.errorbar_plot(suffix)
        elif self.plot_type == 'box_ver':
            self.box_ver_plot(suffix)
        elif self.plot_type == 'box_hor':
            self.box_hor_plot(suffix)


    def box_ver_plot(self, suffix=''):
        pass
    def box_hor_plot(self, suffix=''):
        pass


    def seaborn_plot(self, suffix=""):
        import pandas as pd
        import seaborn as sns
        
        dfs = []
        regions = {}
        for mode in self.sim_plot_mode_list:
            if mode in ['recon', 'recon_random', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                regions[mode] = self.similarity_list_dict[mode]
            else:
                for step in self.step_list:
                    df = pd.DataFrame(self.similarity_list_dict[f'{mode}_step{step}'], columns=['Similarity'])
                    df['Approach'] = mode
                    df['Adaptation Steps'] = step
                    dfs.append(df)
                    # dataset[f'{mode}_{step}'] = self.similarity_list_dict[f'{mode}_step{step}']
        dfs = pd.concat(dfs)

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax = sns.lineplot(x='Adaptation Steps', y='Similarity', hue='Approach', data=dfs, ax=ax)
        plt.show()
        return
        exit()
        xmin, xmax = ax.get_xlim()

        # 1. preprocessing the data
        t = np.array(self.step_list)

        print(self.sim_plot_mode_list)
        self.mu_dict = dict()
        for mode in self.sim_plot_mode_list:
            if mode in ['recon', 'recon_random']:
                # self.mu_dict[mode] = np.array([np.mean(self.similarity_list_dict[mode])]*(len(self.step_list)+2))
                continue
            elif mode in ['scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                # self.mu_dict[mode] = np.array([np.mean(self.similarity_list_dict[mode])]*(len(self.step_list)))
                continue
            else:
                mu_list = []
                for step in self.step_list:
                    mu_list.append(np.mean(self.similarity_list_dict[f'{mode}_step{step}']))
                self.mu_dict[mode] = np.array(mu_list)


        self.sigma_dict = dict()
        for mode in self.sim_plot_mode_list:
            if mode in ['recon', 'recon_random']:
                self.sigma_dict[mode] = np.array([np.std(self.similarity_list_dict[mode])]*(len(self.step_list)+2))
            elif mode in ['scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                self.sigma_dict[mode] = np.array([np.std(self.similarity_list_dict[mode])]*(len(self.step_list)))
            else:
                sigma_list = []
                for step in self.step_list:
                    sigma_list.append(np.std(self.similarity_list_dict[f'{mode}_step{step}']))
                self.sigma_dict[mode] = np.array(sigma_list)
        print(self.mu_dict)
        print(self.sigma_dict)
        # exit()


        # 2. plot
        # fig, ax = plt.subplots(1)
        # fig, ax = plt.subplots(figsize=(4.30, 2.58))
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        line_list = [None for i in range(len(self.sim_plot_mode_list))]
        for i, mode in enumerate(self.sim_plot_mode_list):
            if mode in ['recon', 'recon_random']:
                pass
            else:
                line_list[i] = ax.errorbar(
                    t, self.mu_dict[mode],
                    yerr=self.sigma_dict[mode], marker='o', lw=1, capsize=2,
                    label=self.sim_plot_legend_list[i], color=self.sim_plot_color_list[i], alpha=0.5
                )
        
        xmin, xmax = ax.get_xlim()

        t_refer = np.concatenate((np.array([self.step_list[0]-100]), t, np.array([self.step_list[-1]+100])),axis=0)
        for i,(mode,color) in enumerate(zip(self.sim_plot_mode_list, self.sim_plot_color_list)):
            if mode not in ['recon', 'recon_random']:
                pass
            else:
                line_list[i], = ax.plot(
                    t_refer, self.mu_dict[mode],
                    label=self.sim_plot_legend_list[i], linestyle='--', alpha=0.5, color=self.sim_plot_color_list[i]
                )
                ax.fill_between(
                    t_refer, self.mu_dict[mode]+self.sigma_dict[mode], self.mu_dict[mode]-self.sigma_dict[mode],
                    facecolor=self.sim_plot_color_list[i], alpha=0.15
                )
        
        # 3. set legend
        handles, labels = ax.get_legend_handles_labels()
        _hl = handles[0::2]+handles[1::2], labels[0::2]+labels[1::2]
        # lgd = plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        lgd = plt.legend(*_hl, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, title="Approach")
        # 4. set axis label
        ax.set_xlabel('Adaptation Steps', fontsize=12)
        ax.set_ylabel('Similarity', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 5. set axis limit
        plt.xlim((xmin,xmax))

        plt.savefig(f"images/{self.corpus}/errorbar_plot{suffix}.png", format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        print(f"images/{self.corpus}/errorbar_plot{suffix}.png")
        # plt.show()
        plt.close()
        from PIL import Image
        im = Image.open(f"images/{self.corpus}/errorbar_plot{suffix}.png")
        im.show()


    def errorbar_plot(self, suffix=""):
        # 1. preprocessing the data
        t = np.array(self.step_list)

        print(self.sim_plot_mode_list)
        self.mu_dict = dict()
        for mode in self.sim_plot_mode_list:
            if mode in ['recon', 'recon_random']:
                self.mu_dict[mode] = np.array([np.mean(self.similarity_list_dict[mode])]*(len(self.step_list)+2))
            elif mode in ['scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                self.mu_dict[mode] = np.array([np.mean(self.similarity_list_dict[mode])]*(len(self.step_list)))
            else:
                mu_list = []
                for step in self.step_list:
                    mu_list.append(np.mean(self.similarity_list_dict[f'{mode}_step{step}']))
                self.mu_dict[mode] = np.array(mu_list)


        self.sigma_dict = dict()
        for mode in self.sim_plot_mode_list:
            if mode in ['recon', 'recon_random']:
                self.sigma_dict[mode] = np.array([np.std(self.similarity_list_dict[mode])]*(len(self.step_list)+2))
            elif mode in ['scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                self.sigma_dict[mode] = np.array([np.std(self.similarity_list_dict[mode])]*(len(self.step_list)))
            else:
                sigma_list = []
                for step in self.step_list:
                    sigma_list.append(np.std(self.similarity_list_dict[f'{mode}_step{step}']))
                self.sigma_dict[mode] = np.array(sigma_list)
        print(self.mu_dict)
        print(self.sigma_dict)
        # exit()


        # 2. plot
        # fig, ax = plt.subplots(1)
        # fig, ax = plt.subplots(figsize=(4.30, 2.58))
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        line_list = [None for i in range(len(self.sim_plot_mode_list))]
        for i, mode in enumerate(self.sim_plot_mode_list):
            if mode in ['recon', 'recon_random']:
                pass
            else:
                line_list[i] = ax.errorbar(
                    t, self.mu_dict[mode],
                    yerr=self.sigma_dict[mode], marker='o', lw=1, capsize=2,
                    label=self.sim_plot_legend_list[i], color=self.sim_plot_color_list[i], alpha=0.5
                )
        
        xmin, xmax = ax.get_xlim()

        t_refer = np.concatenate((np.array([self.step_list[0]-100]), t, np.array([self.step_list[-1]+100])),axis=0)
        for i,(mode,color) in enumerate(zip(self.sim_plot_mode_list, self.sim_plot_color_list)):
            if mode not in ['recon', 'recon_random']:
                pass
            else:
                line_list[i], = ax.plot(
                    t_refer, self.mu_dict[mode],
                    label=self.sim_plot_legend_list[i], linestyle='--', alpha=0.5, color=self.sim_plot_color_list[i]
                )
                ax.fill_between(
                    t_refer, self.mu_dict[mode]+self.sigma_dict[mode], self.mu_dict[mode]-self.sigma_dict[mode],
                    facecolor=self.sim_plot_color_list[i], alpha=0.15
                )
        
        # 3. set legend
        title_map = {
            '': 'Approach',
            '_base_emb': 'Baseline (emb table)',
            '_base_emb1': 'Baseline (share emb)',
            '_meta_emb': 'Meta-TTS (emb table)',
            '_meta_emb1': 'Meta-TTS (share emb)',
        }
        handles, labels = ax.get_legend_handles_labels()
        print(labels)
        if suffix == '':
            _hl = handles[0::2]+handles[1::2], labels[0::2]+labels[1::2]
            lgd = plt.legend(*_hl, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, title=title_map[suffix])
        else:
            _hl = handles, labels
            lgd = plt.legend(*_hl, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, title=title_map[suffix])
            lgd.get_title().set_fontsize('large')

        # 4. set axis label
        ax.set_xlabel('Adaptation Steps', fontsize=12)
        ax.set_ylabel('Similarity', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 5. set axis limit
        plt.xlim((xmin,xmax))

        savefile = f"images/{self.corpus}/errorbar_plot{suffix}.png"
        plt.savefig(savefile, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        print(savefile)
        # plt.show()
        plt.close()
        from PIL import Image
        im = Image.open(savefile)
        im.show()

    def set_suffix(self, suffix=""):
        if suffix == "":
            self.sim_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.sim_plot_mode_list = config.sim_plot_mode_list
            self.sim_plot_legend_list = config.sim_plot_legend_list
        elif suffix == '_base_emb':
            self.sim_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.sim_plot_mode_list = ['recon', 'recon_random',  'base_emb_vad', 'base_emb_va', 'base_emb_d', 'base_emb']
            self.sim_plot_legend_list = [
                'Same spk', 'Different spk',
                'Emb, VA, D', 'Emb, VA', 'Emb, D', 'Emb'
            ]
                # 'Baseline (Emb, VA, D)', 'Baseline (Emb, VA)', 'Baseline (Emb, D)', 'Baseline (Emb)'
        elif suffix == '_base_emb1':
            self.sim_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.sim_plot_mode_list = ['recon', 'recon_random',  'base_emb1_vad', 'base_emb1_va', 'base_emb1_d', 'base_emb1']
            self.sim_plot_legend_list = [
                'Same spk', 'Different spk',
                'Emb, VA, D', 'Emb, VA', 'Emb, D', 'Emb'
            ]
                # 'Baseline (Emb, VA, D)', 'Baseline (Emb, VA)', 'Baseline (Emb, D)', 'Baseline (Emb)'
        elif suffix == '_meta_emb':
            self.sim_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.sim_plot_mode_list = ['recon', 'recon_random', 'meta_emb_vad', 'meta_emb_va', 'meta_emb_d', 'meta_emb']
            self.sim_plot_legend_list = [
                'Same spk', 'Different spk',
                'Emb, VA, D', 'Emb, VA', 'Emb, D', 'Emb'
            ]
                # 'Meta-TTS (Emb, VA, D)', 'Meta-TTS (Emb, VA)', 'Meta-TTS (Emb, D)', 'Meta-TTS (Emb)'
        elif suffix == '_meta_emb1':
            self.sim_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.sim_plot_mode_list = ['recon', 'recon_random', 'meta_emb1_vad', 'meta_emb1_va', 'meta_emb1_d', 'meta_emb1']
            self.sim_plot_legend_list = [
                'Same spk', 'Different spk',
                'Emb, VA, D', 'Emb, VA', 'Emb, D', 'Emb'
            ]
                # 'Meta-TTS (Emb, VA, D)', 'Meta-TTS (Emb, VA)', 'Meta-TTS (Emb, D)', 'Meta-TTS (Emb)'
        elif suffix == '_encoder':
            self.sim_plot_color_list = ['purple', 'grey', 'orange', 'red', 'brown', 'green', 'blue']
            self.sim_plot_mode_list = ['recon', 'recon_random', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0', 'meta_emb_vad', 'base_emb_vad']
            self.sim_plot_legend_list = [
                'Same spk', 'Different spk', 'Scrach encoder', 'Pre-trained encoder', 'd-vector',
                'Meta-TTS (Emb, VA, D)', 'Baseline (Emb, VA, D)'
            ]
            self.sim_plot_color_list = ['purple', 'grey', 'orange', 'blue', 'red', 'green', 'brown']
            self.sim_plot_mode_list = ['recon', 'recon_random', 'scratch_encoder_step0', 'base_emb_vad', 'encoder_step0', 'meta_emb_vad', 'dvec_step0']
            self.sim_plot_legend_list = [
                'Same spk', 'Different spk', 'Scrach encoder', 'Baseline (Emb, VA, D)', 'Pre-trained encoder',
                'Meta-TTS (Emb, VA, D)', 'd-vector'
            ]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='errorbar_plot.png')
    args = parser.parse_args()
    main = SimilarityPlot(args)
    main.load_centroid_similarity()
    for suffix in ['', '_base_emb', '_base_emb1', '_meta_emb', '_meta_emb1']:
    # for suffix in ['_encoder']:
        main.set_suffix(suffix)
        main.sim_plot(suffix)

