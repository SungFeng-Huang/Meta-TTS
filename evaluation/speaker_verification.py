import argparse
import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import random
import json

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from sklearn.metrics import det_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns

import config

class SpeakerVerification:
    def __init__(self, args):
        self.corpus = config.corpus
        self.output_path = f'txt/{self.corpus}/{args.output_path}'
        self.step_list = config.step_list
        self.eer_plot_mode_list = config.eer_plot_mode_list
        self.eer_plot_legend_list = config.eer_plot_legend_list
        self.eer_plot_color_list = config.eer_plot_color_list

    def load_pair_similarity(self):
        self.pair_similarity_dict = np.load(f'npy/{self.corpus}/pair_similarity.npy', allow_pickle=True)[()]

    def get_eer(self):
        self.threshold_dict = dict()
        self.eer_dict = dict()
        for mode, s_list in self.pair_similarity_dict.items():
            print(f'precessing {mode}')
            y_score = s_list.flatten()
            y_true = np.ones_like(s_list)
            y_true[1, :] = 0
            y_true = y_true.flatten()
            fpr, fnr, thresholds = det_curve(y_true, y_score, pos_label=1)

            abs_diffs = np.abs(fpr - fnr)
            min_index = np.argmin(abs_diffs)
            eer = np.mean((fpr[min_index], fnr[min_index]))
            threshold = thresholds[min_index]

            self.threshold_dict[mode] = threshold
            self.eer_dict[mode] = eer
            print(mode)
            print('threshold:', threshold)
            print('eer:', self.eer_dict[mode])
            print()

        with open(self.output_path, 'w+') as F:
            for mode in self.pair_similarity_dict.keys():
                F.write(mode+':\n')
                F.write(f'threshold:{self.threshold_dict[mode]:.4f}\t')
                F.write(f'EER:{self.eer_dict[mode]:.4f}\n')

    def plot_eer(self, suffix=""):
        # 1. preprocessing the data
        self.mu_dict = dict()
        for mode in self.eer_plot_mode_list:
            if mode in ['real', 'recon']:
                self.mu_dict[mode] = np.array([self.eer_dict[mode]*100] * (len(self.step_list)+2))
            else:
                mu_list = []
                for step in self.step_list:
                    mu_list.append(self.eer_dict[f'{mode}_step{step}']*100)
                self.mu_dict[mode] = np.array(mu_list)

        # 2. plot
        fig, ax = plt.subplots(figsize=(4.30, 2.58))
        line_list = [None for i in range(len(self.eer_plot_mode_list))]
        t = np.array(self.step_list)
        for i, mode in enumerate(self.eer_plot_mode_list):
            if mode in ['real', 'recon']:
                # just for correct i
                pass
            else:
                line_list[i] = ax.plot(
                    t, self.mu_dict[mode],
                    label=self.eer_plot_legend_list[i], color=self.eer_plot_color_list[i], alpha=0.5,
                    marker='o'
                )
        
        xmin, xmax = ax.get_xlim()
        t_refer = np.concatenate((np.array([self.step_list[0]-100]), t, np.array([self.step_list[-1]+100])),axis=0)
        for i, mode in enumerate(self.eer_plot_mode_list):
            if mode in ['real', 'recon']:
                line_list[i], = ax.plot(
                    t_refer, self.mu_dict[mode],
                    label=self.eer_plot_legend_list[i], linestyle='--', alpha=0.5, color=self.eer_plot_color_list[i]
                )
            else:
                # just for correct i
                pass

        # 3. set legend
        legend_list = self.eer_plot_mode_list
        lgd = plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        # 4. set axis label
        ax.set_xlabel('Adaptation Steps', fontsize=12)
        ax.set_ylabel('Equal Error Rate (%)', fontsize=12)
        plt.xticks(ticks=self.step_list, fontsize=12)
        plt.yticks(fontsize=12)

        # 5. set axis limit
        plt.xlim((xmin,xmax))

        suffix = ""
        plt.savefig(f"images/{self.corpus}/eer{suffix}.png", format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()
        from PIL import Image
        im = Image.open(f"images/{self.corpus}/eer{suffix}.png")
        im.show()


    def get_det(self):
        fig, ax = plt.subplots(figsize=(6.30, 4.18))
        # fig, ax = plt.subplots(figsize=(5.30, 5.18))
        self.threshold_dict = dict()
        self.eer_dict = dict()
        fprs = []
        fnrs = []
        approaches = []
        ftsteps = []
        for i, mode in enumerate(self.eer_plot_mode_list):
            print(f'precessing {mode}')
            if mode in ['real', 'recon']:
                if mode == 'recon':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real Utterance'
                elif mode == 'real':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real Utterance'
                s_list = self.pair_similarity_dict[mode]
                y_score = s_list.flatten()
                y_true = np.repeat(np.array([1,0]), s_list.shape[1])
                fpr, fnr, thresholds = det_curve(y_true, y_score, pos_label=1)
                fprs.append(fpr*100)
                fnrs.append(fnr*100)
                approaches.append([approach] * len(fpr))
                ftsteps.append([ftstep] * len(fpr))
            else:
                for step in ['5', '100']:
                    approach = self.eer_plot_legend_list[i]
                    ftstep = f"Step {step}"
                    s_list = self.pair_similarity_dict[f"{mode}_step{step}"]
                    y_score = s_list.flatten()
                    y_true = np.repeat(np.array([1,0]), s_list.shape[1])
                    fpr, fnr, thresholds = det_curve(y_true, y_score, pos_label=1)
                    fprs.append(fpr*100)
                    fnrs.append(fnr*100)
                    approaches.append([approach] * len(fpr))
                    ftsteps.append([ftstep] * len(fpr))
        data = {
               "False Positive Rate (%)": np.concatenate(fprs),
               "False Negative Rate (%)": np.concatenate(fnrs),
               "Approach": sum(approaches, []),
               "Adaptation": sum(ftsteps, []),
               }
        palette = sns.color_palette(n_colors=8)
        palette_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan']
        palette = [palette[palette_color.index(c)] for c in self.eer_plot_color_list]
        axes = sns.lineplot(x="False Positive Rate (%)", y="False Negative Rate (%)",
                          hue="Approach", style="Adaptation",
                          data=data, ax=ax, palette=palette)

        ax.grid(linestyle='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0.1, 100])
        ax.set_ylim([0.1, 100])

        handles, labels = axes.get_legend_handles_labels()
        num_of_colors = len(self.eer_plot_color_list)+1
        color_hl = handles[:num_of_colors], labels[:num_of_colors]
        sizes_hl = handles[num_of_colors:], labels[num_of_colors:]
        color_leg = axes.legend(*color_hl,
                        bbox_to_anchor = (1.05, 1),
                        loc            = 'upper left',
                        borderaxespad  = 0.)
        sizes_leg = axes.legend(*sizes_hl,
                        bbox_to_anchor = (1.05, 0),
                        loc            = 'lower left',
                        borderaxespad  = 0.)
        axes.add_artist(color_leg)
        plt.tight_layout()
        plt.savefig(f"images/{self.corpus}/det.png", format='png', bbox_extra_artists=(color_leg, sizes_leg), bbox_inches='tight')
        plt.show()
        from PIL import Image
        im = Image.open(f"images/{self.corpus}/det.png")
        im.show()


    def get_roc(self):
        # fig, ax = plt.subplots(figsize=(5.30, 5.18))
        fig, ax = plt.subplots(figsize=(6.30, 4.18))
        fprs = []
        tprs = []
        approaches = []
        ftsteps = []
        real_score = self.pair_similarity_dict['real'][0]
        for i, mode in enumerate(self.eer_plot_mode_list):
            print(f'precessing {mode}')
            if mode in ['real', 'recon']:
                if mode == 'recon':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real Utterance'
                elif mode == 'real':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real Utterance'
                s_list = self.pair_similarity_dict[mode]
                y_score = np.concatenate([real_score, s_list[0]])
                y_true = np.repeat(np.array([1,0]), s_list.shape[1])
                fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                fprs.append(fpr)
                tprs.append(tpr)
                approaches.append([approach] * len(fpr))
                ftsteps.append([ftstep] * len(fpr))
            else:
                for step in ['5', '100']:
                    approach = self.eer_plot_legend_list[i]
                    ftstep = f"Step {step}"
                    s_list = self.pair_similarity_dict[f"{mode}_step{step}"]
                    y_score = np.concatenate([real_score, s_list[0]])
                    y_true = np.repeat(np.array([1,0]), s_list.shape[1])
                    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
                    roc_auc = auc(fpr, tpr)
                    fprs.append(fpr)
                    tprs.append(tpr)
                    approaches.append([approach] * len(fpr))
                    ftsteps.append([ftstep] * len(fpr))
        data = {
               "False Positive Rate": np.concatenate(fprs),
               "True Positive Rate": np.concatenate(tprs),
               "Approach": sum(approaches, []),
               "Adaptation": sum(ftsteps, []),
               }
        palette = sns.color_palette(n_colors=8)
        palette_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan']
        palette = [palette[palette_color.index(c)] for c in self.eer_plot_color_list]
        axes = sns.lineplot(x="False Positive Rate", y="True Positive Rate",
                     hue="Approach", style="Adaptation",
                     data=data, ax=ax, palette=palette)
        ax.grid(linestyle='--')

        handles, labels = axes.get_legend_handles_labels()
        num_of_colors = len(self.eer_plot_color_list)+1
        color_hl = handles[:num_of_colors], labels[:num_of_colors]
        sizes_hl = handles[num_of_colors:], labels[num_of_colors:]
        color_leg = axes.legend(*color_hl,
                        bbox_to_anchor = (1.05, 1),
                        loc            = 'upper left',
                        borderaxespad  = 0.)
        sizes_leg = axes.legend(*sizes_hl,
                        bbox_to_anchor = (1.05, 0),
                        loc            = 'lower left',
                        borderaxespad  = 0.)
        axes.add_artist(color_leg)
        plt.tight_layout()
        plt.savefig(f"images/{self.corpus}/roc.png", format='png', bbox_extra_artists=(color_leg, sizes_leg), bbox_inches='tight')
        plt.show()
        from PIL import Image
        im = Image.open(f"images/{self.corpus}/roc.png")
        im.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='eer.txt')
    args = parser.parse_args()
    main = SpeakerVerification(args)
    main.load_pair_similarity()
    main.get_eer()
    main.plot_eer()
    # main.plot_eer("_share_emb")
    # main.plot_eer("_emb_table")
    # main.get_det()
    # main.get_roc()
