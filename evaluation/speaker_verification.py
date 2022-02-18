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
            if mode in ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                self.mu_dict[mode] = np.array([self.eer_dict[mode]*100] * (len(self.step_list)+2))
            else:
                mu_list = []
                for step in self.step_list:
                    mu_list.append(self.eer_dict[f'{mode}_step{step}']*100)
                self.mu_dict[mode] = np.array(mu_list)

        # 2. plot
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        # fig, ax = plt.subplots()
        # fig, ax = plt.subplots(figsize=(4.30, 2.58))
        line_list = [None for i in range(len(self.eer_plot_mode_list))]
        t = np.array(self.step_list)
        for i, mode in enumerate(self.eer_plot_mode_list):
            if mode in ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                # just for correct i
                pass
            else:
                line_list[i] = ax.plot(
                    t, self.mu_dict[mode],
                    label=self.eer_plot_legend_list[i], color=self.eer_plot_color_list[i], alpha=0.5,
                    marker='o'
                )
        xmin, xmax = ax.get_xlim()
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        line_list = [None for i in range(len(self.eer_plot_mode_list))]
        t = np.array(self.step_list)
        # xmin, xmax = ax.get_xlim()
        t_refer = np.concatenate((np.array([self.step_list[0]-100]), t, np.array([self.step_list[-1]+100])),axis=0)
        for i, mode in enumerate(self.eer_plot_mode_list):
            if mode in ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                line_list[i], = ax.plot(
                    t_refer, self.mu_dict[mode],
                    label=self.eer_plot_legend_list[i], linestyle='--', alpha=0.5, color=self.eer_plot_color_list[i]
                )
            else:
                # just for correct i
                # pass
                line_list[i] = ax.plot(
                    t, self.mu_dict[mode],
                    label=self.eer_plot_legend_list[i], color=self.eer_plot_color_list[i], alpha=0.5,
                    marker='o'
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
        if suffix == '':
            _hl = handles[0::2]+handles[1::2], labels[0::2]+labels[1::2]
            lgd = plt.legend(*_hl, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, title=title_map[suffix])
        else:
            _hl = handles, labels
            lgd = plt.legend(*_hl, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, title=title_map[suffix])
            lgd.get_title().set_fontsize('large')

        # 4. set axis label
        ax.set_xlabel('Adaptation Steps', fontsize=12)
        ax.set_ylabel('Equal Error Rate (%)', fontsize=12)
        plt.xticks(ticks=self.step_list, fontsize=12)
        plt.yticks(fontsize=12)

        # 5. set axis limit
        plt.xlim((xmin,xmax))
        plt.ylim((0, 52))
        print(ax.get_ylim())
        print(plt.gcf().get_size_inches())

        plt.savefig(f"images/{self.corpus}/eer{suffix}.png", format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        # plt.show()
        plt.close()
        from PIL import Image
        im = Image.open(f"images/{self.corpus}/eer{suffix}.png")
        im.show()


    def get_det(self, suffix=""):
        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        # fig, ax = plt.subplots(figsize=(5.30, 5.18))
        self.threshold_dict = dict()
        self.eer_dict = dict()
        fprs = []
        fnrs = []
        approaches = []
        ftsteps = []
        for i, mode in enumerate(self.eer_plot_mode_list):
            print(f'precessing {mode}')
            if mode in ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                # if mode == 'recon':
                    # approach = self.eer_plot_legend_list[i]
                    # ftstep = 'Real Utterance'
                # elif mode == 'real':
                    # approach = self.eer_plot_legend_list[i]
                    # ftstep = 'Real Utterance'
                if mode == 'recon':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real'
                elif mode == 'real':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real'
                else:
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Encoder'
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

        title_map = {
            '': 'Approach',
            '_base_emb': 'Baseline (emb table)',
            '_base_emb1': 'Baseline (share emb)',
            '_meta_emb': 'Meta-TTS (emb table)',
            '_meta_emb1': 'Meta-TTS (share emb)',
        }
        handles, labels = axes.get_legend_handles_labels()
        num_of_colors = len(self.eer_plot_color_list)+1
        if suffix == '':
            color_hl = handles[1:num_of_colors:2]+handles[2:num_of_colors:2], labels[1:num_of_colors:2]+labels[2:num_of_colors:2]
            sizes_hl = handles[num_of_colors+1:], labels[num_of_colors+1:]
            color_leg = axes.legend(
                *color_hl,
                bbox_to_anchor = (0.5, 1.17),
                loc            = 'lower center',
                borderaxespad  = 0.,
                ncol=2,
                title=labels[0],
            )
            sizes_leg = axes.legend(
                *sizes_hl,
                bbox_to_anchor = (0.5, 1.02),
                loc            = 'lower center',
                borderaxespad  = 0.,
                ncol=4,
                title=labels[num_of_colors],
            )
        else:
            color_hl = handles[1:num_of_colors], labels[1:num_of_colors]
            sizes_hl = handles[num_of_colors+1:], labels[num_of_colors+1:]
            color_leg = axes.legend(
                *color_hl,
                bbox_to_anchor = (0.5, 1.17),
                loc            = 'lower center',
                borderaxespad  = 0.,
                ncol=3,
                title=title_map[suffix],
                title_fontsize='large',
            )
            sizes_leg = axes.legend(
                *sizes_hl,
                bbox_to_anchor = (0.5, 1.02),
                loc            = 'lower center',
                borderaxespad  = 0.,
                ncol=4,
                title=labels[num_of_colors],
            )
        axes.add_artist(color_leg)
        plt.tight_layout()
        plt.savefig(f"images/{self.corpus}/det{suffix}.png", format='png', bbox_extra_artists=(color_leg, sizes_leg), bbox_inches='tight')
        # plt.show()
        print(plt.gcf().get_size_inches())
        plt.close()
        from PIL import Image
        im = Image.open(f"images/{self.corpus}/det{suffix}.png")
        im.show()


    def plot_auc(self, suffix=""):
        self.auc_dict = dict()
        fprs = []
        tprs = []
        approaches = []
        ftsteps = []
        real_score = self.pair_similarity_dict['real'][0]
        for i, mode in enumerate(self.eer_plot_mode_list):
            print(f'precessing {mode}')
            if mode in ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                # if mode == 'recon':
                    # approach = self.eer_plot_legend_list[i]
                    # ftstep = 'Real Utterance'
                # elif mode == 'real':
                    # approach = self.eer_plot_legend_list[i]
                    # ftstep = 'Real Utterance'
                if mode == 'recon':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real'
                elif mode == 'real':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real'
                else:
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Encoder'
                s_list = self.pair_similarity_dict[mode]
                y_score = np.concatenate([real_score, s_list[0]])
                y_true = np.repeat(np.array([1,0]), s_list.shape[1])
                fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                self.auc_dict[mode] = roc_auc
            else:
                for step in self.step_list:
                    approach = self.eer_plot_legend_list[i]
                    ftstep = f"Step {step}"
                    s_list = self.pair_similarity_dict[f"{mode}_step{step}"]
                    y_score = np.concatenate([real_score, s_list[0]])
                    y_true = np.repeat(np.array([1,0]), s_list.shape[1])
                    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
                    roc_auc = auc(fpr, tpr)
                    self.auc_dict[f'{mode}_step{step}'] = roc_auc

        # 1. preprocessing the data
        self.mu_dict = dict()
        for mode in self.eer_plot_mode_list:
            if mode in ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                self.mu_dict[mode] = np.array([self.auc_dict[mode]*100] * (len(self.step_list)+2))
            else:
                mu_list = []
                for step in self.step_list:
                    mu_list.append(self.auc_dict[f'{mode}_step{step}']*100)
                self.mu_dict[mode] = np.array(mu_list)

        # 2. plot
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        # fig, ax = plt.subplots()
        # fig, ax = plt.subplots(figsize=(4.30, 2.58))
        line_list = [None for i in range(len(self.eer_plot_mode_list))]
        t = np.array(self.step_list)
        for i, mode in enumerate(self.eer_plot_mode_list):
            if mode in ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                # just for correct i
                pass
            else:
                line_list[i] = ax.plot(
                    t, self.mu_dict[mode],
                    label=self.eer_plot_legend_list[i], color=self.eer_plot_color_list[i], alpha=0.5,
                    marker='o'
                )
        xmin, xmax = ax.get_xlim()
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        line_list = [None for i in range(len(self.eer_plot_mode_list))]
        t = np.array(self.step_list)
        t_refer = np.concatenate((np.array([self.step_list[0]-100]), t, np.array([self.step_list[-1]+100])),axis=0)
        for i, mode in enumerate(self.eer_plot_mode_list):
            if mode in ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                line_list[i], = ax.plot(
                    t_refer, self.mu_dict[mode],
                    label=self.eer_plot_legend_list[i], linestyle='--', alpha=0.5, color=self.eer_plot_color_list[i]
                )
            else:
                # just for correct i
                # pass
                line_list[i] = ax.plot(
                    t, self.mu_dict[mode],
                    label=self.eer_plot_legend_list[i], color=self.eer_plot_color_list[i], alpha=0.5,
                    marker='o'
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
        if suffix == '':
            _hl = handles[0::2]+handles[1::2], labels[0::2]+labels[1::2]
            lgd = plt.legend(*_hl, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, title=title_map[suffix])
        else:
            _hl = handles, labels
            lgd = plt.legend(*_hl, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, title=title_map[suffix])
            lgd.get_title().set_fontsize('large')

        # 4. set axis label
        ax.set_xlabel('Adaptation Steps', fontsize=12)
        ax.set_ylabel('ROC AUC (%)', fontsize=12)
        plt.xticks(ticks=self.step_list, fontsize=12)
        plt.yticks(fontsize=12)

        # 5. set axis limit
        plt.xlim((xmin,xmax))
        plt.ylim((48, 100))
        print(ax.get_ylim())
        print(plt.gcf().get_size_inches())

        plt.savefig(f"images/{self.corpus}/auc{suffix}.png", format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        # plt.show()
        plt.close()
        from PIL import Image
        im = Image.open(f"images/{self.corpus}/auc{suffix}.png")
        im.show()


    def get_roc(self, suffix=""):
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        # fig, ax = plt.subplots(figsize=(6.30, 4.18))
        # fig, ax = plt.subplots(figsize=(5.30, 5.18))
        fprs = []
        tprs = []
        approaches = []
        ftsteps = []
        real_score = self.pair_similarity_dict['real'][0]
        for i, mode in enumerate(self.eer_plot_mode_list):
            print(f'precessing {mode}')
            if mode in ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0']:
                # if mode == 'recon':
                    # approach = self.eer_plot_legend_list[i]
                    # ftstep = 'Real Utterance'
                # elif mode == 'real':
                    # approach = self.eer_plot_legend_list[i]
                    # ftstep = 'Real Utterance'
                if mode == 'recon':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real'
                elif mode == 'real':
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Real'
                else:
                    approach = self.eer_plot_legend_list[i]
                    ftstep = 'Encoder'
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

        title_map = {
            '': 'Approach',
            '_base_emb': 'Baseline (emb table)',
            '_base_emb1': 'Baseline (share emb)',
            '_meta_emb': 'Meta-TTS (emb table)',
            '_meta_emb1': 'Meta-TTS (share emb)',
        }
        handles, labels = axes.get_legend_handles_labels()
        num_of_colors = len(self.eer_plot_color_list)+1
        if suffix == '':
            color_hl = handles[1:num_of_colors:2]+handles[2:num_of_colors:2], labels[1:num_of_colors:2]+labels[2:num_of_colors:2]
            sizes_hl = handles[num_of_colors+1:], labels[num_of_colors+1:]
            color_leg = axes.legend(
                *color_hl,
                bbox_to_anchor = (0.5, 1.17),
                loc            = 'lower center',
                borderaxespad  = 0.,
                ncol=2,
                title=labels[0],
            )
            sizes_leg = axes.legend(
                *sizes_hl,
                bbox_to_anchor = (0.5, 1.02),
                loc            = 'lower center',
                borderaxespad  = 0.,
                ncol=4,
                title=labels[num_of_colors],
            )
        else:
            color_hl = handles[1:num_of_colors], labels[1:num_of_colors]
            sizes_hl = handles[num_of_colors+1:], labels[num_of_colors+1:]
            color_leg = axes.legend(
                *color_hl,
                bbox_to_anchor = (0.5, 1.17),
                loc            = 'lower center',
                borderaxespad  = 0.,
                ncol=3,
                title=title_map[suffix],
                title_fontsize='large',
            )
            sizes_leg = axes.legend(
                *sizes_hl,
                bbox_to_anchor = (0.5, 1.02),
                loc            = 'lower center',
                borderaxespad  = 0.,
                ncol=4,
                title=labels[num_of_colors],
            )
        axes.add_artist(color_leg)
        plt.tight_layout()
        plt.savefig(f"images/{self.corpus}/roc{suffix}.png", format='png', bbox_extra_artists=(color_leg, sizes_leg), bbox_inches='tight')
        # plt.show()
        plt.close()
        from PIL import Image
        im = Image.open(f"images/{self.corpus}/roc{suffix}.png")
        im.show()

    def set_suffix(self, suffix=""):
        if suffix == "":
            self.eer_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.eer_plot_mode_list = config.eer_plot_mode_list
            self.eer_plot_legend_list = config.eer_plot_legend_list
        elif suffix == '_base_emb':
            self.eer_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.eer_plot_mode_list = ['real', 'recon',  'base_emb_vad', 'base_emb_va', 'base_emb_d', 'base_emb']
            self.eer_plot_legend_list = [
                'Real', 'Reconstructed', 'Emb, VA, D', 'Emb, VA', 'Emb, D', 'Emb'
            ]
                # 'Real', 'Reconstructed', 'Baseline (Emb, VA, D)', 'Baseline (Emb, VA)', 'Baseline (Emb, D)', 'Baseline (Emb)'
        elif suffix == '_base_emb1':
            self.eer_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.eer_plot_mode_list = ['real', 'recon',  'base_emb1_vad', 'base_emb1_va', 'base_emb1_d', 'base_emb1']
            self.eer_plot_legend_list = [
                'Real', 'Reconstructed', 'Emb, VA, D', 'Emb, VA', 'Emb, D', 'Emb'
            ]
                # 'Real', 'Reconstructed', 'Baseline (Emb, VA, D)', 'Baseline (Emb, VA)', 'Baseline (Emb, D)', 'Baseline (Emb)'
        elif suffix == '_meta_emb':
            self.eer_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.eer_plot_mode_list = ['real', 'recon', 'meta_emb_vad', 'meta_emb_va', 'meta_emb_d', 'meta_emb']
            self.eer_plot_legend_list = [
                'Real', 'Reconstructed', 'Emb, VA, D', 'Emb, VA', 'Emb, D', 'Emb'
            ]
                # 'Real', 'Reconstructed', 'Meta-TTS (Emb, VA, D)', 'Meta-TTS (Emb, VA)', 'Meta-TTS (Emb, D)', 'Meta-TTS (Emb)'
        elif suffix == '_meta_emb1':
            self.eer_plot_color_list = ['purple', 'grey', 'orange', 'red', 'green', 'blue']
            self.eer_plot_mode_list = ['real', 'recon', 'meta_emb1_vad', 'meta_emb1_va', 'meta_emb1_d', 'meta_emb1']
            self.eer_plot_legend_list = [
                'Real', 'Reconstructed', 'Emb, VA, D', 'Emb, VA', 'Emb, D', 'Emb'
            ]
                # 'Real', 'Reconstructed', 'Meta-TTS (Emb, VA, D)', 'Meta-TTS (Emb, VA)', 'Meta-TTS (Emb, D)', 'Meta-TTS (Emb)'
        elif suffix == '_encoder':
            # self.eer_plot_color_list = ['purple', 'grey', 'orange', 'red', 'brown', 'green', 'blue']
            # self.eer_plot_mode_list = ['real', 'recon', 'scratch_encoder_step0', 'encoder_step0', 'dvec_step0', 'meta_emb_vad', 'base_emb_vad']
            # self.eer_plot_legend_list = [
                # 'Real', 'Reconstructed', 'Scrach encoder', 'Pre-trained encoder', 'd-vector',
                # 'Meta-TTS (Emb, VA, D)', 'Baseline (Emb, VA, D)'
            # ]
            self.eer_plot_color_list = ['purple', 'grey', 'orange', 'blue', 'red', 'green', 'brown']
            self.eer_plot_mode_list = ['real', 'recon', 'scratch_encoder_step0', 'base_emb_vad', 'encoder_step0', 'meta_emb_vad', 'dvec_step0']
            self.eer_plot_legend_list = [
                'Real', 'Reconstructed', 'Scrach encoder', 'Baseline (Emb, VA, D)', 'Pre-trained encoder',
                'Meta-TTS (Emb, VA, D)', 'd-vector'
            ]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='eer.txt')
    args = parser.parse_args()
    main = SpeakerVerification(args)
    main.load_pair_similarity()
    main.get_eer()
    # for suffix in ['', '_base_emb', '_base_emb1', '_meta_emb', '_meta_emb1']:
    for suffix in ['_base_emb', '_base_emb1', '_meta_emb', '_meta_emb1']:
    # for suffix in [ '_encoder']:
        main.set_suffix(suffix)
        main.plot_eer(suffix)
        main.plot_auc(suffix)
        # main.get_det(suffix)
        # main.get_roc(suffix)
    exit()
