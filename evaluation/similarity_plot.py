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
        self.sim_plot_mode_list = config.sim_plot_mode_list
        self.plot_type = config.plot_type
        assert(self.plot_type in ['errorbar', 'box_ver', 'box_hor'])
        self.step_list = config.step_list
        self.sim_plot_color_list = config.sim_plot_color_list
        self.sim_plot_legend_list = config.sim_plot_legend_list
        self.output_path = f"images/{self.corpus}/{args.output_path}"
        
    def load_centroid_similarity(self):
        self.similarity_list_dict = np.load(f'npy/{self.corpus}/centroid_similarity_dict.npy', allow_pickle=True)[()]

    def sim_plot(self):
        if self.plot_type == 'errorbar':
            self.errorbar_plot()
        elif self.plot_type == 'box_ver':
            self.box_ver_plot()
        elif self.plot_type == 'box_hor':
            self.box_hor_plot()


    def box_ver_plot(self):
        pass
    def box_hor_plot(self):
        pass

    def errorbar_plot(self):
        # 1. preprocessing the data
        t = np.array(self.step_list)

        self.mu_dict = dict()
        for mode in self.sim_plot_mode_list:
            if mode in ['recon', 'recon_random']:
                self.mu_dict[mode] = np.array([np.mean(self.similarity_list_dict[mode])]*(len(self.step_list)+2))
            else:
                mu_list = []
                for step in self.step_list:
                    mu_list.append(np.mean(self.similarity_list_dict[f'{mode}_step{step}']))
                self.mu_dict[mode] = np.array(mu_list)


        self.sigma_dict = dict()
        for mode in self.sim_plot_mode_list:
            if mode in ['recon', 'recon_random']:
                self.sigma_dict[mode] = np.array([np.std(self.similarity_list_dict[mode])]*(len(self.step_list)+2))
            else:
                sigma_list = []
                for step in self.step_list:
                    sigma_list.append(np.std(self.similarity_list_dict[f'{mode}_step{step}']))
                self.sigma_dict[mode] = np.array(sigma_list)


        # 2. plot
        # fig, ax = plt.subplots(1)
        fig, ax = plt.subplots(figsize=(4.30, 2.58))
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
        lgd = plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        # 4. set axis label
        ax.set_xlabel('Adaptation Steps', fontsize=12)
        ax.set_ylabel('Similarity', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 5. set axis limit
        plt.xlim((xmin,xmax))

        plt.savefig(self.output_path, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='errorbar_plot.png')
    args = parser.parse_args()
    main = SimilarityPlot(args)
    main.load_centroid_similarity()
    main.sim_plot()

