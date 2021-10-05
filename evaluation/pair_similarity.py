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
from tqdm import tqdm

import config

class PairSimilarity:
    def __init__(self):
        self.corpus = config.corpus
        # self.pair_sim_mode_list = config.pair_sim_mode_list
        self.mode_list = config.mode_list
        self.step_list = config.step_list

    def load_dvector(self):
        self.dvector_list_dict = dict()
        for mode in ['recon', 'real', 'pair']:
            self.dvector_list_dict[mode] = np.load(f'npy/{self.corpus}/{mode}_dvector.npy', allow_pickle=True)
        for mode in tqdm(self.mode_list, desc='mode'):
            for step in tqdm(self.step_list, leave=False):
                if mode in ['scratch_encoder', 'encoder', 'dvec'] and step != 0:
                    continue
                self.dvector_list_dict[f'{mode}_step{step}'] = np.load(
                    f'npy/{self.corpus}/{mode}_step{step}_dvector.npy', allow_pickle=True
                )

    def get_pair_similarity(self):
        self.pair_similarity_dict = dict()
        for mode in self.dvector_list_dict.keys():
            if mode == 'pair':
                continue
            self.pair_similarity_dict[mode] = self.compute_pair_similarity(self.dvector_list_dict[mode])


    def compute_pair_similarity(self, check_list):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        dvector_test_repeat_tensor = torch.from_numpy(np.repeat(check_list, 4, axis=0))
        pair_dvector_list_positive = torch.from_numpy(self.dvector_list_dict['pair'][0,:])
        pair_dvector_list_negative = torch.from_numpy(self.dvector_list_dict['pair'][1,:])

        with torch.no_grad():
            pair_similarity_list_positive = cos(dvector_test_repeat_tensor, pair_dvector_list_positive).detach().cpu().numpy()
            pair_similarity_list_negative = cos(dvector_test_repeat_tensor, pair_dvector_list_negative).detach().cpu().numpy()
        pos_exp = np.expand_dims(pair_similarity_list_positive, axis=0)
        neg_exp = np.expand_dims(pair_similarity_list_negative, axis=0)
        pair_similarity_list =  np.concatenate((pos_exp, neg_exp), axis=0)

        return pair_similarity_list # [2, num_test_samples]

    def save_pair_similarity(self):
        np.save(f'npy/{self.corpus}/pair_similarity.npy', self.pair_similarity_dict, allow_pickle=True)

    def load_pair_similarity(self):
        self.pair_similarity_dict = np.load(f'npy/{self.corpus}/pair_similarity.npy', allow_pickle=True)[()]

if __name__ == '__main__':
    main = PairSimilarity()
    main.load_dvector()
    main.get_pair_similarity()
    main.save_pair_similarity()
    #main.load_pair_similarity()
