import torch
import numpy as np
import os
import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from resemblyzer import VoiceEncoder , preprocess_wav
from pathlib import Path

import config

encoder = VoiceEncoder()

class VisualizeDvector:
    def __init__(self, args):
        self.corpus = config.corpus
        self.tsne_mode_list = config.tsne_mode_list  # ex:['recon', 'baseline_step20', 'meta_step20']
        self.tsne_pseudo_speaker_list = config.tsne_pseudo_speaker_list
        self.tsne_plot_color_list = config.tsne_plot_color_list
        self.tsne_legend_list = config.tsne_legend_list
        self.n_speaker = config.n_speaker
        self.n_sample = config.n_sample
        self.seed = args.seed
        self.output_path = f"images/{self.corpus}/{args.output_path}"
        
        np.random.seed(self.seed)
        with open(os.path.join(config.data_dir_dict['recon'], 'test_SQids.json'), 'r+') as F:
            self.sq_list = json.load(F)
        self.speaker_id_map, self.inv_speaker_id_map = self.get_speaker_id_map()

        # get self.tsne_speaker_list
        self.tsne_speaker_list = []
        for i in self.tsne_pseudo_speaker_list:
            self.tsne_speaker_list.append(self.speaker_id_map[i])

    def load_dvector(self):
        self.dvector_list_dict = dict()
        for mode in self.tsne_mode_list:
            self.dvector_list_dict[mode] = np.load(f'npy/{self.corpus}/{mode}_dvector.npy', allow_pickle=True)

    #get the mapping from pseudo speaker id to actual speaker id in LibriTTS or VCTK
    def get_speaker_id_map(self):
        speaker_id_map = dict()       # pseudo to actual
        inv_speaker_id_map = dict()   # actual to pseudo
        for speaker_id in range(self.n_speaker):
            search_dict = self.sq_list[speaker_id * self.n_sample]
            real_speaker_id = search_dict['qry_id'][0].split('_')[0]
            speaker_id_map[speaker_id] = real_speaker_id
            inv_speaker_id_map[real_speaker_id] = speaker_id
        return speaker_id_map, inv_speaker_id_map

    def tsne(self):
      #performing tsne
      tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
      cat_dvector_list = np.concatenate(
          [self.dvector_list_dict[mode] for mode in self.tsne_mode_list],
          axis=0
      )
      transformed_dvector_list = tsne.fit_transform(cat_dvector_list)

      #split and re-concatenate transformed_dvector_list
      self.trans_dvector_list_dict_all = dict()
      pointer = 0
      for mode in self.tsne_mode_list:
          n_vector = self.dvector_list_dict[mode].shape[0]
          self.trans_dvector_list_dict_all[mode] = transformed_dvector_list[pointer:pointer+n_vector,:]
          pointer = pointer + n_vector


    def get_speaker_dvectors(self):
        #getting tranformed dvector of test_speakers from transformed dvectors

        # getting parts of speaker
        self.trans_dvector_list_dict = dict()
        for mode in self.tsne_mode_list:
            self.trans_dvector_list_dict[mode] = np.concatenate(
                (
                    [self.trans_dvector_list_dict_all[mode][spk_id*self.n_sample:(spk_id+1)*self.n_sample, :]
                     for spk_id in self.tsne_pseudo_speaker_list]
                ),
                axis=0
            )

    def get_speaker_id_list_dict(self):
        self.speaker_id_list_dict = dict()
        for mode in self.tsne_mode_list:
            self.speaker_id_list_dict[mode] = []
            for speaker_id in self.tsne_speaker_list:
                # repeat for seaborn plotting
                self.speaker_id_list_dict[mode] += [f'{speaker_id}']*self.n_sample
            self.speaker_id_list_dict[mode] = np.array(self.speaker_id_list_dict[mode])
                 

    def visualize_dvector(self):
        transformed_dvector_list = np.concatenate(
            [self.trans_dvector_list_dict[mode] for mode in self.tsne_mode_list],
            axis=0
        )
        cat_id_list = np.concatenate(
            [self.speaker_id_list_dict[mode] for mode in self.tsne_mode_list],
            axis=0
        )
        mode_list = np.concatenate(
            [np.array([mode]*len(self.tsne_speaker_list)*self.n_sample) for mode in self.tsne_legend_list],
            axis=0
        )

        #jointly shuffle the list
        joint_list = np.concatenate(
            (
                transformed_dvector_list,
                np.expand_dims(cat_id_list,axis=1),
                np.expand_dims(mode_list,axis=1)
            ),
            axis=1
        )
        np.random.shuffle(joint_list)
        transformed_dvector_list = joint_list[:,:2].astype(float)
        cat_id_list = np.squeeze(joint_list[:,2:3], axis=1)
        mode_list = np.squeeze(joint_list[:,3:], axis=1)

        facecolor_list = np.array(['none' for i in range(cat_id_list.shape[0])])
        # mask out outliers
        mask1 = transformed_dvector_list[:, 0] < 12
        mask2 = transformed_dvector_list[:, 0] > -12
        mask3 = transformed_dvector_list[:, 1] > -12
        mask = np.logical_and.reduce([mask1, mask2, mask3])

        data = {
                "dim-1":transformed_dvector_list[mask, 0],
                "dim-2":transformed_dvector_list[mask, 1],
                "Speaker": cat_id_list[mask],
                "Approach": mode_list[mask]
                }

        fig = plt.figure(figsize=(5,3.5))
        ax = fig.add_subplot(111, aspect='equal')
        edge_color_dict = {f"{speaker_id}":f"C{i}" for i, speaker_id in enumerate(self.tsne_speaker_list)}
        markers = ['$\u20DD$','$\u20E4$','$\u00D7$']
        kws = {"ec": "face", "lw": 0.3}
        palette = sns.color_palette(n_colors=8)
        palette_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan']
        palette = [palette[palette_color.index(c)] for c in self.tsne_plot_color_list]
        axes = sns.scatterplot(
                x="dim-1",
                y="dim-2",
                hue="Approach",
                style="Speaker",
                markers=markers,
                hue_order=self.tsne_legend_list,
                palette = palette,
                data=data,
                legend="full",
                ax=ax,
                **kws,
                )
        handles, labels = axes.get_legend_handles_labels()
        num_of_colors = len(self.tsne_legend_list)+1
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
        plt.savefig(self.output_path, format='png', bbox_extra_artists=(color_leg, sizes_leg), bbox_inches='tight')
        plt.show()
        from PIL import Image
        im = Image.open(self.output_path)
        im.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=531)
    parser.add_argument('--output_path', type=str, default='tsne.png')
    args = parser.parse_args()
    main = VisualizeDvector(args)
    main.load_dvector()
    main.tsne()
    main.get_speaker_dvectors()
    main.get_speaker_id_list_dict()
    main.visualize_dvector()
