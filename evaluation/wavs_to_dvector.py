import torch
import torchaudio
import numpy as np
import os
import json
import random
from argparse import ArgumentParser
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate

from resemblyzer import VoiceEncoder , preprocess_wav
from pathlib import Path

import config

encoder = VoiceEncoder()

class WavsToDvector:
    def __init__(self, args):
        self.corpus = config.corpus
        self.data_dir_dict = config.data_dir_dict  #value: the dir of test_xxxs, ex: result/LibriTTS/670../audio/Testing

        self.n_sample = config.n_sample # number of samples per speaker ex : 16
        self.n_speaker = config.n_speaker # number of speakers ex: 39
        self.mode_list = config.mode_list
        self.step_list = config.step_list
        self.mode_step_list = config.mode_step_list
        # self.use_new_pair = args.new_pair
        self.use_new_pair = False

        with open(os.path.join(self.data_dir_dict['recon'], 'test_SQids.json'), 'r+') as F:
            self.sq_list = json.load(F)
        self.speaker_id_map, self.inv_speaker_id_map = self.get_speaker_id_map()

        self.setup_filelist()
        self.dvector_list_dict = self.get_dvector()

    def setup_filelist(self):
        self.enrollment_filelist = self.get_enrollment_filelist()
        self.real_filelist = self.get_real_filelist()
        if self.use_new_pair or not(os.path.exists(f'json/{self.corpus}/pair.json')):
            print('Getting new pair list')
            self.pair_list = self.get_and_save_pair_list()
        else:
            self.pair_list = self.load_pair_list()
    
    def get_dvector(self):
        dvector_list_dict = dict()
        for mode in ['enrollment', 'centroid', 'pair', 'real', 'recon']:
            if os.path.exists(f'npy/{self.corpus}/{mode}_dvector.npy'):
                print(f'Getting dvector of mode: {mode}')
                print(f'\tLoading from: \n\t\tnpy/{self.corpus}/{mode}_dvector.npy')
                dvector_list_dict[mode] = np.load(f'npy/{self.corpus}/{mode}_dvector.npy', allow_pickle=True)
            elif mode == 'centroid':
                dvector_list_dict['centroid'] = self.get_centroid_dvector_list(dvector_list_dict['enrollment'])
            elif mode == 'pair':
                dvector_list_dict['pair'] = self.get_pair_dvector_list()
            elif mode == 'real':
                dvector_list_dict['real'] = self.get_real_dvector_list()
            elif mode == 'recon':
                dvector_list_dict['recon'] = self.get_recon_dvector_list(self.data_dir_dict['recon'])
            elif mode == 'enrollment':
                dvector_list_dict['enrollment'] = self.get_enrollment_dvector_list()
            if not os.path.exists(f'npy/{self.corpus}/{mode}_dvector.npy'):
                print(f'\tSaving to: \n\t\tnpy/{self.corpus}/{mode}_dvector.npy')
                np.save(f'npy/{self.corpus}/{mode}_dvector.npy', dvector_list_dict[mode], allow_pickle=True)

        for mode, steps in self.mode_step_list:
            for step in steps:
                # if mode in ['scratch_encoder', 'encoder', 'dvec'] and step != 0:
                    # continue
                if os.path.exists(f'npy/{self.corpus}/{mode}_step{step}_dvector.npy'):
                    print(f'Getting dvector of mode: {mode}, step: {step}')
                    print(f'\tLoading from: \n\t\tnpy/{self.corpus}/{mode}_step{step}_dvector.npy')
                    dvector_list_dict[f'{mode}_step{step}'] = np.load(
                        f'npy/{self.corpus}/{mode}_step{step}_dvector.npy', allow_pickle=True
                    )
                else:
                    dvector_list_dict[f'{mode}_step{step}'] = self.get_syn_dvector_list(
                        data_dir=self.data_dir_dict[mode], step=step
                    )
                    print(f'\tSaving to: \n\t\tnpy/{self.corpus}/{mode}_step{step}_dvector.npy')
                    np.save(
                        f'npy/{self.corpus}/{mode}_step{step}_dvector.npy',
                        dvector_list_dict[f'{mode}_step{step}'],
                        allow_pickle=True
                    )
        return dvector_list_dict

    #get the mapping from pseudo speaker id to actual speaker id in LibriTTS or VCTK
    def get_speaker_id_map(self):
        speaker_id_map = dict()  # pseudo id to actual id
        inv_speaker_id_map = dict() # actual id to pseudo id
        for speaker_id in range(self.n_speaker):
            search_dict = self.sq_list[speaker_id * self.n_sample]
            real_speaker_id = str(search_dict['qry_id'][0].split('_')[0])
            speaker_id_map[speaker_id] = real_speaker_id
            inv_speaker_id_map[real_speaker_id] = speaker_id
        return speaker_id_map, inv_speaker_id_map

    # get the list of real wav files for each speaker so that we can compute centroid in get_centroid_dvector_list 
    def get_enrollment_filelist(self):
        # getting full file_set for each speaker

        # NOTE: all the real wavs are used as enrollment set,
        # we do not pick out the query samples since some speakers would lead to
        # empty enrollment sets.

        enrollment_fileset_list = [set() for i in range(self.n_speaker)]
        for speaker_id in range(self.n_speaker):
            real_speaker_id = self.speaker_id_map[speaker_id]
            wav_dir = os.path.join(self.data_dir_dict['enrollment'], real_speaker_id)
            for filename in os.listdir(wav_dir):
                if filename.endswith('.wav'):
                    enrollment_fileset_list[speaker_id].add(os.path.join(wav_dir, filename))

        enrollment_filelist = [[] for i in range(self.n_speaker)]
        for i in range(self.n_speaker):
            enrollment_filelist[i] = list(enrollment_fileset_list[i])
        return enrollment_filelist

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

    def get_and_save_pair_list(self):
        pair_list = [dict() for i in range(self.n_speaker * self.n_sample)]
        for speaker_id in range(self.n_speaker):
            for sample_id in range(self.n_sample):
                data_id = speaker_id * self.n_sample + sample_id

                #get positive sample
                source_list = [
                    filename for filename in self.enrollment_filelist[speaker_id]
                    if filename != self.real_filelist[data_id]
                ]
                positive_filelist = random.sample(self.enrollment_filelist[speaker_id], 4)
                for i in range(4):
                    positive_filelist[i] = positive_filelist[i].split('/')[-1]
                pair_list[data_id]['p'] = positive_filelist

                #get negative sample
                speaker_source_list = [i for i in range(self.n_speaker) if i!=speaker_id]
                speaker_sample_list = random.sample(speaker_source_list, 4)
                negative_filelist = []
                for s_id in speaker_sample_list:
                    negative_filelist = negative_filelist + random.sample(self.enrollment_filelist[s_id],1)
                for i in range(4):
                    negative_filelist[i] = negative_filelist[i].split('/')[-1]
                pair_list[data_id]['n'] = negative_filelist
                assert(len(pair_list[data_id]['p']) == 4 and len(pair_list[data_id]['n']) == 4)

        # save
        with open(f'json/{self.corpus}/pair.json', 'w', encoding='utf8') as fp:
            json.dump(pair_list, fp)

        return pair_list

    def load_pair_list(self):
        with open(f'json/{self.corpus}/pair.json', newline='') as fp:
            pair_list = json.load(fp)
        return pair_list
    
    # compute the centroid of each speaker with self.enrollment_filelist
    def get_centroid_dvector_list(self, enrollment_list):
        print(f'Getting dvector of mode: centroid')
        centroid_list = []
        for speaker_id in trange(self.n_speaker, desc='Speaker', leave=False):
            centroid = np.mean(enrollment_list[speaker_id], axis=0)
            centroid = centroid / np.linalg.norm(centroid, 2)
            centroid_list.append(centroid)
        return centroid_list

    def get_pair_dvector_list(self):
        print(f'Getting dvector of mode: pair')
        pair_dvector_list = []
        positive_filelist = []
        negative_filelist = []
        for data_id in trange(self.n_speaker*self.n_sample, desc='Test_id', leave=False):
            positive_filelist = positive_filelist + self.pair_list[data_id]['p']
            negative_filelist = negative_filelist + self.pair_list[data_id]['n']

        def get_filepath_list(filename):
            filepath_list = []
            for f in filename:
                speaker_id = f.split('_')[0]
                filepath = os.path.join(self.data_dir_dict['real'], speaker_id, f)
                filepath_list.append(filepath)
            return filepath_list

        def files_to_dvectors(wavpath):
            # wavpath : list of wav file's path
            dvector_list = []
            for i, wav_file in tenumerate(wavpath, leave=False):
                wav_tensor = preprocess_wav(wav_file)
                dvector = encoder.embed_utterance(wav_tensor)
                dvector_list.append(dvector)
            return dvector_list

        pair_dvector_list.append(files_to_dvectors(get_filepath_list(positive_filelist)))
        pair_dvector_list.append(files_to_dvectors(get_filepath_list(negative_filelist)))

        return pair_dvector_list

    def get_enrollment_dvector_list(self):
        print(f'Getting dvector of mode: enrollment')
        spk_emb_tensor_list = []
        for speaker_id in trange(self.n_speaker, desc='Speaker', leave=False):
            emb_tensor_list = []
            for filename in tqdm(self.enrollment_filelist[speaker_id], desc='Sample', leave=False):
                wav_tensor = preprocess_wav(filename)
                emb_tensor = encoder.embed_utterance(wav_tensor)
                emb_tensor_list.append(emb_tensor)
            spk_emb_tensor_list.append(emb_tensor_list)

        return spk_emb_tensor_list

    def get_real_dvector_list(self):
        print(f'Getting dvector of mode: real')
        emb_tensor_list = []
        for speaker_id in trange(self.n_speaker, desc='Speaker', leave=False):
            for sample_id in trange(self.n_sample, desc='Sample', leave=False):
                data_id = speaker_id * self.n_sample + sample_id
                wav_tensor = preprocess_wav(self.real_filelist[data_id])
                emb_tensor = encoder.embed_utterance(wav_tensor)
                emb_tensor_list.append(emb_tensor)
        assert(len(emb_tensor_list)==self.n_speaker * self.n_sample)

        return emb_tensor_list

    def get_recon_dvector_list(self, data_dir):    # todo: change to full data of target speaker
        ###############
        # output: [<dvector of utterance 0>, <dvector of utterance 1>,....]
        ###############
        print(f'Getting dvector of mode: recon')
        if os.path.exists(os.path.join(data_dir, 'audio/Testing/step_100000')):
            data_dir = os.path.join(data_dir, 'audio/Testing/step_100000')
        else:
            data_dir = os.path.join(data_dir, 'audio/Testing')
        emb_tensor_list = []
        for speaker_id in trange(self.n_speaker, desc='Speaker', leave=False):
            for sample_id in trange(self.n_sample, desc='Sample', leave=False):
                data_id = speaker_id*self.n_sample + sample_id
                wav_dir = os.path.join(data_dir, f'test_{data_id:03d}')
                for wav_file in os.listdir(wav_dir):
                    if wav_file.endswith('recon.wav'):
                        filepath = os.path.join(wav_dir, wav_file)
                        wav_tensor = preprocess_wav(filepath)
                        emb_tensor = encoder.embed_utterance(wav_tensor)
                        emb_tensor_list.append(emb_tensor)
                        break

        return emb_tensor_list

    def get_syn_dvector_list(self, data_dir, step=10):    # todo: change to full data of target speaker
        ###############
        # output: [<dvector of utterance 0>, <dvector of utterance 1>,....]
        ###############
        mode = data_dir.split('/')[-1]
        print(f'Getting dvector of mode: {mode}, step: {step}')
        if os.path.exists(os.path.join(data_dir, 'audio/Testing/step_100000')):
            data_dir = os.path.join(data_dir, 'audio/Testing/step_100000')
        else:
            data_dir = os.path.join(data_dir, 'audio/Testing')
        emb_tensor_list = []
        for speaker_id in trange(self.n_speaker, desc='Speaker', leave=False):
            for sample_id in trange(self.n_sample, desc='Sample', leave=False):
                data_id = speaker_id*self.n_sample + sample_id
                if os.path.exists(os.path.join(data_dir, f'test_{data_id:03d}')):
                    wav_dir = os.path.join(data_dir, f'test_{data_id:03d}')
                    for wav_file in os.listdir(wav_dir):
                        if wav_file.endswith(f'FTstep_{step}.synth.wav'):
                            filepath = os.path.join(wav_dir, wav_file)
                            wav_tensor = preprocess_wav(filepath)
                            emb_tensor = encoder.embed_utterance(wav_tensor)
                            emb_tensor_list.append(emb_tensor)
                            break
                else:
                    assert os.path.exists(os.path.join(data_dir, f'test_{data_id:03d}_0'))
                    for i in range(5):
                        wav_dir = os.path.join(data_dir, f'test_{data_id:03d}_{i}')
                        for wav_file in os.listdir(wav_dir):
                            if wav_file.endswith(f'FTstep_{step}.synth.wav'):
                                filepath = os.path.join(wav_dir, wav_file)
                                wav_tensor = preprocess_wav(filepath)
                                emb_tensor = encoder.embed_utterance(wav_tensor)
                                emb_tensor_list.append(emb_tensor)
                                break

        return emb_tensor_list
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--new_pair', type=bool, default=False)
    args = parser.parse_args()
    main = WavsToDvector(args)
    #main.get_dvector()
