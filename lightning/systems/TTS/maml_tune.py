#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from lightning.model.loss import PhonemeClassificationLoss

from ..utils import CodebookAnalyzer, generate_hubert_features
from utils.tools import get_mask_from_lengths
from ..system2 import System
from lightning.systems.utils import Task
from lightning.utils import loss2dict, LightningMelGAN, MatchingGraphInfo
from lightning.model.phoneme_embedding import PhonemeEmbedding
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks.baseline_saver import Saver
from lightning.model.asr_model import ASRCenterHead, Codebook, SoftBank
from lightning.callbacks.utils import synth_samples, recon_samples
from text.define import LANG_ID2SYMBOLS, LANG_ID2NAME


STATSDICT = {
    0: "./preprocessed_data/miniLibriTTS",
    1: "./preprocessed_data/miniAISHELL-3",
    2: "./preprocessed_data/miniGlobalPhone-fr",
    3: "./preprocessed_data/miniGlobalPhone-de",
    4: "",
    5: "./preprocessed_data/miniGlobalPhone-es",
    6: "./preprocessed_data/miniJVS",
    7: "./preprocessed_data/miniGlobalPhone-cz",
}


class MetaTuneSystem(System):
    """ 
    Tune version of maml system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_codebook_type()
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)

    def build_model(self):
        self.embedding_model = PhonemeEmbedding(self.model_config, self.algorithm_config)
        self.model = FastSpeech2(self.preprocess_config, self.model_config, self.algorithm_config)
        self.loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)

        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

        self.lang_id = self.preprocess_config["lang_id"]
        d_word_vec = self.model_config["transformer"]["encoder_hidden"]
        
        # Tune init
        self.emb_layer = nn.Embedding(len(LANG_ID2SYMBOLS[self.lang_id]), d_word_vec, padding_idx=0)

        # Tune loss
        self.loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)

    def build_optimized_model(self):
        return nn.ModuleList([self.emb_layer, self.model])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        return saver

    def init_codebook_type(self):        
        codebook_config = self.algorithm_config["adapt"]["phoneme_emb"]
        if codebook_config["type"] == "embedding":
            self.codebook_type = "table-sep"
        elif codebook_config["type"] == "codebook":
            self.codebook_type = codebook_config["attention"]["type"]
        else:
            raise NotImplementedError
        self.adaptation_steps = self.algorithm_config["adapt"]["train"]["steps"]

    def tune_init(self):
        txt_path = f"{self.preprocess_config['path']['preprocessed_path']}/{self.preprocess_config['subsets']['train']}.txt"
        ref_phn_feats = generate_hubert_features(txt_path, lang_id=self.lang_id)
        ref_phn_feats = torch.from_numpy(ref_phn_feats).float()
        with torch.no_grad():
            embedding = self.embedding_model.get_new_embedding(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=self.lang_id)
            self.emb_layer._parameters['weight'] = embedding

            # visualization
            self.visualize_matching(ref_phn_feats)

    def common_step(self, batch, batch_idx, train=True):
        if getattr(self, "fix_spk_args", None) is None:
            self.fix_spk_args = batch[2]
        emb_texts = self.emb_layer(batch[3])
        output = self.model(batch[2], emb_texts, *(batch[4:]))
        loss = self.loss_func(batch, output)
        return loss, output

    def synth_step(self, batch, batch_idx):  # only used when tune
        emb_texts = self.emb_layer(batch[3])
        output = self.model(self.fix_spk_args, emb_texts, *(batch[4:6]), average_spk_emb=True)
        return output
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 12, "data with 12 elements"

    def training_step(self, batch, batch_idx):
        loss, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': loss[0], 'losses': loss, 'output': output, '_batch': batch}

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 12, "data with 12 elements"

    def validation_step(self, batch, batch_idx):
        val_loss, predictions = self.common_step(batch, batch_idx, train=False)
        synth_predictions = self.synth_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}":v for k,v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': batch, 'synth': synth_predictions}

    def visualize_matching(self, ref_phn_feats):
        matching = self.embedding_model.get_matching(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=self.lang_id)
        self.codebook_analyzer.visualize_matching(0, matching)
