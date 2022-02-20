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


class DualMetaTuneSystem(System):
    """ 
    Tune version of dual system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)

    def build_model(self):
        self.lang_id = self.preprocess_config["lang_id"]
        codebook_config = self.algorithm_config["adapt"]["phoneme_emb"]
        codebook_size = codebook_config["size"]
        d_feat = codebook_config["representation_dim"]
        d_word_vec = self.model_config["transformer"]["encoder_hidden"]
        num_heads = 4
        
        # Shared part
        self.codebook = Codebook(codebook_size, d_feat, 
                                d_word_vec, num_heads=num_heads)
        self.banks = SoftBank(codebook_size, d_word_vec, num_heads=num_heads)

        # ASR part
        self.asr_head = ASRCenterHead(d_word_vec, multilingual=False)

        # TTS part
        self.model = FastSpeech2(self.preprocess_config, self.model_config, self.algorithm_config)
        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

        # Tune init
        self.emb_layer = nn.Embedding(len(LANG_ID2SYMBOLS[self.lang_id]), d_word_vec, padding_idx=0)

        # Loss (Not used by tune)
        # self.tts_loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)
        # self.asr_loss_func1 = PhonemeClassificationLoss()
        # self.asr_loss_func2 = nn.MSELoss()

        # Tune loss
        self.loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)

    def build_optimized_model(self):
        return nn.ModuleList([self.emb_layer, self.model])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        return saver

    def tune_init(self):
        txt_path = f"{self.preprocess_config['path']['preprocessed_path']}/{self.preprocess_config['subsets']['train']}.txt"
        ref_phn_feats = generate_hubert_features(txt_path, lang_id=self.lang_id)
        ref_phn_feats = torch.from_numpy(ref_phn_feats).float()
        with torch.no_grad():
            attn = self.codebook(ref_phn_feats.unsqueeze(0))
            embedding = self.banks(attn, pad=True).squeeze(0)  # N, d_word_vec
            self.emb_layer._parameters['weight'] = embedding

            # visualization
            self.visualize_matching(ref_phn_feats)
            self.phoneme_transfer(ref_phn_feats, target_ids=[0, 1, 2])

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
        matching = self.codebook.get_matching(ref=ref_phn_feats, lang_id=self.lang_id)
        self.codebook_analyzer.visualize_matching(0, matching)

    def phoneme_transfer(self, ref_phn_feats, target_ids=[]):
        def transfer_embedding(embedding, src_lang_id, target_lang_id, mask):
            transfer_dist = self.asr_head(embedding.unsqueeze(0), lang_ids=target_lang_id)  # transfer to target language
            soft_transfer_dist = F.softmax(transfer_dist, dim=2)
            title = f"{LANG_ID2NAME[src_lang_id]}-{LANG_ID2NAME[target_lang_id]}"
            info = MatchingGraphInfo({
                "title": title,
                "y_labels": [LANG_ID2SYMBOLS[lang_id][int(m)] for m in mask[0]],
                "x_labels": LANG_ID2SYMBOLS[target_lang_id],
                "attn": soft_transfer_dist[0][mask].detach().cpu().numpy(),
                "quantized": False,
            })
            return info

        ref, lang_id = ref_phn_feats, self.lang_id
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)
        ref[ref != ref] = 0

        ref_mask = torch.nonzero(ref.sum(dim=1), as_tuple=True)

        attn = self.codebook(ref.unsqueeze(0))
        embedding = self.banks(attn, pad=True).squeeze(0)  # N, d_word_vec

        if len(target_ids) > 0:
            infos = [transfer_embedding(embedding, lang_id, t, ref_mask) for t in target_ids]
            self.codebook_analyzer.visualize_phoneme_transfer(0, infos)
