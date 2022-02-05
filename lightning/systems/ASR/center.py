#!/usr/bin/env python3

import torch
import torch.nn as nn
from lightning.model.asr_model import ASRCenterHead, Codebook, SoftBank

from ..system2 import System
from lightning.model.loss import PhonemeClassificationLoss
from lightning.callbacks.asr_saver import Saver
from lightning.utils import asr_loss2dict as loss2dict


class CenterSystem(System):
    """
    Concrete class of ASR head for codebook quality evaluation (w/o average).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # tests
        self.test_list = {
            "print_bank_norm": self.print_bank_norm,
            "print_head_norm": self.print_head_norm,
            "print_dist_norm": self.print_dist_norm,
        }

    def build_model(self):
        codebook_config = self.algorithm_config["adapt"]["phoneme_emb"]
        codebook_size = codebook_config["size"]
        d_feat = codebook_config["representation_dim"]
        d_word_vec = self.model_config["transformer"]["encoder_hidden"]
        num_heads = 4

        self.codebook = Codebook(codebook_size, d_feat, 
                                d_word_vec, num_heads=num_heads)
        self.banks = SoftBank(codebook_size, d_word_vec, num_heads=num_heads)

        self.asr_head = ASRCenterHead(d_word_vec, multilingual=True)
        self.loss_func = PhonemeClassificationLoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.codebook, self.banks, self.asr_head])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        saver.re_id = True
        return saver
    
    def common_step(self, batch, batch_idx, train=True):
        lang_ids, representations = batch[12], batch[13]
        attn = self.codebook(representations)
        emb_texts = self.banks(attn, pad=False)  # B, L, d_word_vec
        predictions = self.asr_head(emb_texts, lang_ids=lang_ids)

        loss = self.loss_func(batch, predictions)
        return loss, predictions
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 14, "data with 14 elements"
    
    def training_step(self, batch, batch_idx):
        train_loss, predictions = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss, 'losses': train_loss, 'output': predictions, '_batch': batch, 'lang_id': batch[12][0]}

    def on_val_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 14, "data with 14 elements"
    
    def validation_step(self, batch, batch_idx):
        val_loss, predictions = self.common_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        
        # calculate acc
        mask = (batch[3] != 0)
        acc = ((batch[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        loss_dict.update({"Val/Acc": acc.item()})

        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': batch, 'lang_id': batch[12][0]}

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        outputs = {}
        for test_name, test_fn in getattr(self, "test_list", {}).items(): 
            outputs[test_name] = test_fn(batch, batch_idx)

        return outputs

    def print_bank_norm(self, batch, batch_idx):
        if batch_idx == 0:  # Execute only once
            self.eval()
            with torch.no_grad():
                print("Bank norm:")
                print(torch.mean(self.banks.banks ** 2, dim=1))

    def print_head_norm(self, batch, batch_idx):
        if batch_idx == 0:  # Execute only once
            self.eval()
            with torch.no_grad():
                concat_table = self.asr_head.get_concat_table()
                print("Head norm:")
                print(torch.mean(concat_table ** 2, dim=1))

    def print_dist_norm(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            print("Distance norm:")
            texts = batch[3]  # B, L
            _, prediction = self.common_step(batch, batch_idx, train=False)

            # Reference: https://stackoverflow.com/questions/66604482/indexing-using-pytorch-tensors-along-one-specific-dimension-with-3-dimensional-t
            output = -torch.gather(prediction, -1, texts.unsqueeze(-1)).squeeze(-1)
            print(output)
