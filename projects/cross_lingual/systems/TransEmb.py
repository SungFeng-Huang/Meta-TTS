import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from lightning.build import build_all_speakers
from systems.adaptor import AdaptorSystem
from lightning.model import FastSpeech2Loss, FastSpeech2
from cross_lingual.callbacks.saver import Saver
from cross_lingual.model.interface import Hookable
from cross_lingual.model.fscl import get_fscl_plugin_cls


class TransEmbSystem(AdaptorSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build_configs(self):
        self.spk_config = {
            "emb_type": self.model_config["speaker_emb"],
            "speakers": build_all_speakers(self.data_configs)
        }
        self.bs = self.train_config["optimizer"]["batch_size"]
    
    def build_model(self):
        self.model = FastSpeech2(self.model_config, spk_config=self.spk_config)
        self.loss_func = FastSpeech2Loss(self.model_config)
        self.fscl = get_fscl_plugin_cls(self.model_config["fscl_type"])(self.model_config)

    def build_optimized_model(self):
        return nn.ModuleList([self.model]) + self.fscl.build_optimized_model()

    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver
    
    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 3, "sup + qry + sup_info"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 13, "data with 13 elements"
    
    def common_step(self, batch, batch_idx, train=True):
        emb_table, attn = self.fscl.build_embedding_table([batch[0][2]], return_attn=~train)
        qry_batch = batch[0][1][0]
        emb_texts = F.embedding(qry_batch[3], emb_table, padding_idx=0)
        output = self.model(qry_batch[2], emb_texts, *(qry_batch[4:]))
        loss = self.loss_func(qry_batch[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        
        return loss_dict, output, attn

    def training_step(self, batch, batch_idx):
        train_loss_dict, output, _ = self.common_step(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions, attn = self.common_step(batch, batch_idx, train=False)
        qry_batch = batch[0][1][0]

        if batch_idx == 0:
            try:
                assert isinstance(self.fscl, Hookable)
                layer_weights = self.fscl.get_hook("layer_weights")
                self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
            except:
                pass
        if batch_idx % 4 == 0:
            lang_id = qry_batch[-1][0].item()  # all batch belongs to the same language
            self.saver.log_codebook_attention(self.logger, attn, lang_id, batch_idx, self.global_step + 1, "val")
        
        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': qry_batch}
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint = self.fscl.on_save_checkpoint(checkpoint, prefix="fscl")
        return checkpoint
