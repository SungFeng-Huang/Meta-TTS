import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .FastSpeech2 import BaselineSystem
from cross_lingual.model.interface import Hookable, Tunable
from cross_lingual.model.fscl import get_fscl_plugin_cls
from cross_lingual.utils.tool import generate_reference_info


class TransEmbTuneSystem(BaselineSystem, Tunable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        super().build_model()
        self.fscl = get_fscl_plugin_cls(self.model_config["fscl_type"])(self.model_config)

    def tune_init(self, data_configs):
        assert len(data_configs) == 1, f"Currently only support adapting to one language"
        print("Generate reference...")
        ref_infos = generate_reference_info(data_configs[0])
        self.target_lang_id = ref_infos[0]["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")

        print("Embedding initialization...")
        self.cuda()
        with torch.no_grad():
            table, attn = self.fscl.build_embedding_table(ref_infos, return_attn=True)
            self.attn = attn
            self.embedding_model.tables[f"table-{ref_infos[0]['symbol_id']}"].copy_(table)
        for p in self.embedding_model.parameters():
            p.requires_grad = True
        self.cpu()
        self.setup_tune_mode()

    def setup_tune_mode(self):
        from cross_lingual.utils.freezing import get_norm_params, get_bias_params
        tune_modules = self.algorithm_config.get("tune_modules", ["all"])
        tune_specific = self.algorithm_config.get("tune_specific", "none")
        if tune_modules == ["all"] and tune_specific == "none":
            return
        print("Freezing some parameters:")
        print(f"Tune modules: {tune_modules}, tune specific: {tune_specific}.")
        mapping = {
            "all": super().build_optimized_model(),
            "embedding": self.embedding_model,
            "encoder": self.model.encoder,
            "variance_adaptor": self.model.variance_adaptor,
            "decoder": self.model.decoder,
            "mel_linear": self.model.mel_linear,
            "postnet": self.model.postnet
        }
        for param in self.model.parameters():
            param.requires_grad = False
        for name in tune_modules:
            if tune_specific is None:
                for param in mapping[name]:
                    param.requires_grad = True
            elif tune_specific == "bias":
                for param in get_bias_params(mapping[name]):
                    param.requires_grad = True
            elif tune_specific == "norm":
                for param in get_norm_params(mapping[name]):
                    param.requires_grad = True
            else:
                raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)
        synth_predictions = self.synth_step(batch, batch_idx)
        
        if batch_idx == 0:
            try:
                assert isinstance(self.fscl, Hookable)
                layer_weights = self.fscl.get_hook("layer_weights")
                self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
            except:
                pass
            self.saver.log_codebook_attention(self.logger, self.attn, batch[-1][0].item(), batch_idx, self.global_step + 1, "val")
        
        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': batch, 'synth': synth_predictions}
    
    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.split('.')[0] in ["fscl"]:
                continue
            new_state_dict[k] = state_dict[k]
        checkpoint["state_dict"] = new_state_dict

        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        state_dict_pop_keys = []
        state_dict_remap_keys = []
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if self.local_rank == 0:
                        print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                if k.startswith("codebook_attention"):  # Checkpoints before PlugIn classes are created
                    k_new = "fscl." + k
                    state_dict_remap_keys.append((k_new, k))
                
                if self.local_rank == 0:
                    print(f"Dropping parameter {k}")
                state_dict_pop_keys.append(k)
                is_changed = True

        if len(state_dict_remap_keys) > 0:
            for (k_new, k) in state_dict_remap_keys:
                print(f"Remap parameters from old to new ({k} => {k_new}).")
                state_dict[k_new] = state_dict[k]

        # modify state_dict format to model_state_dict format
        for k in state_dict_pop_keys:
            state_dict.pop(k)
        for k in model_state_dict:
            if k not in state_dict:
                if k.startswith("fscl.upstream"):
                    pass
                else:
                    print("Reinitialized: ", k)
                state_dict[k] = model_state_dict[k]

        if is_changed:
            checkpoint.pop("optimizer_states", None)
