from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from dlhlp_lib.s3prl import S3PRLExtractor
from dlhlp_lib.common.layers import WeightedSumLayer

import Define
from text.define import LANG_ID2SYMBOLS
from .interface import FSCLPlugIn, Hookable
from .upstream import Padding
from .codebook import SoftMultiAttCodebook
from .reduction import PhonemeQueryExtractor
from cross_lingual.utils.tool import ssl_match_length


class LinearFSCLPlugIn(FSCLPlugIn, Hookable):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__()
        self.model_config = model_config
        self._build_model()
        self._custom_hooks = {}

    def _build_model(self):
        if self.upstream == "mel":
            self.upstream = Padding()
            self.featurizer = None
            upstream_dim = 80
        else:
            self.upstream = S3PRLExtractor(Define.UPSTREAM)
            self.featurizer = WeightedSumLayer(
                n_in_layers=self.upstream.n_layers,
                specific_layer=Define.LAYER_IDX
            )
            upstream_dim = self.upstream.dim
            self.upstream.freeze()
        self.downstream = nn.Linear(
            upstream_dim, self.model_config["transformer"]["encoder_hidden"],
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.codebook_attention = SoftMultiAttCodebook(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=self.model_config["transformer"]["encoder_hidden"],
            num_heads=self.model_config["downstream"]["transformer"]["nhead"],
        )
        self.checkpoint_remove_list = ["upstream"]

    def build_optimized_model(self):
        return nn.ModuleList([self.codebook_attention, self.downstream])

    def build_embedding_table(self, ref_infos, return_attn=False):
        hiddens, avg_frames_list, phonemes_list = [], [], []
        for info in ref_infos:
            with torch.no_grad():
                repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                repr = repr.detach()
            if self.featurizer is not None:
                repr = self.featurizer(repr)
            repr = self.downstream(repr)
            hiddens.extend([x1 for x1 in repr])
            avg_frames_list.extend(info["avg_frames"])
            phonemes_list.extend(info["phonemes"])
            
        table_pre = self.phoneme_query_extractor(hiddens, avg_frames_list, 
                            len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
        
        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        table[0].fill_(0)
        return table, attn
        
    def on_save_checkpoint(self, checkpoint, prefix=""):
        """ Remove pretrained weights in checkpoint to save disk space. """
        if prefix != "":
            remove_list = [f"{prefix}.{name}" for name in self.checkpoint_remove_list]
        else:
            remove_list = self.checkpoint_remove_list
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.startswith(tuple(remove_list)):
                continue
            new_state_dict[k] = state_dict[k]
        checkpoint["state_dict"] = new_state_dict

        return checkpoint

    # Hooks
    def build_hook(self, key: str, *args, **kwargs):
        if key in [
            "layer_weights"
        ]:
            self._custom_hooks[key] = True
    
    def get_hook(self, key: str, *args, **kwargs):
        if key not in self._custom_hooks:
            raise ValueError(f"Hook ({key}) does not exist.")
        if key == "layer_weights":
            return F.softmax(self.featurizer.weight_raw, dim=0)
        else:
            raise NotImplementedError(f"Hook ({key}) is not done.")


# class TransformerFSCLPlugIn(FSCLPlugIn, Hookable):
#     def __init__(self, model_config, *args, **kwargs) -> None:
#         super().__init__()
#         self.model_config = model_config
#         self._build_model()
#         self._custom_hooks = {}

#     def _build_model(self):
#         if self.upstream == "mel":
#             self.upstream = Padding()
#             self.featurizer = None
#             upstream_dim = 80
#         else:
#             self.upstream = S3PRLExtractor(Define.UPSTREAM)
#             self.featurizer = WeightedSumLayer(
#                 n_in_layers=self.upstream.n_layers,
#                 specific_layer=Define.LAYER_IDX
#             )
#             upstream_dim = self.upstream.dim
#             self.upstream.freeze()
#         self.downstream = nn.Linear(
#             upstream_dim, self.model_config["transformer"]["encoder_hidden"],
#         )
#         self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
#         self.codebook_attention = SoftMultiAttCodebook(
#             codebook_size=self.model_config["codebook_size"],
#             embed_dim=self.model_config["transformer"]["encoder_hidden"],
#             num_heads=self.model_config["downstream"]["transformer"]["nhead"],
#         )
#         self.checkpoint_remove_list = ["upstream"]

#     def build_optimized_model(self):
#         return nn.ModuleList([self.codebook_attention, self.downstream])

#     def build_embedding_table(self, ref_infos, return_attn=False):
#         self.upstream.eval()
#         hiddens, avg_frames_list, phonemes_list = [], [], []
#         for info in ref_infos:
#             with torch.no_grad():
#                 repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
#                 repr = ssl_match_length(repr, info["max_len"].item())  # Unavoidable since we need to match shape in transformer forward.
#                 repr = repr.detach()
#             if self.featurizer is not None:
#                 repr = self.featurizer(repr)
#             repr = self.downstream(repr, info["lens"].cpu())
#             hiddens.extend([x1 for x1 in repr])
#             avg_frames_list.extend(info["avg_frames"])
#             phonemes_list.extend(info["phonemes"])
            
#         table_pre = self.phoneme_query_extractor(hiddens, avg_frames_list, 
#                             len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
        
#         table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
#         table = table.squeeze(0)  # n_symbols, dim
#         table[0].fill_(0)
#         return table, attn

#     # Hooks
#     def build_hook(self, key: str, *args, **kwargs):
#         if key in [
#             "layer_weights"
#         ]:
#             self._custom_hooks[key] = True
    
#     def get_hook(self, key: str, *args, **kwargs):
#         if key not in self._custom_hooks:
#             raise ValueError(f"Hook ({key}) does not exist.")
#         if key == "layer_weights":
#             return F.softmax(self.featurizer.weight_raw, dim=0)
#         else:
#             raise NotImplementedError(f"Hook ({key}) is not done.")


def get_fscl_plugin_cls(key: str) -> Type[FSCLPlugIn]:
    if key == "linear":
        return LinearFSCLPlugIn
    elif key == "transformer":
        pass
    else:
        raise NotImplementedError
