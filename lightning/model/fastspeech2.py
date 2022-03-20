import os
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from .speaker_encoder import SpeakerEncoder
from .phoneme_embedding import PhonemeEmbedding
from .adapters import SpeakerAdaLNAdapter, ProsodyEncoder, ProsodyAdaLNAdapter
from utils.tools import get_mask_from_lengths


class FastSpeech2(pl.LightningModule):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, algorithm_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.algorithm_config = algorithm_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        # If not using multi-speaker, would return None
        self.speaker_emb = SpeakerEncoder(preprocess_config, model_config, algorithm_config)

        if algorithm_config["adapt"]["type"] == "lang":
            self.phn_emb_generator = PhonemeEmbedding(model_config, algorithm_config)
            print("PhonemeEmbedding", self.phn_emb_generator)

        if algorithm_config["adapt"].get("AdaLN", None) is not None:
            self.add_AdaLN_adapters()

    def add_AdaLN_adapters(self):
        d_hidden_map = {
            "encoder": self.model_config["transformer"]["encoder_hidden"],
            "variance_adaptor": self.model_config["variance_predictor"]["filter_size"],
            "decoder": self.model_config["transformer"]["decoder_hidden"],
        }

        def reset_LN(module):
            module.reset_parameters()
            module.elementwise_affine = False
            for p in module.parameters():
                p.requires_grad = False

        self.AdaLN_adapters = {}
        self.spk_AdaLN_adapters = nn.ModuleList()
        self.psd_AdaLN_adapters = nn.ModuleList()

        if "speaker" in self.algorithm_config["adapt"]["AdaLN"]:
            d_spk_emb = self.model_config["transformer"]["encoder_hidden"]
            for module_name in self.algorithm_config["adapt"]["AdaLN"]["speaker_modules"]:
                d_hidden = d_hidden_map[module_name]
                for name, module in self.get_submodule(module_name).named_modules():
                    key = f"{module_name}.{name}"
                    if isinstance(module, nn.LayerNorm):
                        if key not in self.AdaLN_adapters:
                            self.AdaLN_adapters[key] = {}
                        self.AdaLN_adapters[key].update({
                            "speaker": len(self.spk_AdaLN_adapters),
                        })
                        self.spk_AdaLN_adapters.append(SpeakerAdaLNAdapter(d_spk_emb, d_hidden))
                        reset_LN(module)

        if "prosody" in self.algorithm_config["adapt"]["AdaLN"]:
            d_psd_emb = self.model_config["transformer"]["encoder_hidden"]
            self.prosody_encoder = ProsodyEncoder(
                d_psd_emb, n_layers=3,
                log_f0=self.algorithm_config["adapt"]["AdaLN"]["prosody"]["log_f0"],
                energy=self.algorithm_config["adapt"]["AdaLN"]["prosody"]["energy"],
            )
            for module_name in self.algorithm_config["adapt"]["AdaLN"]["prosody_modules"]:
                d_hidden = d_hidden_map[module_name]
                for name, module in self.get_submodule(module_name).named_modules():
                    key = f"{module_name}.{name}"
                    if isinstance(module, nn.LayerNorm):
                        if key not in self.AdaLN_adapters:
                            self.AdaLN_adapters[key] = {}
                        self.AdaLN_adapters[key].update({
                            "prosody": len(self.psd_AdaLN_adapters),
                        })
                        self.psd_AdaLN_adapters.append(ProsodyAdaLNAdapter(d_psd_emb, d_hidden))
                        reset_LN(module)

    def register_AdaLN_hooks(self, spk_emb=None, psd_emb=None):
        d_hidden_map = {
            "encoder": self.model_config["transformer"]["encoder_hidden"],
            "variance_adaptor": self.model_config["variance_predictor"]["filter_size"],
            "decoder": self.model_config["transformer"]["decoder_hidden"],
        }
        handles = {}

        for key in self.AdaLN_adapters:
            d_hidden = d_hidden_map[key.split('.')[0]]
            module = self.get_submodule(key)

            if psd_emb is not None:
                AdaLN_params = torch.zeros(psd_emb.shape[0], 2, d_hidden).to(psd_emb.device)
            else:
                AdaLN_params = torch.zeros(spk_emb.shape[0], 2, d_hidden).to(spk_emb.device)

            if spk_emb is not None and "speaker" in self.AdaLN_adapters[key]:
                spk_AdaLN_params = self.spk_AdaLN_adapters[self.AdaLN_adapters[key]["speaker"]](spk_emb)
                AdaLN_params += spk_AdaLN_params
            if psd_emb is not None and "prosody" in self.AdaLN_adapters[key]:
                psd_AdaLN_params = self.psd_AdaLN_adapters[self.AdaLN_adapters[key]["prosody"]](psd_emb)
                AdaLN_params += psd_AdaLN_params

            def hook(layer, input, output):
                return output * AdaLN_params[:, 0, None] + AdaLN_params[:, 1, None]
            handles[key] = module.register_forward_hook(hook)

        return handles

    def forward(
        self,
        speaker_args,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        average_spk_emb=False,
        reference_prosody=None,
    ):
        if self.speaker_emb is not None:
            spk_emb = self.speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True)
        else:
            spk_emb = None

        if reference_prosody is not None and hasattr(self, "prosody_encoder"):
            psd_emb = self.prosody_encoder(reference_prosody)
        else:
            psd_emb = None

        if self.algorithm_config["adapt"].get("AdaLN", None) is not None:
            handles = self.register_AdaLN_hooks(spk_emb, psd_emb)

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if spk_emb is not None:
            output += spk_emb.unsqueeze(1).expand(output.shape[0], max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        if spk_emb is not None:
            output += spk_emb.unsqueeze(1).expand(output.shape[0], max(mel_lens), -1)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        if self.algorithm_config["adapt"].get("AdaLN", None) is not None:
            for k, h in handles.items():
                h.remove()

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )


class AdaptiveFastSpeech2(FastSpeech2):
    def add_adapters(self):
        n_src_vocab, d_word_vec = self.encoder.src_word_emb.weight.shape
        self.meta_phn_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=self.encoder.src_word_emb.padding_idx
        )
        nn.init.zeros_(self.meta_phn_emb.weight)

        for layer in self.encoder.layer_stack:
            for sublayer in [layer.slf_attn, layer.pos_ffn]:
                sublayer.add_adapter(
                    32, # bottleneck
                    self.model_config["transformer"]["conv_kernel_size"],
                )

        for layer in self.decoder.layer_stack:
            for sublayer in [layer.slf_attn, layer.pos_ffn]:
                sublayer.add_adapter(
                    32, # bottleneck
                    self.model_config["transformer"]["conv_kernel_size"],
                )

        self.handles = {}

    def add_hooks(self):
        handles = []
        def src_seq_pre_hook(layer, input):
            self.hook["meta_phn_emb_seq"] = self.meta_phn_emb(input)
            return

        def duration_pre_hook(layer, input):
            self.hook["expanded_meta_phn_emb_seq"] = (
                self.variance_adaptor.length_regulator(
                    self.hook["meta_phn_emb_seq"],
                    input[1],
                    input[2]
                )
            )
            return

        handles += [
            self.encoder.src_word_emb.register_forward_pre_hook(src_seq_pre_hook),
            self.variance_adaptor.length_regulator.register_forward_pre_hook(duration_pre_hook),
        ]
        for layer in self.encoder.layer_stack:
            for sublayer in [layer.slf_attn, layer.pos_ffn]:
                sublayer.add_adapter(
                    32, # bottleneck
                    self.model_config["transformer"]["conv_kernel_size"],
                )
                handles.append(sublayer.meta_adapter.register_forward_pre_hook(
                    lambda _, input: (
                        input + self.hook["meta_phn_emb_seq"],
                    )
                ))

        for layer in self.decoder.layer_stack:
            for sublayer in [layer.slf_attn, layer.pos_ffn]:
                sublayer.add_adapter(
                    32, # bottleneck
                    self.model_config["transformer"]["conv_kernel_size"],
                )
                handles.append(sublayer.meta_adapter.register_forward_pre_hook(
                    lambda _, input: (
                        input + self.hook["expanded_meta_phn_emb_seq"],
                    )
                ))

        self.handles[id(self)] = handles

    def remove_hooks(self):
        for h in self.handles[id(self)]:
            h.remove()
        del self.handles[id(self)]

    def forward(
        self,
        speaker_args,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        average_spk_emb=False,
        reference_prosody=None,
    ):
        psd_AdaLN_params = self.prosody_AdaLN_adapter(reference_prosody)
        if self.speaker_emb is not None:
            spk_emb = self.speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True)
            spk_AdaLN_params = self.speaker_AdaLN_adapter(spk_emb)
        else:
            spk_emb = None
            spk_AdaLN_params = torch.zeros_like(psd_AdaLN_params)

        AdaLN_params = spk_AdaLN_params + psd_AdaLN_params
        i = 0
        for module in [self.encoder, self.decoder]:
            for layer in module.layer_stack:
                layer.slf_attn.set_AdaLN_params(AdaLN_params[:, i, 0])
                layer.pos_ffn.set_AdaLN_params(AdaLN_params[:, i, 1])
                i += 1

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if spk_emb is not None:
            output += spk_emb.unsqueeze(1).expand(output.shape[0], max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        if spk_emb is not None:
            output += spk_emb.unsqueeze(1).expand(output.shape[0], max(mel_lens), -1)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
