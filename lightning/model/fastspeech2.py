from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from .speaker_encoder import SpeakerEncoder
from .phoneme_embedding import PhonemeEmbedding
from .adapters import SpeakerAdaLNAdapter, ProsodyEncoder, ProsodyAdaLNAdapter
from .utils import reset_LN
from utils.tools import get_mask_from_lengths


class FastSpeech2(pl.LightningModule):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, *args, **kwargs):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

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
        **kwargs,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

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


class AdaptiveFastSpeech2(FastSpeech2):
    """ Adaptive FastSpeech2 """

    def __init__(self, preprocess_config, model_config, algorithm_config):
        super().__init__(preprocess_config, model_config)
        self.algorithm_config = algorithm_config

        transformer_config = model_config["transformer"]
        variance_adaptor_config = model_config["variance_predictor"]

        self.d_hidden_map = {
            "encoder": transformer_config["encoder_hidden"],
            "variance_adaptor": variance_adaptor_config["filter_size"],
            "decoder": transformer_config["decoder_hidden"],
        }

        self.handles = defaultdict(dict)
        self.forward_hooks = defaultdict(list)
        self.forward_pre_hooks = defaultdict(list)

        if algorithm_config["adapt"]["speaker_emb"] is not None:
            # If not using multi-speaker, would return None
            self.speaker_emb = SpeakerEncoder(preprocess_config, model_config, algorithm_config)
            self.d_spk_emb = self.model_config["transformer"]["encoder_hidden"]

        if algorithm_config["adapt"]["type"] == "lang":
            self.phn_emb_generator = PhonemeEmbedding(model_config, algorithm_config)
            print("PhonemeEmbedding", self.phn_emb_generator)

        if algorithm_config["adapt"].get("AdaLN", None) is not None:
            AdaLN_config = algorithm_config["adapt"]["AdaLN"]
            self.AdaLN_adapter_ids = defaultdict(dict)

            if "speaker" in AdaLN_config:
                self.spk_AdaLN_adapters = nn.ModuleList()
                self.add_spk_AdaLN_adapters(AdaLN_config)

            if "prosody" in AdaLN_config:
                self.d_psd_emb = model_config["transformer"]["encoder_hidden"]
                log_f0 = AdaLN_config["prosody"]["log_f0"]
                energy = AdaLN_config["prosody"]["energy"]

                self.prosody_encoder = ProsodyEncoder(
                    self.d_psd_emb, n_layers=3, log_f0=log_f0, energy=energy,
                )
                self.psd_AdaLN_adapters = nn.ModuleList()
                self.add_psd_AdaLN_adapters(AdaLN_config)

    def add_spk_AdaLN_adapters(self, AdaLN_config):
        for name in AdaLN_config["speaker_modules"]:
            for sub_name, sub_module in self.get_submodule(name).named_modules():
                if isinstance(sub_module, nn.LayerNorm):
                    self.AdaLN_adapter_ids[f"{name}.{sub_name}"].update({
                        "speaker": len(self.spk_AdaLN_adapters),
                    })
                    self.spk_AdaLN_adapters.append(
                        SpeakerAdaLNAdapter(self.d_spk_emb, self.d_hidden_map[name])
                    )
                    reset_LN(sub_module)

    def add_psd_AdaLN_adapters(self, AdaLN_config):
        for name in AdaLN_config["prosody_modules"]:
            for sub_name, sub_module in self.get_submodule(name).named_modules():
                if isinstance(sub_module, nn.LayerNorm):
                    self.AdaLN_adapter_ids[f"{name}.{sub_name}"].update({
                        "prosody": len(self.psd_AdaLN_adapters),
                    })
                    self.psd_AdaLN_adapters.append(
                        ProsodyAdaLNAdapter(self.d_psd_emb, self.d_hidden_map[name])
                    )
                    reset_LN(sub_module)

    def register_AdaLN_hooks(self, spk_emb=None, psd_emb=None):
        for key in self.AdaLN_adapter_ids:
            d_hidden = self.d_hidden_map[key.split('.')[0]]
            module = self.get_submodule(key)
            adapter_ids = self.AdaLN_adapter_ids[key]

            if psd_emb is not None:
                AdaLN_params = torch.zeros(psd_emb.shape[0], 2, d_hidden).to(psd_emb.device)
            else:
                AdaLN_params = torch.zeros(spk_emb.shape[0], 2, d_hidden).to(spk_emb.device)

            if spk_emb is not None and "speaker" in adapter_ids:
                AdaLN_params += self.spk_AdaLN_adapters[adapter_ids["speaker"]](spk_emb)
            if psd_emb is not None and "prosody" in adapter_ids:
                AdaLN_params += self.psd_AdaLN_adapters[adapter_ids["prosody"]](psd_emb)

            def hook(layer, input, output):
                return output * AdaLN_params[:, 0, None] + AdaLN_params[:, 1, None]
            self.forward_hooks[key].append(module.register_forward_hook(hook))

    def register_additive_hooks(self, spk_emb, max_src_len, max_mel_len=None):
        additive_config = self.algorithm_config["adapt"]["additive"]

        if (additive_config.get("speaker_modules", None) is not None
                and spk_emb is not None):
            # for name in additive_config["speaker_modules"]:
            if "variance_adaptor" in additive_config["speaker_modules"]:
                def VA_pre_hook(module, input):
                    batch_size = input[0].shape[0]
                    return (
                        input[0]
                        + (spk_emb.unsqueeze(1)
                           .expand(batch_size, max_src_len, -1)),
                        *input[1:]
                    )

                self.forward_pre_hooks["variance_adaptor"].append(
                    self.variance_adaptor.register_forward_pre_hook(VA_pre_hook)
                )

            if "decoder" in additive_config["speaker_modules"]:
                if max_mel_len is None:
                    # def get_max_mel_len(module, input, output):
                    #     # self.register_buffer("max_mel_len", output[5], persistent=False)
                    #     self.register_buffer("max_mel_len", output[0].shape[1],
                    #                          persistent=False)
                    #
                    # self.forward_hooks["variance_adaptor"].append(
                    #     self.variance_adaptor.register_forward_hook(get_max_mel_len)
                    # )

                    def decoder_pre_hook(module, input):
                        batch_size = input[0].shape[0]
                        _max_mel_len = input[0].shape[1]
                        return (
                            input[0]
                            + (spk_emb.unsqueeze(1)
                               .expand(batch_size, _max_mel_len, -1)),
                            *input[1:]
                        )

                    self.forward_pre_hooks["decoder"].append(
                        self.decoder.register_forward_pre_hook(decoder_pre_hook)
                    )

                else:
                    def decoder_pre_hook(module, input):
                        batch_size = input[0].shape[0]
                        return (
                            input[0]
                            + (spk_emb.unsqueeze(1)
                               .expand(batch_size, max_mel_len, -1)),
                            *input[1:]
                        )

                    self.forward_pre_hooks["decoder"].append(
                        self.decoder.register_forward_pre_hook(decoder_pre_hook)
                    )

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

        # self.handles[id(self)] = handles

    def remove_hooks(self):
        for h in self.handles[id(self)]:
            h.remove()
        # del self.handles[id(self)]

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
        **kwargs,
    ):
        spk_emb = None
        if hasattr(self, "speaker_emb"):
            spk_emb = self.speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True)

        psd_emb = None
        if hasattr(self, "prosody_encoder") and reference_prosody is not None:
            psd_emb = self.prosody_encoder(reference_prosody)

        if hasattr(self, "AdaLN_adapter_ids"):
            self.register_AdaLN_hooks(spk_emb, psd_emb)

        if self.algorithm_config["adapt"].get("additive", None) is not None:
            self.register_additive_hooks(spk_emb, max_src_len, max_mel_len)

        outputs = super().forward(
            speaker_args,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        for key in self.forward_hooks:
            for handle in self.forward_hooks[key]:
                handle.remove()
        for key in self.forward_pre_hooks:
            for handle in self.forward_pre_hooks[key]:
                handle.remove()
        # if self.algorithm_config["adapt"].get("AdaLN", None) is not None:
            # for k, h in handles.items():
                # h.remove()
        self.forward_hooks = defaultdict(list)
        self.forward_pre_hooks = defaultdict(list)

        return outputs
