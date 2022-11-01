import torch
import torch.nn as nn
from collections import defaultdict
from typing import List
from lightning.modules.speaker import SpeakerEncoder
from lightning.modules.prosody import ProsodyEncoder
from pytorch_lightning.utilities.cli import instantiate_class

from lightning.model.utils import reset_LN
from .base import CallbackHooks, EmbeddingTransform


class AdaLN(CallbackHooks):
    """One `AdaLN` for both speaker and prosody."""
    def __init__(self,
                 spk_hook_modules: List[str],
                 psd_hook_modules: List[str],
                 d_emb: int,
                 d_hidden: int,
                 *args,
                 **kwargs):
        """Additive conditioning usually doesn't need extra transform."""
        super().__init__(*args, **kwargs)
        #TODO: d_emb -> self.d_spk_emb, self.d_psd_emb
        self.d_emb, self.d_hidden = d_emb, d_hidden

        self.hook_typelist = defaultdict(list)
        for module in spk_hook_modules:
            self.hook_typelist[module].append("spk")
        for module in psd_hook_modules:
            self.hook_typelist[module].append("psd")

        self.transform_ids = defaultdict(dict)
        self.transforms = nn.ModuleList()

    def register_hooks(self, model):
        for name in self.hook_typelist:
            for sub_name, sub_module in model.get_submodule(name).named_modules():
                if not isinstance(sub_module, nn.LayerNorm):
                    continue

                reset_LN(sub_module)

                module_name = f"{name}.{sub_name}"

                self.forward_hooks[module_name].append(
                    self.register_module_hook(
                        sub_module, module_name, self.hook_typelist[name])
                )

    def register_module_hook(self, sub_module, module_name, hook_types):
        spk_emb = self.extracted_embs["spk_emb"]
        psd_emb = self.extracted_embs["psd_emb"]

        assert spk_emb.shape == self.d_emb
        assert psd_emb.shape == self.d_emb
        assert spk_emb.device == psd_emb.device

        def hook(layer, input, output):
            assert isinstance(output, tuple)
            if module_name not in self.transform_ids:
                if "spk" in hook_types:
                    self.transform_ids[module_name].update({
                        "spk": len(self.transforms),
                    })
                    self.transforms.append(
                        EmbeddingTransform(spk_emb.shape[2],
                                           output[0].shape[2])
                        .to(output[0].device)
                    )
                if "psd" in hook_types:
                    self.transform_ids[module_name].update({
                        "psd": len(self.transforms),
                    })
                    self.transforms.append(
                        EmbeddingTransform(psd_emb.shape[2],
                                           output[0].shape[2])
                        .to(output[0].device)
                    )
            transform_ids = self.transform_ids[module_name]

            scale = torch.zeros_like(output[0])
            shift = torch.zeros_like(output[0])
            spk_params, psd_params = None, None
            if "spk" in transform_ids:
                if spk_emb.shape[0] not in {output[0].shape[0], 1}:
                    raise ValueError
                spk_transform = self.transforms[transform_ids["spk"]]
                spk_params = spk_transform(spk_emb)
                scale += spk_params[:, 0]
                shift += spk_params[:, 1]

            if "psd" in transform_ids:
                if psd_emb.shape[0] not in {output[0].shape[0], 1}:
                    raise ValueError
                psd_transform = self.transforms[transform_ids["psd"]]
                psd_params= psd_transform(psd_emb)
                scale += psd_params[:, 0]
                shift += psd_params[:, 1]

            return (output[0] * scale + shift, *output[1:])

        return sub_module.register_forward_hook(hook)
