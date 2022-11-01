import torch.nn as nn
from collections import defaultdict
from typing import List
from lightning.modules.speaker import SpeakerEncoder
from lightning.modules.prosody import ProsodyEncoder
from pytorch_lightning.utilities.cli import instantiate_class

from .base import CallbackHooks


class Additive(CallbackHooks):
    """One `Additive` for speaker, another `Additive` for prosody."""
    def __init__(self,
                 *args,
                 **kwargs):
        """Additive conditioning usually doesn't need extra transform."""
        super().__init__(*args, **kwargs)

    def register_hooks(self, model, _emb):
        if _emb is None:
            raise ValueError

        if _emb.dim() not in [2, 3]:
            raise ValueError

        if _emb.dim() == 2:
            _emb = _emb.unsqueeze(1)

        for module_name in self.pre_hook_modules:
            assert module_name in ("variance_adaptor", "decoder")
            module = model.get_submodule(module_name)
            self.forward_pre_hooks[module_name].append(
                self.register_pre_hook(module, _emb)
            )

    def register_pre_hook(self, module, _emb):
        def pre_hook(_module, input):
            if _emb.shape[0] not in {input[0].shape[0], 1}:
                raise ValueError
            max_len = input[0].shape[1]
            return (
                input[0] + _emb.expand(1, max_len, -1),
                *input[1:]
            )

        return module.register_forward_pre_hook(pre_hook)
