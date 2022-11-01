import torch.nn as nn
from collections import defaultdict
from typing import List
from lightning.modules.speaker import SpeakerEncoder
from lightning.modules.prosody import ProsodyEncoder
from pytorch_lightning.utilities.cli import instantiate_class


class PreCallbackExtractor(nn.Module):
    def __init__(self,
                 speaker_enc: SpeakerEncoder = None,
                 prosody_enc: ProsodyEncoder = None):
        """Pre-callback feature extractors.

        Currently includes speaker and prosody encoders.
        """
        super().__init__()

        if speaker_enc is not None:
            self.speaker_enc = speaker_enc
            self.d_spk_emb = self.speaker_enc.d_hidden

        if prosody_enc is not None:
            self.prosody_enc = prosody_enc
            self.d_psd_emb = self.prosody_enc.d_emb

    def forward(self, speaker_args=None, average_speaker_emb=False, reference_prosody=None):
        output = {}
        if hasattr(self, "speaker_enc") and speaker_args is not None:
            # shape: [B, D]
            spk_emb = self.speaker_enc(speaker_args)
            if average_speaker_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True)
            output["spk_emb"] = spk_emb
        if hasattr(self, "prosody_enc") and reference_prosody is not None:
            # shape: [B, D]
            psd_emb = self.prosody_enc(reference_prosody)
            output["psd_emb"] = psd_emb
        return output


class CallbackHooks(nn.Module):
    def __init__(self,
                 pre_callback: PreCallbackExtractor,
                 *args,
                 spk_hook_modules: List[str] = None,
                 psd_hook_modules: List[str] = None,
                 spk_pre_hook_modules: List[str] = None,
                 psd_pre_hook_modules: List[str] = None,
                 **kwargs):
        super().__init__()
        self.pre_callback_extractor = pre_callback
        self.extracted_embs = {}
        def extractor_hook(module, input, output):
            for key, _emb in output[0].items():
                if _emb is None or _emb.dim() not in [2, 3]:
                    raise ValueError
                if _emb.dim() == 2:
                    _emb = _emb.unsqueeze(1)
                self.extracted_embs[key] = _emb
        self.pre_callback_forward_hooks = [
            self.pre_callback_extractor.register_forward_hook(extractor_hook)
        ]

        self.handles = defaultdict(dict)
        self.forward_hooks = defaultdict(list)
        self.forward_pre_hooks = defaultdict(list)

    def reset_hooks(self):
        self.forward_hooks = defaultdict(list)
        self.forward_pre_hooks = defaultdict(list)

    def register_hooks(self, *args, **kwargs):
        """ CallbackModules.forward -> register_hooks -> TTS.forward """
        pass

    def remove_hooks(self):
        # for h in self.handles[id(self)]:
        #     h.remove()
        # del self.handles[id(self)]
        for key in self.forward_hooks:
            for handle in self.forward_hooks[key]:
                handle.remove()
        for key in self.forward_pre_hooks:
            for handle in self.forward_pre_hooks[key]:
                handle.remove()
        self.reset_hooks()


class EmbeddingTransform(nn.Module):
    def __init__(self, d_emb, d_hidden):
        super().__init__()
        self.d_hidden = d_hidden
        self.linear = nn.Linear(d_emb, 2*d_hidden)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.ones_(self.linear.bias.reshape(2, d_hidden)[0])

    def forward(self, _emb):
        # psd_emb: (B, d_emb)
        output = self.linear(_emb)
        output = output.reshape(-1, 2, self.d_hidden)
        # (B, mean/std, d_hidden)
        return output
