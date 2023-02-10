import sys
import os

current_dir =  os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(current_dir + "/../../")
sys.path.insert(0, root_dir)

from collections import OrderedDict
import torch
import json
from lightning.model import FastSpeech2
from projects.prune.pruned_transformer import \
    MultiHeadAttention, PositionwiseFeedForward, \
    PrunedMultiHeadAttention, PrunedPositionwiseFeedForward, \
    PostNet, VariancePredictor, \
    PrunedPostNet, PrunedVariancePredictor


def print_ckpt_stats(ckpt):
    print(ckpt.keys())
    for key in ckpt:
        if key == "state_dict":
            print(key, ckpt[key].keys())
        elif key == "optimizer_states":
            continue
        elif key == "lam_state_dict":
            print(key, ckpt[key].keys())
        elif key == "model_state_dict":
            print(key, ckpt[key].keys())
            for k in ckpt[key]:
                print(key, k, ckpt[key][k].keys())
        else:
            continue
            print(key, ckpt[key])
    print(ckpt["model_state_dict"]["mask"]["model.postnet.convolutions.2.0.conv.weight_mask"])
    key = "model.postnet.convolutions.2.0.conv.weight"
    mask = ckpt["model_state_dict"]["mask"][f"{key}_mask"]
    print(torch.nonzero(torch.sum(mask, dim=(1,2))), mask.shape[0])
    print(torch.nonzero(torch.sum(mask, dim=(0,2))), mask.shape[1])
    print(torch.nonzero(torch.sum(mask, dim=(0,1))), mask.shape[2])
    print(torch.nonzero(mask, as_tuple=True))
    orig = ckpt["model_state_dict"]["orig"][f"{key}_orig"]

def logits_to_prob_hist(ckpt):
    logits = torch.nn.utils.parameters_to_vector(ckpt["lam_state_dict"]["logits"].values())
    probs = torch.distributions.utils.logits_to_probs(logits, is_binary=True)
    hist = torch.histc(probs, min=0, max=1, bins=20)
    return hist / torch.sum(hist)

def resolve(model, ckpt):

    print(model)
    # TODO: spk_emb_w_avg_weight_orig, spk_emb_w_avg_subset_logits
    for n, m in list(model.named_modules()):
        if isinstance(m, MultiHeadAttention):
            new_m = PrunedMultiHeadAttention(orig=m)
            new_m.prune(n, ckpt["model_state_dict"])
        elif isinstance(m, PositionwiseFeedForward):
            new_m = PrunedPositionwiseFeedForward(orig=m)
            new_m.prune(n, ckpt["model_state_dict"])
        elif isinstance(m, PostNet):
            new_m = PrunedPostNet(orig=m)
            new_m.prune(n, ckpt["model_state_dict"])
        elif isinstance(m, VariancePredictor):
            new_m = PrunedVariancePredictor(orig=m)
            new_m.prune(n, ckpt["model_state_dict"])
        else:
            continue

        # print(n)
        # print(m)
        # print(new_m)
        if len(n.split('.')) > 1:
            parent_name, name = n.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, name, new_m)
        else:
            setattr(model, n, new_m)
    print(model)


if __name__ == "__main__":
    ckpt =\
    torch.load("output/learnable_structured/p251/lightning_logs/version_5/checkpoints/epoch=8-step=1815.ckpt")
    # torch.load("output/learnable_structured/p251/lightning_logs/version_4/checkpoints/epoch=10-step=2222.ckpt")
    # print_ckpt_stats(ckpt)
    # print(logits_to_prob_hist(ckpt))

    # [1] * 2311 + [0] * ...
    print(set(ckpt["state_dict"]["spk_emb_w_avg_weight_orig"].cpu().tolist()))
    # sigmoid -> [{0,1}] * 2311 + [0] * ...
    print(torch.sigmoid(ckpt["state_dict"]["spk_emb_w_avg_subset_logits"]).sum())

    from src.utils.tools import load_yaml
    model = FastSpeech2(
        load_yaml("config/preprocess/LibriTTS_VCTK.yaml"),
        load_yaml("config/model/base.yaml"),
        load_yaml("config/algorithm/pretrain/pretrain_LibriTTS.2.yaml")
    ).cuda()

    resolve(model, ckpt)
    # print(model)

    input = {
        "speaker_args": torch.zeros((5,)).long().cuda(),
        "texts": torch.zeros((5, 10)).long().cuda(),
        "src_lens": torch.arange(6, 11).long().cuda(),
        "max_src_len": torch.tensor(10).long().cuda(),
    }
    print(model(**input))