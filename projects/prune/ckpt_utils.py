import sys
import os
current_dir =  os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(current_dir + "/../../")
sys.path.insert(0, root_dir)

from collections import OrderedDict
import torch
import json
from lightning.model import FastSpeech2


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
    new_orig = resolve_tensor(orig, mask)
    print(new_orig)
    print(new_orig.shape)

def logits_to_prob_hist(ckpt):
    logits = torch.nn.utils.parameters_to_vector(ckpt["lam_state_dict"]["logits"].values())
    probs = torch.distributions.utils.logits_to_probs(logits, is_binary=True)
    hist = torch.histc(probs, min=0, max=1, bins=20)
    return hist / torch.sum(hist)

def resolve(model, ckpt):

    def resolve_linear(n, model, ckpt):
        state_dict = OrderedDict()

        weight_orig = ckpt["model_state_dict"]["orig"][f"model.{n}.weight_orig"]
        weight_mask = ckpt["model_state_dict"]["mask"][f"model.{n}.weight_mask"]
        keep_layer = weight_mask.sum() > 0
        if keep_layer:
            new_weight = resolve_tensor(weight_orig, weight_mask)
        else:
            #TODO: remove layer
            # current: zero-out but keep weight
            new_weight = weight_orig * weight_mask
        state_dict["weight"] = new_weight

        bias_orig = ckpt["model_state_dict"]["orig"][f"model.{n}.bias_orig"]
        bias_mask = ckpt["model_state_dict"]["mask"][f"model.{n}.bias_mask"]
        use_bias = bias_mask.sum() > 0
        if use_bias:
            new_bias = resolve_tensor(bias_orig, bias_mask)
            state_dict["bias"] = new_bias

        new_linear = torch.nn.Linear(
            new_weight.shape[1], new_weight.shape[0],
            bias=use_bias,
            device=new_weight.device
        )
        assert new_weight.shape[0] > 0 and new_weight.shape[1] > 0, f"{n}, {new_weight.shape}\n{weight_mask}\n{new_bias.shape}"
        print(new_linear)
        new_linear.load_state_dict(state_dict)

        if len(n.split('.')) > 1:
            parent_name, name = n.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, name, new_linear)
        else:
            setattr(model, n, new_linear)

    # TODO: spk_emb_w_avg_weight_orig, spk_emb_w_avg_subset_logits
    for n, m in list(model.named_modules()):
        if isinstance(m, torch.nn.Linear):
            print(n, "linear")
            # weight, bias
            assert f"model.{n}.weight_mask" in ckpt["model_state_dict"]["mask"] \
                    and f"model.{n}.weight_orig" in ckpt["model_state_dict"]["orig"] \
                    and f"model.{n}.bias_mask" in ckpt["model_state_dict"]["mask"] \
                    and f"model.{n}.bias_orig" in ckpt["model_state_dict"]["orig"]

            if n.startswith('encoder.layer_stack.1.slf_attn'):
                print(n)
                weight_mask = ckpt["model_state_dict"]["mask"][f"model.{n}.weight_mask"]
                bias_mask = ckpt["model_state_dict"]["mask"][f"model.{n}.bias_mask"]
                print(weight_mask.sum())
                print(bias_mask.sum())

            continue

        elif isinstance(m, torch.nn.Embedding):
            continue
            print(n, "embedding")
            # weight
        elif isinstance(m, torch.nn.BatchNorm1d):
            print(n, "BN1d")
            # weight, bias
            # running_mean, running_var, num_batches_tracked
            print(m.state_dict().keys())
            for key in ckpt["model_state_dict"]["mask"]:
                if n in key:
                    print(key)
            # assert f"model.{n}.weight_mask" in ckpt["model_state_dict"]["mask"] \
            #         and f"model.{n}.weight_orig" in ckpt["model_state_dict"]["orig"] \
            #         and f"model.{n}.bias_mask" in ckpt["model_state_dict"]["mask"] \
            #         and f"model.{n}.bias_orig" in ckpt["model_state_dict"]["orig"] \
            #         and f"model.{n}.running_mean_mask" in ckpt["model_state_dict"]["mask"] \
            #         and f"model.{n}.running_mean_orig" in ckpt["model_state_dict"]["orig"] \
            #         and f"model.{n}.running_var_mask" in ckpt["model_state_dict"]["mask"] \
            #         and f"model.{n}.running_var_orig" in ckpt["model_state_dict"]["orig"]
            continue
        elif isinstance(m, torch.nn.Conv1d):
            continue
            print(n, "conv1d")
            # weight, bias
        elif isinstance(m, torch.nn.LayerNorm):
            continue
            print(n, "LN")
            # weight, bias
        else:
            # dropout, softmax, relu
            continue
        print(m)
        params = list(zip(*list(m.named_parameters())))
        buffs = list(zip(*list(m.named_buffers())))
        if len(params) > 0:
            print(params[0])
        if len(buffs) > 0:
            print(buffs[0])
        print()


def resolve_tensor(orig, mask):
    try:
        assert orig.shape == mask.shape
    except:
        mask = mask.expand(orig.shape)

    if mask.sum() == 0:
        #TODO
        pass
    ndim = len(mask.shape)
    new_orig = orig
    if ndim > 1:
        for i in range(ndim):
            dim = [j for j in range(ndim) if j != i]
            new_orig = new_orig.index_select(
                i,
                mask.sum(dim=dim).nonzero().flatten()
            )
    else:
        new_orig = new_orig.masked_select(mask.bool())
    return new_orig

if __name__ == "__main__":
    ckpt =\
    torch.load("output/learnable_structured/p251/lightning_logs/version_5/checkpoints/epoch=8-step=1815.ckpt")
    # torch.load("output/learnable_structured/p251/lightning_logs/version_4/checkpoints/epoch=10-step=2222.ckpt")
    print_ckpt_stats(ckpt)
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
