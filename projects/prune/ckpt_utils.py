import sys
import os

current_dir =  os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(current_dir + "/../../")
sys.path.insert(0, root_dir)

from collections import OrderedDict
from copy import deepcopy
import torch
import json
from lightning.model import FastSpeech2
from lightning.model.modules import VarianceAdaptor
from lightning.model.speaker_encoder import SpeakerEncoder
from src.transformer.Models import Encoder, Decoder, FFTBlock
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
    # print(ckpt["model_state_dict"]["mask"]["model.postnet.convolutions.2.0.conv.weight_mask"])
    # key = "model.postnet.convolutions.2.0.conv.weight"
    # mask = ckpt["model_state_dict"]["mask"][f"{key}_mask"]
    # print(torch.nonzero(torch.sum(mask, dim=(1,2))), mask.shape[0])
    # print(torch.nonzero(torch.sum(mask, dim=(0,2))), mask.shape[1])
    # print(torch.nonzero(torch.sum(mask, dim=(0,1))), mask.shape[2])
    # print(torch.nonzero(mask, as_tuple=True))
    # orig = ckpt["model_state_dict"]["orig"][f"{key}_orig"]

def logits_to_prob_hist(ckpt):
    logits = torch.nn.utils.parameters_to_vector(ckpt["lam_state_dict"]["logits"].values())
    probs = torch.distributions.utils.logits_to_probs(logits, is_binary=True)
    hist = torch.histc(probs, min=0, max=1, bins=20)
    return hist / torch.sum(hist)

def mask_model(
    model: FastSpeech2,
    ckpt: dict[str, dict[str, dict[str, torch.Tensor]]],
):
    model = deepcopy(model)
    state_dict = OrderedDict()
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        if f"model.{n}" in ckpt["model_state_dict"]["others"]:
            # pitch_bins, energy_bins, BN_running_mean/var/num_batches_tracked
            state_dict[n] = ckpt["model_state_dict"]["others"][f"model.{n}"]
        else:
            assert f"model.{n}_mask" in ckpt["model_state_dict"]["mask"], n
            assert f"model.{n}_orig" in ckpt["model_state_dict"]["orig"], n
            state_dict[n] = (ckpt["model_state_dict"]["mask"][f"model.{n}_mask"].bool()
                             * ckpt["model_state_dict"]["orig"][f"model.{n}_orig"])
    model.load_state_dict(state_dict)
    return model

def prune_model(
    model: FastSpeech2,
    ckpt: dict[str, dict[str, dict[str, torch.Tensor]]],
):
    # NOTE: It's impossible to dirrectly prune leaf nodes without knowing its adjacent layers.
    # Reason:
    # 1. When a layer's weight params are all 0, we won't know whether it's because of d_in=0 or d_out=0 or both.
    # 2. When a layer's bias params are all 0, we won't know whether it's because of d_out=0 or use_bias=False.
    # -> When a layer's weight/bias are all 0, we won't know any of 'd_in', 'd_out', 'use_bias'.

    # TODO: spk_emb_w_avg_weight_orig, spk_emb_w_avg_subset_logits
    model = deepcopy(model)
    for n, c in list(model.named_children()):
        # encoder.src_word_emb
        if isinstance(c, Encoder):
            mask = ckpt["model_state_dict"]["mask"][f"model.{n}.src_word_emb.weight_mask"]
            orig = ckpt["model_state_dict"]["orig"][f"model.{n}.src_word_emb.weight_orig"]
            d_index = mask.sum(dim=[0]).nonzero().flatten()
            new_emb = orig.index_select(1, d_index)
            c.src_word_emb = torch.nn.Embedding(
                new_emb.shape[0], new_emb.shape[1],
                padding_idx=c.src_word_emb.padding_idx,
                _weight=new_emb,
                device=new_emb.device,
            )

        # {encoder,decoder}.position_enc
        # {encoder,decoder}.layer_stack
        if isinstance(c, Encoder) or isinstance(c, Decoder):
            # position_enc
            mask = ckpt["model_state_dict"]["mask"][f"model.{n}.position_enc_mask"]
            orig = ckpt["model_state_dict"]["orig"][f"model.{n}.position_enc_orig"]
            d_index = mask.sum(dim=[0,1]).nonzero().flatten()
            new_pos_enc = orig.index_select(2, d_index)
            c.position_enc = torch.nn.Parameter(new_pos_enc, requires_grad=False)

            # layer_stack
            for i, block in enumerate(c.layer_stack):
                assert isinstance(block, FFTBlock)
                new_mha = PrunedMultiHeadAttention(orig=block.slf_attn)
                new_mha.prune(f"{n}.layer_stack.{i}.slf_attn", ckpt["model_state_dict"])
                block.slf_attn = new_mha
                new_ffn = PrunedPositionwiseFeedForward(orig=block.pos_ffn)
                new_ffn.prune(f"{n}.layer_stack.{i}.pos_ffn", ckpt["model_state_dict"])
                block.pos_ffn = new_ffn

        if isinstance(c, VarianceAdaptor):
            # {duration,pitch,energy}_predictor
            for psd in {"duration", "pitch", "energy"}:
                orig = getattr(c, f"{psd}_predictor", None)
                assert isinstance(orig, VariancePredictor)
                new_psd_predictor = PrunedVariancePredictor(orig=orig)
                new_psd_predictor.prune(f"{n}.{psd}_predictor", ckpt["model_state_dict"])
                setattr(
                    c, f"{psd}_predictor", new_psd_predictor
                )

            # pitch_embedding, energy_embedding
            for psd in {"pitch", "energy"}:
                mask = ckpt["model_state_dict"]["mask"][f"model.{n}.{psd}_embedding.weight_mask"]
                orig = ckpt["model_state_dict"]["orig"][f"model.{n}.{psd}_embedding.weight_orig"]
                d_index = mask.sum(dim=[0]).nonzero().flatten()
                new_psd_emb = orig.index_select(1, d_index)
                setattr(
                    c, f"{psd}_embedding",
                    torch.nn.Embedding(
                        new_psd_emb.shape[0], new_psd_emb.shape[1],
                        _weight=new_psd_emb,
                        device=new_psd_emb.device,
                    )
                )

        # Mel-linear
        if isinstance(c, torch.nn.Linear):
            assert n == "mel_linear"
            w_mask = ckpt["model_state_dict"]["mask"][f"model.{n}.weight_mask"]
            w_orig = ckpt["model_state_dict"]["orig"][f"model.{n}.weight_orig"]
            assert w_mask.sum() > 0
            d_in_index = w_mask.sum(dim=[0]).nonzero().flatten()
            d_out_index = w_mask.sum(dim=[1]).nonzero().flatten()
            new_w = w_orig.index_select(0, d_out_index).index_select(1, d_in_index)

            b_mask = ckpt["model_state_dict"]["mask"][f"model.{n}.bias_mask"]
            b_orig = ckpt["model_state_dict"]["orig"][f"model.{n}.bias_orig"]
            new_b = (b_orig*b_mask).index_select(0, d_out_index)
            new_lin = torch.nn.Linear(len(d_in_index), len(d_out_index), device=new_b.device)
            new_lin.load_state_dict({"weight": new_w, "bias": new_b})
            model.mel_linear = new_lin
        
        # PostNet
        if isinstance(c, PostNet):
            new_c = PrunedPostNet(orig=c)
            new_c.prune(n, ckpt["model_state_dict"])
            model.postnet = new_c

        # Speaker encoder
        if isinstance(c, SpeakerEncoder):
            # Only using speaker embedding (speaker encoder not supported)
            #TODO
            assert n == "speaker_emb"
            mask = ckpt["model_state_dict"]["mask"][f"model.{n}.model.weight_mask"]
            orig = ckpt["model_state_dict"]["orig"][f"model.{n}.model.weight_orig"]
            d_index = mask.sum(dim=[0]).nonzero().flatten()
            new_emb = orig.index_select(1, d_index)
            c.model = torch.nn.Embedding(
                new_emb.shape[0], new_emb.shape[1],
                _weight=new_emb,
                device=new_emb.device,
            )

    return model


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

    masked_model = mask_model(model, ckpt).eval()
    pruned_model = prune_model(model, ckpt).eval()
    del model
    # print(model)

    # masked_params = list(masked_model.named_parameters()) + list(masked_model.named_buffers())
    # pruned_params = list(pruned_model.named_parameters()) + list(pruned_model.named_buffers())
    masked_params = list(masked_model.named_parameters())
    pruned_params = list(pruned_model.named_parameters())
    # for (nm, mp), (np, pp) in zip(masked_params, pruned_params):
    #     assert nm == np
    #     if np.startswith("variance_adaptor"):
    #         print(nm)
    #         print(mp, mp.shape)
    #         print(pp, pp.shape)
    #         print()

    # for n, c in masked_model.named_children():
    #     print(n)

    # print(list(masked_model.encoder.named_children()))

    # exit()

    input = {
        "speaker_args": torch.zeros((5,)).long().cuda(),
        "texts": torch.ones((5, 20)).long().cuda(),
        "src_lens": torch.arange(16, 21).long().cuda(),
        "max_src_len": torch.tensor(20).long().cuda(),
        # "d_targets": torch.randint(1,20,(5,20)).long().cuda(),
        # "p_targets": torch.randn((5,20)).long().cuda(),
        # "e_targets": torch.randn((5,20)).long().cuda(),
    }
    # for i in range(5):
    #     input["d_targets"][i, 16+i:] = 0
    #     input["p_targets"][i, 16+i:] = 0
    #     input["e_targets"][i, 16+i:] = 0

    # buffer: dict[str, dict[str, torch.Tensor]] = {"inputs": {}, "outputs": {}}
    # def mask_hook(module, inputs, outputs):
    #     buffer["inputs"]["mask"] = inputs
    #     buffer["outputs"]["mask"] = outputs
    # def prune_hook(module, inputs, outputs):
    #     buffer["inputs"]["prune"] = inputs
    #     buffer["outputs"]["prune"] = outputs
    # submodule = "variance_adaptor.duration_predictor.conv_layer.layer_norm_1"
    # masked_model.get_submodule(submodule).register_forward_hook(mask_hook)
    # pruned_model.get_submodule(submodule).register_forward_hook(prune_hook)
    # mw = masked_model.get_submodule(submodule).weight
    # mb = masked_model.get_submodule(submodule).bias
    # pw = pruned_model.get_submodule(submodule).weight
    # pb = pruned_model.get_submodule(submodule).bias
    # print("mask", mw, mw.shape)
    # print("prune", pw, pw.shape)
    # print("mask", mb, mb.shape)
    # print("prune", pb, pb.shape)

    masked_output = masked_model(**input)
    pruned_output = pruned_model(**input)
    # for key in buffer:
    #     for mo, po in zip(buffer[key]["mask"], buffer[key]["prune"]):
    #         # print(key, "mask", mo[mo.nonzero(as_tuple=True)], len(mo.nonzero()))
    #         # print(key, "prune", po[po.nonzero(as_tuple=True)], po.shape, len(po.nonzero()))
    #         d_m_index = mo.sum(0).nonzero().flatten()
    #         # new_mo = mo.index_select(1, d_m_index)
    #         # print(key, "mask", new_mo, new_mo.shape)
    #         print(mo.sum(0).nonzero().shape)
    #         print(key, "mask", mo, mo.shape)
    #         print(key, "prune", po, po.shape)
    #         print()
    for mo, po in zip(masked_output, pruned_output):
        print(torch.allclose(mo, po, atol=1.1e-6))
        if not torch.allclose(mo, po, atol=1.1e-6):
            print(mo)
            print(po)
            print(mo[~torch.isclose(mo, po, atol=1.1e-6)])
            print(po[~torch.isclose(mo, po, atol=1.1e-6)])
            print((mo-po)[~torch.isclose(mo, po, atol=1.1e-6)])
            # mod = mo[mo != po]
            # pod = po[mo != po]
            # print(torch.abs((mod-pod)/mod))
        print()