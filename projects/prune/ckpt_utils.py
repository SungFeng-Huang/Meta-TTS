import sys
import os
import argparse

from collections import OrderedDict
from copy import deepcopy
import torch
from lightning.model import FastSpeech2
from lightning.model.modules import VarianceAdaptor
from lightning.model.speaker_encoder import SpeakerEncoder
from src.transformer.Models import Encoder, Decoder, FFTBlock
from projects.prune.pruned_transformer import \
    PrunedMultiHeadAttention, PrunedPositionwiseFeedForward, \
    PostNet, VariancePredictor, \
    PrunedPostNet, PrunedVariancePredictor


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

        # {encoder,decoder}.{position_enc,layer_stack}
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
    from src.utils.tools import load_yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path)

    preprocess_yaml: str = ckpt["hyper_parameters"]["preprocess_config"]
    model_yaml: str = ckpt["hyper_parameters"]["model_config"]
    algorithm_yaml: str = ckpt["hyper_parameters"]["algorithm_config"]

    model = FastSpeech2(
        load_yaml(preprocess_yaml),
        load_yaml(model_yaml),
        load_yaml(algorithm_yaml)
    ).cuda()

    pruned_model = prune_model(model, ckpt).eval()
    output_path = args.output_path or ckpt_path.replace(".ckpt", "_pruned.pth")
    torch.save(pruned_model, output_path)