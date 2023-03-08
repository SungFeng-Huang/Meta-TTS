import sys
import os
import torch
import pytest
import unittest

# print(os.environ["PYTHONPATH"])
# current_dir =  os.path.abspath(os.path.dirname(__file__))
# root_dir = os.path.abspath(current_dir + "/../")
# sys.path.insert(0, root_dir)

from torch.testing import assert_allclose, assert_close
from collections import OrderedDict

from lightning.model import FastSpeech2
from projects.prune.ckpt_utils import mask_model, prune_model
from projects.prune.pruned_transformer.SubLayers import PrunedLinear, PrunedConv1d, PrunedLayerNorm
from src.utils.tools import load_yaml


@pytest.mark.parametrize(
    ("d_in_ratio", "d_out_ratio"),
    [
        (0, 0),
        (0, 0.3),
        (0.2, 0),
        (0.2, 0.3),
    ]
)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize(
    "device",
    [
        'cpu',
        pytest.param(
            'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU was detected')
        ),
    ]
)
def test_prune_linear(d_in_ratio: float, d_out_ratio: float, use_bias: bool, device: str):
    d_in, d_out = 256, 256

    d_in_mask = torch.rand(d_in, device=torch.device(device)) < d_in_ratio
    d_out_mask = torch.rand(d_out, device=torch.device(device)) < d_out_ratio
    d_in_index = d_in_mask.nonzero().flatten()
    d_out_index = d_out_mask.nonzero().flatten()
    pruned_d_in = len(d_in_index)
    pruned_d_out = len(d_out_index)

    linear = torch.nn.Linear(d_in, d_out, device=torch.device(device))
    input = torch.randn(4, d_in, device=torch.device(device))
    pruned_input = input.index_select(1, d_in_index)

    state_dict = OrderedDict()
    w = (linear.state_dict()["weight"]
         .index_select(0, d_out_index)
         .index_select(1, d_in_index)
         .reshape(pruned_d_out, pruned_d_in))
    state_dict["weight"] = w
    if use_bias:
        b = (linear.state_dict()["bias"]
             .index_select(0, d_out_index)
             .reshape(pruned_d_out))
        state_dict["bias"] = b

    new_linear = PrunedLinear(pruned_d_in, pruned_d_out, bias=use_bias, device=torch.device(device))
    new_linear.load_state_dict(state_dict)

    masked_state_dict = linear.state_dict()
    masked_state_dict["weight"] = masked_state_dict["weight"] * d_out_mask.reshape(-1, 1) * d_in_mask.reshape(1, -1)
    if use_bias:
        masked_state_dict["bias"] = masked_state_dict["bias"] * d_out_mask
    else:
        masked_state_dict["bias"] = masked_state_dict["bias"] * 0
    linear.load_state_dict(masked_state_dict)

    assert_allclose(linear(input).index_select(1, d_out_index), new_linear(pruned_input))


@pytest.mark.parametrize(
    ("d_in_ratio", "d_out_ratio"),
    [
        (0, 0),
        (0, 0.3),
        (0.2, 0),
        (0.2, 0.3),
    ]
)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize(
    "device",
    [
        'cpu',
        pytest.param(
            'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU was detected')
        ),
    ]
)
def test_prune_conv1d(d_in_ratio: float, d_out_ratio: float, use_bias: bool, device: str):
    d_in, d_out = 256, 256
    kernel_size = 9
    padding = (kernel_size-1)//2

    d_in_mask = torch.rand(d_in, device=torch.device(device)) < d_in_ratio
    d_out_mask = torch.rand(d_out, device=torch.device(device)) < d_out_ratio
    d_in_index = d_in_mask.nonzero().flatten()
    d_out_index = d_out_mask.nonzero().flatten()
    pruned_d_in = len(d_in_index)
    pruned_d_out = len(d_out_index)

    conv1d = torch.nn.Conv1d(
        d_in, d_out,
        bias=use_bias,
        kernel_size=kernel_size,
        padding=padding,
        device=torch.device(device),
    )
    input = torch.randn(4, d_in, 32, device=torch.device(device))
    pruned_input = input.index_select(1, d_in_index)

    state_dict = OrderedDict()
    w = (conv1d.state_dict()["weight"]
         .index_select(0, d_out_index)
         .index_select(1, d_in_index)
         .reshape(pruned_d_out, pruned_d_in, kernel_size))
    state_dict["weight"] = w
    if use_bias:
        b = (conv1d.state_dict()["bias"]
             .index_select(0, d_out_index)
             .reshape(pruned_d_out))
        state_dict["bias"] = b

    new_conv1d = PrunedConv1d(
        pruned_d_in, pruned_d_out,
        bias=use_bias,
        kernel_size=kernel_size,
        padding=padding,
        device=torch.device(device),
    )
    new_conv1d.load_state_dict(state_dict)

    masked_state_dict = conv1d.state_dict()
    masked_state_dict["weight"] = masked_state_dict["weight"] * d_out_mask.reshape(-1, 1, 1) * d_in_mask.reshape(1, -1, 1)
    if use_bias:
        masked_state_dict["bias"] = masked_state_dict["bias"] * d_out_mask
    conv1d.load_state_dict(masked_state_dict)

    assert_allclose(conv1d(input).index_select(1, d_out_index), new_conv1d(pruned_input))


@pytest.mark.parametrize("d_out_ratio", [0, 0.2, 1])
@pytest.mark.parametrize(
    "device",
    [
        'cpu',
        pytest.param(
            'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU was detected')
        ),
    ]
)
def test_prune_LN(d_out_ratio: float, device: str):
    d_out = 256

    d_out_mask = torch.rand(d_out, device=torch.device(device)) < d_out_ratio
    d_out_index = d_out_mask.nonzero().flatten()
    pruned_d_out = len(d_out_index)

    layer_norm = torch.nn.LayerNorm(d_out, device=torch.device(device))
    input = torch.randn(4, d_out, device=torch.device(device))
    pruned_input = input.index_select(1, d_out_index)
    masked_input = input * d_out_mask

    w = torch.randn(256, device=torch.device(device))
    b = torch.randn(256, device=torch.device(device))
    layer_norm.load_state_dict({"weight": w, "bias": b})

    pruned_state_dict = OrderedDict({
        "weight": w.index_select(0, d_out_index),
        "bias": b.index_select(0, d_out_index),
    })
    new_layer_norm = PrunedLayerNorm(d_orig=d_out, normalized_shape=pruned_d_out, device=torch.device(device))
    new_layer_norm.load_state_dict(pruned_state_dict)

    masked_state_dict = OrderedDict({
        "weight": w * d_out_mask,
        "bias": b * d_out_mask,
    })
    layer_norm.load_state_dict(masked_state_dict)

    assert_allclose(layer_norm(masked_input).index_select(1, d_out_index), new_layer_norm(pruned_input))


# @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU was detected')
@pytest.mark.parametrize(
    "ckpt_path",
    ["output/learnable_structured/p251/lightning_logs/version_5/checkpoints/epoch=8-step=1815.ckpt",
    "output/learnable_structured/p251/lightning_logs/version_4/checkpoints/epoch=10-step=2222.ckpt"]
)
def test_prune(ckpt_path: str):
    ckpt = torch.load(ckpt_path)

    device = ckpt["state_dict"]["spk_emb_w_avg_weight_orig"].device

    # [1] * 2311 + [0] * ...
    spk_emb_w_avg_weight_orig = ckpt["state_dict"]["spk_emb_w_avg_weight_orig"]
    assert torch.equal(spk_emb_w_avg_weight_orig[:2311], torch.ones(2311, device=device))

    # sigmoid -> [{0,1}] * 2311 + [0] * ...
    spk_emb_w_avg_subset_logits = ckpt["state_dict"]["spk_emb_w_avg_subset_logits"]

    preprocess_yaml: str = ckpt["hyper_parameters"]["preprocess_config"]
    model_yaml: str = ckpt["hyper_parameters"]["model_config"]
    algorithm_yaml: str = ckpt["hyper_parameters"]["algorithm_config"]

    model = FastSpeech2(
        load_yaml(preprocess_yaml),
        load_yaml(model_yaml),
        load_yaml(algorithm_yaml)
    ).cuda()

    masked_model = mask_model(model, ckpt).eval()
    pruned_model = prune_model(model, ckpt).eval()
    del model

    input = {
        "speaker_args": torch.zeros((5,)).long().cuda(),
        "texts": torch.ones((5, 20)).long().cuda(),
        "src_lens": torch.arange(16, 21).long().cuda(),
        "max_src_len": torch.tensor(20).long().cuda(),
        # "d_targets": torch.randint(1,20,(5,20)).long().cuda(),
        # "p_targets": torch.randn((5,20)).long().cuda(),
        # "e_targets": torch.randn((5,20)).long().cuda(),
    }

    masked_output = masked_model(**input)
    pruned_output = pruned_model(**input)
    for mo, po in zip(masked_output, pruned_output):
        # some difference would come to 1.0??e-6
        assert_close(
            mo, po, rtol=1e-5, atol=1.1e-6,
            # msg=f"mask: {mo[~torch.isclose(mo, po, rtol=1e-5, atol=1.1e-6)]}\nprune: {po[~torch.isclose(mo, po, rtol=1e-5, atol=1.1e-6)]}\n{(mo-po)[~torch.isclose(mo, po, rtol=1e-5, atol=1.1e-6)]}"
        )


# def _test_prune_LN(d_out_ratio: float, device: str):
