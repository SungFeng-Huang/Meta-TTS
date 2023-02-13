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
from lightning.model import FastSpeech2
from projects.prune.ckpt_utils import mask_model, prune_model
from src.utils.tools import load_yaml

@unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
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