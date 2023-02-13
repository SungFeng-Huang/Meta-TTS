import sys
import os

current_dir =  os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(current_dir + "/../../../")
# sys.path.insert(0, root_dir)
assert root_dir in os.environ['PYTHONPATH']

import time
from tqdm import tqdm
from copy import deepcopy
from glob import glob

import torch
from torch.utils.data import DataLoader, Subset
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_lightning.utilities import move_data_to_device
from lightning.dataset.text import TextDataset
from lightning.model import FastSpeech2
from projects.prune.ckpt_utils import mask_model, prune_model
from src.utils.tools import load_yaml


def last_stage_ckpt_prefix(dir, stage: str):
    stage_dir = {
        "libri_mask": "mask",
        "mask": "mask",
        "FT": "sup",
        "joint": "sup",
    }[stage]
    csv_name = sorted(
        os.listdir(f"{dir}/train/csv/{stage_dir}"),
        key=lambda x: int(x.split('.')[0].split('=')[-1])
    )[-1]
    epoch_prefix = csv_name[:-4]
    return epoch_prefix

def get_newest_ckpt_path(spk: str, pipeline: str, stage: str):
    pipeline_dir = f"output/learnable_structured_pipeline/{spk}/{pipeline}/lightning_logs"
    ver = sorted(
        os.listdir(pipeline_dir),
        key=lambda x: int(x.split('_')[-1])
    )[-1]
    dir = f"{pipeline_dir}/{ver}/fit"
    prefix = last_stage_ckpt_prefix(dir, stage)
    ckpt_path = glob(f"{pipeline_dir}/{ver}/checkpoints/{prefix}-step=*.ckpt")[0]
    return ckpt_path


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir(root_dir)
    print(os.getcwd())

    print("load ckpt")
    ckpt = torch.load("output/learnable_structured/p251/lightning_logs/version_5/checkpoints/epoch=8-step=1815.ckpt")

    preprocess_yaml: str = ckpt["hyper_parameters"]["preprocess_config"]
    model_yaml: str = ckpt["hyper_parameters"]["model_config"]
    algorithm_yaml: str = ckpt["hyper_parameters"]["algorithm_config"]

    dataset: TextDataset = TextDataset("preprocessed_data/LibriTTS_VCTK/total.txt", load_yaml(preprocess_yaml))
    subset_ids = [i for i, spk in enumerate(dataset.speaker) if spk=="p251"]
    subset_dataset = Subset(dataset, subset_ids)
    dataloader = DataLoader(subset_dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=8, collate_fn=dataset.dict_collate)

    print("construct model")
    # random-init model
    model: FastSpeech2 = FastSpeech2(
        load_yaml(preprocess_yaml),
        load_yaml(model_yaml),
        load_yaml(algorithm_yaml)
    ).eval().cuda()

    print("mask model")
    masked_model = mask_model(model, ckpt).eval()
    print("prune model")
    pruned_model = prune_model(model, ckpt).eval()
    
    pretrain_ckpt = torch.load(ckpt["hyper_parameters"]["ckpt_path"])
    model.load_state_dict({
        k[6:]: v
        for k, v in pretrain_ckpt["state_dict"].items()
        if k.startswith("model.")
    })
    with torch.no_grad():
        try:
            model.speaker_emb.model.weight[2311:] = \
                model.speaker_emb.model.weight[:2311].mean(dim=0)
        except:
            model.speaker_emb.model.weight_orig[2311:] = \
                model.speaker_emb.model.weight_orig[:2311].mean(dim=0)

    test_models = {
        "full": model,
        "mask": masked_model,
        "prune": pruned_model,
    }

    print("Dry run")
    for data in tqdm(dataloader):
        data = move_data_to_device(data, model.device)
        model(**data)
        masked_model(**data)
        pruned_model(**data)

    memory_peak = {
        **{k: [] for k in test_models},
    }
    nparams = {
        k: sum(p.numel() for p in m.parameters()) for k, m in test_models.items()
    }
    with profile(
        profile_memory=True,
        with_modules=True,
    ) as prof:
        for i in range(2):
            for data in tqdm(dataloader):
                data = move_data_to_device(data, model.device)
                for mode in test_models:
                    torch.cuda.reset_peak_memory_stats(model.device)
                    test_model = test_models[mode]
                    with record_function(f"model_inference-{mode}"):
                        test_model(**data)
                        prof.step()
                    peak_mem = torch.cuda.max_memory_allocated(model.device)
                    memory_peak[mode].append(peak_mem)
    print(prof.key_averages().table(sort_by="cuda_time_total", top_level_events_only=True, row_limit=len(test_models)))
    for key in nparams:
        print(key)
        print(f"\tMax: {max(memory_peak[key])/1024/1024/1024:.3f}GB")
        print(f"\tAvg: {sum(memory_peak[key])/len(memory_peak[key])/1024/1024/1024:.3f}GB")
        print(f"\tnParams: {nparams[key]/1024/1024:.3f}MB")