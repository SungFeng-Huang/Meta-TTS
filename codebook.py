import os
import numpy as np
import torch
import yaml

from lightning.model import FastSpeech2


# ckpt_path = "./logs/epoch=30-step=30999.ckpt"
# # Read Config
# preprocess_configs = [yaml.load(
#     open(path, "r"), Loader=yaml.FullLoader
# ) for path in ['config/preprocess/LibriTTS.yaml']]
# model_config = yaml.load(
#     open('config/model/ml_base.yaml', "r"), Loader=yaml.FullLoader)
# algorithm_config = yaml.load(
#     open('config/algorithm/ml_meta_dotsim_va_d.yaml', "r"), Loader=yaml.FullLoader)
# model = FastSpeech2(
#     algorithm_config["adapt"]["speaker_emb"], preprocess_configs[0], model_config)

# checkpoint = torch.load(ckpt_path)
# for k in checkpoint["state_dict"]:
#     if k == "model.emb_generator.banks":
#         print("Name", k)
#         print(checkpoint["state_dict"][k].shape)
#         with open(f"{os.path.splitext(ckpt_path)[0]}-banks.npy", 'wb')as f:
#             np.save(f, checkpoint["state_dict"][k].cpu().numpy())

a = np.load("./logs/epoch=20-step=20999-banks.npy")
b = np.load("./logs/epoch=30-step=30999-banks.npy")
print(a-b)
