import os
import numpy as np
import torch
import yaml

from lightning.model import FastSpeech2
from utils.similarity import dot_product_similarity


ckpt_path = "./logs/epoch=31-step=31999.ckpt"
feats_path = "./logs/test-clean_phoneme-features.npy"

checkpoint = torch.load(ckpt_path)
codebook = checkpoint["state_dict"]["model.emb_generator.banks"].cpu().numpy()

phoneme_features = np.load(feats_path)
phoneme_features = phoneme_features[np.sum(phoneme_features, axis=1) != 0]
print(phoneme_features.shape)
similarity_matrix = dot_product_similarity(
    codebook, phoneme_features.T)  # [codebook_size, n_symbols]

nearest_codes = np.argmax(similarity_matrix, axis=0)  # [n_symbols]
print(nearest_codes)
