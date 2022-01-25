import os
import numpy as np
import json
from functools import partial

import torch
from torch import nn

import pytorch_lightning as pl

import transformer.Constants as Constants
from transformer import Encoder, Decoder, PostNet
from transformer.Modules import ScaledDotProductAttention
from text.symbols import symbols
from text.define import LANG_ID2SYMBOLS


class PhonemeEmbeddingOld(pl.LightningModule):
    # NOTE:
    #   Tested:
    #       - hard att
    #       - soft att
    # TODO:
    #   - share bank
    #   - embedding mode

    def __init__(self, model_config, algorithm_config):
        """
        All the data members would be moved to cuda together with the whole model
        by pytorch_lightning default settings.
        Directly replacing `model.encoder.src_word_emb` with this class would do
        no harm to the `model.encoder` forwarding.
        """
        super().__init__()
        self.emb_type = algorithm_config["adapt"]["phoneme_emb"]["type"]

        n_src_vocab = len(symbols) + 1
        d_word_vec = model_config["transformer"]["encoder_hidden"]
        self.d_word_vec = d_word_vec

        if self.emb_type == "embedding":
            # TODO
            pass
        
        elif self.emb_type == "codebook":
            self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
            codebook_size = self.codebook_config["size"]

            self.emb_banks = nn.Parameter(torch.randn(codebook_size, d_word_vec))

            att_config = self.codebook_config["attention"]
            d_feat = self.codebook_config["representation_dim"]

            if att_config["type"] == "hard":
                # One-hot similarity

                # feats <-> att_banks -> token_id -> emb_banks
                # TODO: init from SSL-feature centroids of LibriTTS, need to change path here.
                from sklearn.cluster import KMeans
                repr = np.load("preprocessed_data/LibriTTS/train-clean-100_phoneme-features.npy")
                kmeans = KMeans(n_clusters=codebook_size, random_state=0).fit(repr)
                centers = kmeans.cluster_cneters_
                centers = torch.from_numpy(centers)
                self.att_banks = nn.Parameter(centers)

            elif att_config["type"] == "soft":
                # Attention layer
                #   key: att_banks
                #   value: emb_banks
                #   query: refs

                # att(feats, att_banks) -> token_id weights -> emb_banks
                self.att_banks = nn.Parameter(torch.randn(codebook_size, d_word_vec))
                if att_config["share"]:
                    # TPU shared weights are copied independently
                    # on the XLA device and this line won't have any effect.
                    # However, it works fine for CPU and GPU.
                    self.att_banks.weight = self.emb_banks.weight

                self.w_qs = nn.Linear(d_feat, d_word_vec)
                self.w_ks = nn.Linear(d_word_vec, d_word_vec)
                self.attention = ScaledDotProductAttention(
                    temperature=np.power(d_word_vec, 0.5)
                )


    def on_post_move_to_device(self):
        if (hasattr(self, 'codebook_config')
                and self.codebook_config["attention"]["type"] == "soft"
                and self.codebook_config["attention"]["share"]):
            # Weights shared after the model has been moved to TPU Device
            self.att_banks.weight = self.emb_banks.weight


    def get_new_embedding(self, ref):
        """ Compute binary quantize matrix from reference representations.
        Should run before inner_loop update.

        Args:
            ref: Reference representations with size (vocab_size, codebook_size),
                 where vocab_size <= n_src_vocab.
                 Assert the tensor is already moved to cuda by pytorch_lightning.
        """
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)

        if self.codebook_config["attention"]["type"] == "hard":
            ref_norm = ref.norm(dim=1, keepdim=True)
            ref_mask, _ = torch.nonzero(ref_norm, as_tuple=True)
            normed_ref = ref[ref_mask] / ref_norm[ref_mask]

            bank_norms = self.att_banks.norm(dim=1, keepdim=True)
            normed_banks = self.att_banks / bank_norms

            similarity = normed_ref @ normed_banks.T

            with torch.no_grad():
                weighting_matrix = torch.zeros(
                    ref.shape[0], self.att_banks.shape[0],
                    device=self.device
                )
                # NOTE: Can't directly assign = 1 since need to ensure tensors on the same device, use ones_like() instead. 
                weighting_matrix[ref_mask, similarity.argmax(1)] = torch.ones_like(ref_mask).float()
                # padding_idx
                weighting_matrix[Constants.PAD].fill_(0)
            weighted_embedding = weighting_matrix @ self.emb_banks
            return weighted_embedding

        elif self.codebook_config["attention"]["type"] == "soft":
            """ Soft attention weight. Better not share_att_banks.
            key: self.att_banks
            value: self.emb_banks
            query: ref
            """
            q = self.w_qs(ref).view(1, -1, 1, self.d_word_vec)
            k = self.w_ks(self.att_banks).view(1, -1, 1, self.d_word_vec)
            q = q.permute(2, 0, 1, 3).contiguous().view(1, -1, self.d_word_vec)  # 1 x vocab_size x dk
            k = k.permute(2, 0, 1, 3).contiguous().view(1, -1, self.d_word_vec)  # 1 x codebook_size x dk
            v = self.emb_banks.unsqueeze(0)
            weighted_embedding, attn = self.attention(q, k, v)
            with torch.no_grad():
                weighted_embedding[Constants.PAD].fill_(0)
            return weighted_embedding.squeeze(0)


class BaseEmbeddings(pl.LightningModule):
    def __init__(self, model_config, algorithm_config):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        if algorithm_config["type"] in ["baseline"]:
            for lang_id, v in LANG_ID2SYMBOLS.items():
                if len(v) > 0:
                    self.embeddings[f"table-{lang_id}"] = TablePhonemeEmbedding(model_config, algorithm_config, lang_id)
        if algorithm_config["adapt"]["type"] == "spk":  # Multi-speaker, English only
            pass
        elif algorithm_config["adapt"]["type"] == "lang":
            self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
            if self.codebook_config["type"] == "codebook":
                if self.codebook_config["attention"]["type"] in ["hard", "hard-s"]:
                    self.embeddings["hard"] = HardAttCodebook(model_config, algorithm_config)
                    if self.codebook_config["attention"]["type"] == "hard-s":
                        self.embeddings["hard"].soft = True
                elif self.codebook_config["attention"]["type"] == "soft":
                    self.embeddings["soft"] = SoftAttCodebook(model_config, algorithm_config)
                elif self.codebook_config["attention"]["type"] == "soft2":
                    self.embeddings["soft2"] = SoftAttCodebook2(model_config, algorithm_config)
            elif self.codebook_config["type"] == "embedding":
                for lang_id, v in LANG_ID2SYMBOLS.items():
                    if len(v) > 0:
                        self.embeddings[f"table-{lang_id}"] = TablePhonemeEmbedding(model_config, algorithm_config, lang_id)
            else:
                print(self.codebook_config["type"])
                raise NotImplementedError
        else:
            raise NotImplementedError
    

class PhonemeEmbedding(pl.LightningModule):
    def __init__(self, model_config, algorithm_config):
        super().__init__()
        self.get_new_embedding_funcs = {}
        self.hub = BaseEmbeddings(model_config, algorithm_config)
        self.register_func("table", self.get_table)
        self.register_func("table-sep", self.get_table_sep)
        self.register_func("hard", partial(self.get_from_codebook, "hard"))
        self.register_func("soft", partial(self.get_from_codebook, "soft"))
        self.register_func("soft2", partial(self.get_from_codebook, "soft2"))
        self.register_func("hard-s", partial(self.get_from_codebook, "hard"))
        print("PhonemeEmbedding", self.hub)

    def register_func(self, mode, func):
        self.get_new_embedding_funcs[mode] = func

    def get_table(self, *args, **kwargs):
        return torch.cat([self.hub.embeddings[f"table-{lang_id}"].get_new_embedding() 
            for lang_id, v in LANG_ID2SYMBOLS.items() if len(v) > 0], dim=0)

    def get_table_sep(self, *args, **kwargs):
        return self.hub.embeddings[f"table-{kwargs['lang_id']}"].get_new_embedding(init=True)

    def get_from_codebook(self, mode, *args, **kwargs):
        return self.hub.embeddings[mode].get_new_embedding(kwargs["ref_phn_feats"])
    
    def get_new_embedding(self, mode, *args, **kwargs):
        return self.get_new_embedding_funcs[mode](*args, **kwargs)


class TablePhonemeEmbedding(pl.LightningModule):
    def __init__(self, model_config, algorithm_config, lang_id):
        super().__init__()
        self.d_word_vec = model_config["transformer"]["encoder_hidden"]
        self.codebook_size = len(LANG_ID2SYMBOLS[lang_id])
        w_init = torch.randn(self.codebook_size, self.d_word_vec)
        w_init[Constants.PAD].fill_(0)
        self.banks = nn.Parameter(w_init)

    def get_new_embedding(self, init=False):
        if init:
            w_init = torch.randn(self.codebook_size, self.d_word_vec)
            w_init[Constants.PAD].fill_(0)
            return w_init.to(self.device)
        else:
            return self.banks.clone()


class HardAttCodebook(pl.LightningModule):
    def __init__(self, model_config, algorithm_config):
        super().__init__()
        self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
        self.codebook_size = self.codebook_config["size"]
        self.d_word_vec = model_config["transformer"]["encoder_hidden"]

        self.emb_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        d_feat = self.codebook_config["representation_dim"]

        # One-hot similarity
        # feats <-> att_banks -> token_id -> emb_banks, need to change path here.
        centroids = load_hubert_centroids([
            "preprocessed_data/LibriTTS/train-clean-100_phoneme-features.npy",
            "preprocessed_data/AISHELL-3/train_phoneme-features.npy",
            "preprocessed_data/GlobalPhone/fr/train_phoneme-features.npy",
            # "preprocessed_data/GlobalPhone/de/train_phoneme-features.npy",
        ], self.codebook_size)
        self.att_banks = nn.Parameter(centroids)

        self.attention = ScaledDotProductAttention(temperature=0.03)
        self.soft = False

    def get_new_embedding(self, ref, *args, **kwargs):
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)
        ref[ref != ref] = 0

        if not self.soft:
            # normalize
            ref_norm = ref.norm(dim=1, keepdim=True)
            ref_mask, _ = torch.nonzero(ref_norm, as_tuple=True)
            normed_ref = ref[ref_mask] / ref_norm[ref_mask]

            bank_norms = self.att_banks.norm(dim=1, keepdim=True)
            normed_banks = self.att_banks / bank_norms

            similarity = normed_ref @ normed_banks.T

            with torch.no_grad():
                weighting_matrix = torch.zeros(
                    ref.shape[0], self.att_banks.shape[0],
                    device=self.device
                )
                # NOTE: Can't directly assign = 1 since need to ensure tensors on the same device, use ones_like() instead. 
                weighting_matrix[ref_mask, similarity.argmax(1)] = torch.ones_like(ref_mask).float()
                # padding_idx
                weighting_matrix[Constants.PAD].fill_(0)
            weighted_embedding = weighting_matrix @ self.emb_banks
            # print(torch.sum(ref))
            # print(torch.sum(self.att_banks, axis=1))
            return weighted_embedding
        else:
            # normalize
            ref_norms = ref.norm(dim=1, keepdim=True) + 1e-6
            normed_ref = ref / ref_norms

            bank_norms = self.att_banks.norm(dim=1, keepdim=True)
            normed_banks = self.att_banks / bank_norms

            q = normed_ref.unsqueeze(0)  # 1 x vocab_size x drepr
            k = normed_banks.unsqueeze(0)  # 1 x codebook_size x drepr
            v = self.emb_banks.unsqueeze(0)

            weighted_embedding, attn = self.attention(q, k, v)
            with torch.no_grad():
                weighted_embedding[Constants.PAD].fill_(0)
            # print(weighted_embedding.shape)
            # print(normed_ref.shape)
            # print(attn.squeeze().shape)
            # print(torch.max(attn.squeeze(), dim=1))
            # print(torch.sum(ref))
            # print(self.att_banks)
            return weighted_embedding.squeeze(0)


class SoftAttCodebook(pl.LightningModule):
    def __init__(self, model_config, algorithm_config):
        super().__init__()
        # Attention layer
        #   key: att_banks
        #   value: emb_banks
        #   query: refs
        self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
        self.codebook_size = self.codebook_config["size"]
        self.d_word_vec = model_config["transformer"]["encoder_hidden"]

        self.emb_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        att_config = self.codebook_config["attention"]
        d_feat = self.codebook_config["representation_dim"]

        # att(feats, att_banks) -> token_id weights -> emb_banks
        if att_config["share"]:
            # TPU shared weights are copied independently
            # on the XLA device and this line won't have any effect.
            # However, it works fine for CPU and GPU.
            self.att_banks = self.emb_banks
        else:
            self.att_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        self.w_qs = nn.Linear(d_feat, self.d_word_vec)
        self.w_ks = nn.Linear(self.d_word_vec, self.d_word_vec)
        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.d_word_vec, 0.5)
        )

    # def on_post_move_to_device(self):
    #     if self.codebook_config["attention"]["share"]:
    #         # Weights shared after the model has been moved to TPU Device
    #         self.att_banks.weight = self.emb_banks.weight

    def get_new_embedding(self, ref, *args, **kwargs):
        """ Soft attention weight. Better not share_att_banks.
        key: self.att_banks
        value: self.emb_banks
        query: ref
        """
        ref[ref != ref] = 0
        q = self.w_qs(ref).view(1, -1, 1, self.d_word_vec)
        k = self.w_ks(self.att_banks).view(1, -1, 1, self.d_word_vec)
        q = q.permute(2, 0, 1, 3).contiguous().view(1, -1, self.d_word_vec)  # 1 x vocab_size x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(1, -1, self.d_word_vec)  # 1 x codebook_size x dk
        v = self.emb_banks.unsqueeze(0)
        weighted_embedding, attn = self.attention(q, k, v)
        with torch.no_grad():
            weighted_embedding[Constants.PAD].fill_(0)
        # print(torch.sum(ref))
        # print(torch.sum(self.att_banks, axis=1))
        return weighted_embedding.squeeze(0)


class SoftAttCodebook2(pl.LightningModule):
    def __init__(self, model_config, algorithm_config):
        super().__init__()
        # Attention layer
        #   key: att_banks
        #   value: emb_banks
        #   query: refs
        self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
        self.codebook_size = self.codebook_config["size"]
        self.d_word_vec = model_config["transformer"]["encoder_hidden"]

        self.emb_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        self.d_feat = self.codebook_config["representation_dim"]

        # att(feats, att_banks) -> token_id weights -> emb_banks
        centroids = load_hubert_centroids([
            "preprocessed_data/LibriTTS/train-clean-100_phoneme-features.npy",
            "preprocessed_data/AISHELL-3/train_phoneme-features.npy",
            "preprocessed_data/GlobalPhone/fr/train_phoneme-features.npy",
            # "preprocessed_data/GlobalPhone/de/train_phoneme-features.npy",
        ], self.codebook_size)
        self.att_banks = nn.Parameter(centroids)
        # self.att_banks.requires_grad = False

        # Since we use cosine similarity, do not need sqrt(d_k) factor as in paper.
        # self.attention = ScaledDotProductAttention(
        #     temperature=np.power(self.d_word_vec, 0.5)
        # )

        self.attention = ScaledDotProductAttention(temperature=0.1)

    def get_new_embedding(self, ref, *args, **kwargs):
        """ Soft attention weight. Better not share_att_banks.
        key: self.att_banks
        value: self.emb_banks
        query: ref
        """
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)
        ref[ref != ref] = 0

        # normalize
        ref_norms = ref.norm(dim=1, keepdim=True) + 1e-6
        normed_ref = ref / ref_norms

        bank_norms = self.att_banks.norm(dim=1, keepdim=True)
        normed_banks = self.att_banks / bank_norms

        q = normed_ref.unsqueeze(0)  # 1 x vocab_size x drepr
        k = normed_banks.unsqueeze(0)  # 1 x codebook_size x drepr
        v = self.emb_banks.unsqueeze(0)
        weighted_embedding, attn = self.attention(q, k, v)
        with torch.no_grad():
            weighted_embedding[Constants.PAD].fill_(0)
        # print(attn.squeeze().shape)
        # print(torch.max(attn.squeeze(), dim=1))
        # print(torch.sum(ref))
        # print(self.att_banks)
        return weighted_embedding.squeeze(0)


def load_hubert_centroids(paths, size):
    from sklearn.cluster import KMeans
    repr = [np.load(path) for path in paths]
    repr = np.concatenate(repr, axis=0)
    kmeans = KMeans(n_clusters=size, random_state=0).fit(repr)
    centers = kmeans.cluster_centers_
    return torch.from_numpy(centers)
