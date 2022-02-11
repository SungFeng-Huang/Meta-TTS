#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lightning.model.asr_model import Codebook, SoftBank, ASRCenterHead

from ..system2 import System
from ..utils import CodebookAnalyzer
from lightning.model.loss import PhonemeClassificationLoss
from lightning.callbacks.asr_saver import Saver
from lightning.utils import MatchingGraphInfo
from lightning.utils import asr_loss2dict as loss2dict
from text.define import LANG_ID2SYMBOLS


class CenterRefSystem(System):
    """
    Concrete class of ASR head for codebook quality evaluation (w/ average).
    """

    def __init__(self, *args, **kwargs):
        self.reg = 3
        super().__init__(*args, **kwargs)

        # tests
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)
        self.test_list = {
            "codebook visualization": self.visualize_matching,
            # "print head norm": self.print_head_norm,
            # "print dist norm": self.print_dist_norm,
            "phoneme transfer": self.phoneme_transfer,
        }

    def build_model(self):
        codebook_config = self.algorithm_config["adapt"]["phoneme_emb"]
        codebook_size = codebook_config["size"]
        d_feat = codebook_config["representation_dim"]
        d_word_vec = self.model_config["transformer"]["encoder_hidden"]
        num_heads = 4

        self.codebook = Codebook(codebook_size, d_feat, 
                                d_word_vec, num_heads=num_heads)
        self.banks = SoftBank(codebook_size, d_word_vec, num_heads=num_heads)

        self.asr_head = ASRCenterHead(d_word_vec, multilingual=False)
        self.loss_func = PhonemeClassificationLoss()
        self.loss_func2 = nn.MSELoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.codebook, self.banks, self.asr_head])

    def build_saver(self):
        return Saver(self.preprocess_config, self.log_dir, self.result_dir)
    
    def common_step(self, batch, batch_idx, train=True):
        _, _, ref, lang_id = batch[0]
        qry_batch = batch[0][1][0]

        attn = self.codebook(ref.unsqueeze(0))
        embedding = self.banks(attn, pad=True).squeeze(0)  # N, d_word_vec
        emb_texts = F.embedding(qry_batch[3], embedding, padding_idx=0)  # B, L, d_word_vec
        predictions = self.asr_head(emb_texts, lang_ids=lang_id)

        phn_loss = self.loss_func(qry_batch, predictions)
        cluster_loss = self.reg * self.loss_func2(embedding, self.asr_head.get_table(lang_id))
        return (phn_loss + cluster_loss, phn_loss, cluster_loss), predictions
    
    def training_step(self, batch, batch_idx):
        train_loss, predictions = self.common_step(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss[0], 'losses': train_loss, 'output': predictions, '_batch': qry_batch, 'lang_id': batch[0][3]}

    def validation_step(self, batch, batch_idx):
        self.log_matching(batch, batch_idx)
        val_loss, predictions = self.common_step(batch, batch_idx)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        
        # calculate acc
        mask = (qry_batch[3] != 0)
        acc = ((qry_batch[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        loss_dict.update({"Val/Acc": acc.item()})

        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch, 'lang_id': batch[0][3]}

    def visualize_matching(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            _, _, ref_phn_feats, lang_id = batch[0]
            matching = self.codebook.get_matching(ref=ref_phn_feats, lang_id=lang_id)
            self.codebook_analyzer.visualize_matching(batch_idx, matching)
        return None

    def log_matching(self, batch, batch_idx, stage="val"):
        step = self.global_step + 1
        _, _, ref_phn_feats, lang_id = batch[0]
        matchings = self.codebook.get_matching(ref=ref_phn_feats, lang_id=lang_id)
        for matching in matchings:
            fig = self.codebook_analyzer.plot_matching(matching, quantized=False)
            figure_name = f"{stage}/step_{step}_{batch_idx:03d}_{matching['title']}"
            self.logger[0].experiment.log_figure(
                figure_name=figure_name,
                figure=fig,
                step=step,
            )
            plt.close(fig)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        outputs = {}
        for test_name, test_fn in getattr(self, "test_list", {}).items(): 
            outputs[test_name] = test_fn(batch, batch_idx)

        return outputs

    def print_head_norm(self, batch, batch_idx):
        if batch_idx == 0:  # Execute only once
            self.eval()
            with torch.no_grad():
                head1 = self.asr_head.tables["table-0"]
                print("En Head norm:")
                print(torch.mean(head1 ** 2, dim=1))
                head2 = self.asr_head.tables["table-1"]
                print("Zh Head norm:")
                print(torch.mean(head2 ** 2, dim=1))
                head = self.asr_head.tables["table-2"]
                print("Fr Head norm:")
                print(torch.mean(head ** 2, dim=1))

    def print_dist_norm(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            print("Distance norm:")
            qry_batch = batch[0][1][0]
            texts = qry_batch[3]  # B, L
            _, prediction = self.common_step(batch, batch_idx, train=False)

            # Reference: https://stackoverflow.com/questions/66604482/indexing-using-pytorch-tensors-along-one-specific-dimension-with-3-dimensional-t
            output = -torch.gather(prediction, -1, texts.unsqueeze(-1)).squeeze(-1)
            print(output[0])
            print("Max Dist:")
            print(torch.max(-prediction, dim=2)[0][0])
            print("Mean Dist:")
            print(torch.mean(-prediction, dim=2)[0])

    def phoneme_transfer(self, batch, batch_idx):  # TBD
        lang_id2name = {
            0: "en",
            1: "zh",
            2: "fr",
            3: "de",
            4: "ru",
            5: "es",
            6: "jp",
            7: "cz",
        }
        def transfer_embedding(embedding, src_lang_id, target_lang_id, mask):
            transfer_dist = self.asr_head(embedding.unsqueeze(0), lang_ids=target_lang_id)  # transfer to target language
            soft_transfer_dist = F.softmax(transfer_dist, dim=2)
            title = f"{lang_id2name[src_lang_id]}-{lang_id2name[target_lang_id]}"
            # print(f"Min Dist {title}:")
            # print(torch.min(-transfer_dist, dim=2)[0][0])
            info = MatchingGraphInfo({
                "title": title,
                "y_labels": [LANG_ID2SYMBOLS[lang_id][int(m)] for m in mask[0]],
                "x_labels": LANG_ID2SYMBOLS[target_lang_id],
                "attn": soft_transfer_dist[0][mask].detach().cpu().numpy(),
                "quantized": False,
            })
            return info

        self.eval()
        with torch.no_grad():
            _, _, ref, lang_id = batch[0]
            try:
                assert ref.device == self.device
            except:
                ref = ref.to(device=self.device)
            ref[ref != ref] = 0

            ref_mask = torch.nonzero(ref.sum(dim=1), as_tuple=True)

            attn = self.codebook(ref.unsqueeze(0))
            embedding = self.banks(attn, pad=True).squeeze(0)  # N, d_word_vec

            infos = []
            infos.append(transfer_embedding(embedding, lang_id, 0, ref_mask))
            infos.append(transfer_embedding(embedding, lang_id, 1, ref_mask))
            infos.append(transfer_embedding(embedding, lang_id, 2, ref_mask))

            self.codebook_analyzer.visualize_phoneme_transfer(batch_idx, infos)
