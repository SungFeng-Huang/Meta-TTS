#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from lightning.model.loss import PhonemeClassificationLoss

from ..utils import MAML, CodebookAnalyzer
from utils.tools import get_mask_from_lengths
from ..adaptor import AdaptorSystem
from lightning.systems.utils import Task
from lightning.utils import dual_loss2dict as loss2dict
from lightning.utils import LightningMelGAN, MatchingGraphInfo
from lightning.model.phoneme_embedding import PhonemeEmbedding
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks.dual_saver import Saver
from lightning.model.asr_model import ASRCenterHead, Codebook, SoftBank
from lightning.callbacks.utils import synth_samples, recon_samples
from text.define import LANG_ID2SYMBOLS


STATSDICT = {
    0: "./preprocessed_data/miniLibriTTS",
    1: "./preprocessed_data/miniAISHELL-3",
    2: "./preprocessed_data/miniGlobalPhone-fr",
    3: "./preprocessed_data/miniGlobalPhone-de",
    4: "",
    5: "./preprocessed_data/miniGlobalPhone-es",
    6: "./preprocessed_data/miniJVS",
    7: "./preprocessed_data/miniGlobalPhone-cz",
}

class DualMetaSystem(AdaptorSystem):
    """ 
    Concrete class with ANIL fastspeech2 and dual structure.
    Use SoftMultiAttCodeBook for ASR & TTS.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg = 1.0

        # tests
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)
        self.test_list = {
            "codebook visualization": self.visualize_matching,
            "phoneme transfer": self.phoneme_transfer,
            "adaptation": self.test_adaptation, 
        }

    def build_model(self):
        codebook_config = self.algorithm_config["adapt"]["phoneme_emb"]
        codebook_size = codebook_config["size"]
        d_feat = codebook_config["representation_dim"]
        d_word_vec = self.model_config["transformer"]["encoder_hidden"]
        num_heads = 4
        
        # Shared part
        self.codebook = Codebook(codebook_size, d_feat, 
                                d_word_vec, num_heads=num_heads)
        self.banks = SoftBank(codebook_size, d_word_vec, num_heads=num_heads)

        # ASR part
        self.asr_head = ASRCenterHead(d_word_vec, multilingual=False)

        # TTS part
        self.model = FastSpeech2(self.preprocess_config, self.model_config, self.algorithm_config)
        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

        # Loss
        self.tts_loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)
        self.asr_loss_func1 = PhonemeClassificationLoss()
        self.asr_loss_func2 = nn.MSELoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.codebook, self.banks, self.asr_head, self.model])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        saver.tts_saver.set_meta_saver()
        return saver

    def get_embedding(self, batch, freeze_banks=False):
        _, _, ref_phn_feats, lang_id = batch[0]
        if freeze_banks:
            with torch.no_grad():  # freeze asr pretrained part
                attn = self.codebook(ref_phn_feats.unsqueeze(0))
                embedding = self.banks(attn, pad=True).squeeze(0)  # N, d_word_vec
        else:
            attn = self.codebook(ref_phn_feats.unsqueeze(0))
            embedding = self.banks(attn, pad=True).squeeze(0)  # N, d_word_vec
        return embedding

    def build_learner(self, embedding):        
        # Directly assign weight here, do not turn embedding into nn.Parameters since gradients can not flow back!
        # e.g. do not use emb_layer.weights = ... or nn.Embedding.from_pretrained
        emb_layer = nn.Embedding(*embedding.shape, padding_idx=0).to(self.device)
        emb_layer._parameters['weight'] = embedding
        adapt_dict = nn.ModuleDict({
            k: getattr(self.model, k) for k in self.algorithm_config["adapt"]["modules"]
        })
        adapt_dict["embedding"] = emb_layer
        return MAML(adapt_dict, lr=self.adaptation_lr)
    
    def forward_learner(
        self, learner, speaker_args, texts, src_lens, max_src_len,
        mels=None, mel_lens=None, max_mel_len=None,
        p_targets=None, e_targets=None, d_targets=None,
        p_control=1.0, e_control=1.0, d_control=1.0,
        average_spk_emb=False,
    ):
        _get_module = lambda name: getattr(learner.module, name, getattr(self.model, name, None))
        embedding        = _get_module('embedding')
        encoder          = _get_module('encoder')
        variance_adaptor = _get_module('variance_adaptor')
        decoder          = _get_module('decoder')
        mel_linear       = _get_module('mel_linear')
        postnet          = _get_module('postnet')
        speaker_emb      = _get_module('speaker_emb')

        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        emb_texts = embedding(texts)
        output = encoder(emb_texts, src_masks)

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        )

        if speaker_emb is not None:
            spk_emb = speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            output += spk_emb.unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output, p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_masks,
        ) = variance_adaptor(
            output, src_masks, mel_masks, max_mel_len,
            p_targets, e_targets, d_targets, p_control, e_control, d_control,
        )

        if speaker_emb is not None:
            spk_emb = speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            if max_mel_len is None:  # inference stage
                max_mel_len = max(mel_lens)
            output += spk_emb.unsqueeze(1).expand(-1, max_mel_len, -1)

        output, mel_masks = decoder(output, mel_masks)
        output = mel_linear(output)

        tmp = postnet(output)
        postnet_output = tmp + output

        return (
            output, postnet_output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_masks, mel_masks, src_lens, mel_lens,
        )
    
    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None, task=None, train=True):
        # MAML
        first_order = not train
        for step in range(adaptation_steps):
            mini_batch = task.next_batch()

            preds = self.forward_learner(learner, *mini_batch[2:])
            train_error = self.tts_loss_func(mini_batch, preds)
            learner.adapt_(
                train_error[0], first_order=first_order,
                allow_unused=False, allow_nograd=True
            )
        return learner

    def meta_learn(self, batch, batch_idx, train=True):
        embedding = self.get_embedding(batch, freeze_banks=False)

        # TTS part
        learner = self.build_learner(embedding)
        sup_batch, qry_batch, _, lang_id = batch[0]
        batch = [(sup_batch, qry_batch)]
        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]

        if self.adaptation_steps > 0:
            task = Task(sup_data=sup_batch,
                        qry_data=qry_batch,
                        batch_size=self.algorithm_config["adapt"]["imaml"]["batch_size"]) # The batch size is borrowed from the imaml config.
            
            learner = learner.clone()
            learner = self.adapt(batch, min(self.adaptation_steps, self.test_adaptation_steps), learner=learner, task=task, train=train)

        # Evaluating the adapted model
        tts_predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
        tts_error = self.tts_loss_func(qry_batch, tts_predictions)

        # ASR part
        emb_texts = F.embedding(qry_batch[3], embedding, padding_idx=0)  # B, L, d_word_vec
        asr_predictions = self.asr_head(emb_texts, lang_ids=lang_id)
        center_emb_texts = F.embedding(qry_batch[3], self.asr_head.get_table(lang_id), padding_idx=0) 

        phn_loss = self.asr_loss_func1(qry_batch, asr_predictions)
        center_loss = self.reg * self.asr_loss_func2(emb_texts, center_emb_texts)
        asr_error = (phn_loss + center_loss, phn_loss, center_loss)

        return (tts_error, asr_error), (tts_predictions, asr_predictions)

    def training_step(self, batch, batch_idx):
        train_loss, predictions = self.meta_learn(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss[0][0] + train_loss[1][0], 'dual_losses': train_loss, 'dual_output': predictions, '_batch': qry_batch, 'lang_id': batch[0][3]}

    def validation_step(self, batch, batch_idx):
        self.log_matching(batch, batch_idx)
        val_loss, predictions = self.meta_learn(batch, batch_idx)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}

        # calculate acc
        mask = (qry_batch[3] != 0)
        acc = ((qry_batch[3] == predictions[1].argmax(dim=2)) * mask).sum() / mask.sum()
        loss_dict.update({"Val/Acc": acc.item()})

        self.log_dict(loss_dict, sync_dist=True)
        return {'dual_losses': val_loss, 'dual_output': predictions, '_batch': qry_batch, 'lang_id': batch[0][3]}

    def test_step(self, batch, batch_idx):
        return super().test_step(batch, batch_idx)["adaptation"]
    
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

    def test_adaptation(self, batch, batch_idx):
        embedding = self.get_embedding(batch, freeze_banks=False)
        learner = self.build_learner(embedding)
        outputs = {}

        sup_batch, qry_batch, _, lang_id = batch[0]
        batch = [(sup_batch, qry_batch)]
        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]

        # Create result directory
        task_id = f"test_{batch_idx:03d}"
        figure_dir = os.path.join(self.result_dir, "figure", "Testing", f"step_{self.test_global_step}", task_id)
        audio_dir = os.path.join(self.result_dir, "audio", "Testing", f"step_{self.test_global_step}", task_id)
        figure_fit_dir = os.path.join(self.result_dir, "figure-fit", "Testing", f"step_{self.test_global_step}", task_id)
        audio_fit_dir = os.path.join(self.result_dir, "audio-fit", "Testing", f"step_{self.test_global_step}", task_id)
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(figure_fit_dir, exist_ok=True)
        os.makedirs(audio_fit_dir, exist_ok=True)
        config = copy.deepcopy(self.preprocess_config)
        config["path"]["preprocessed_path"] = STATSDICT[lang_id]

        # Build mini-batches
        task = Task(sup_data=sup_batch,
                    qry_data=qry_batch,
                    batch_size=self.algorithm_config["adapt"]["test"]["batch_size"])

        # Evaluate some training data to check overfit.
        fit_batch = task.next_batch()
        outputs['_batch_fit'] = fit_batch
        outputs['_batch'] = qry_batch

        # Evaluating the initial model (zero shot)
        learner.eval()
        self.model.eval()
        with torch.no_grad():
            fit_preds = self.forward_learner(learner, *fit_batch[2:], average_spk_emb=True)

            predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
            valid_error = self.tts_loss_func(qry_batch, predictions)
            outputs["step_0"] = {"recon": {"losses": valid_error}}
      
            # synth_samples & save & log
            # No reference from unseen speaker, use reference from support set instead.
            recon_samples(
                fit_batch, fit_preds, self.vocoder, config,
                figure_fit_dir, audio_fit_dir
            )
            recon_samples(
                qry_batch, predictions, self.vocoder, config,
                figure_dir, audio_dir
            )

            predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
            synth_samples(
                fit_batch, fit_preds, self.vocoder, config,
                figure_fit_dir, audio_fit_dir, f"step_{self.test_global_step}-FTstep_0"
            )
            synth_samples(
                qry_batch, predictions, self.vocoder, config,
                figure_dir, audio_dir, f"step_{self.test_global_step}-FTstep_0"
            )
        learner.train()
        self.model.train()

        # Determine fine tune checkpoints.
        ft_steps = list(range(1000, 20001, 1000))
        
        # Adapt
        learner = learner.clone()
        self.test_adaptation_steps = max(ft_steps)
        
        for ft_step in tqdm(range(self.adaptation_steps, self.test_adaptation_steps+1, self.adaptation_steps)):
            learner = self.adapt(batch, self.adaptation_steps, learner=learner, task=task, train=False)
            
            learner.eval()
            self.model.eval()
            with torch.no_grad():
                # Evaluating the adapted model
                predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
                valid_error = self.tts_loss_func(qry_batch, predictions)
                outputs[f"step_{ft_step}"] = {"recon": {"losses": valid_error}}

                # synth_samples & save & log
                if ft_step in ft_steps:
                    fit_preds = self.forward_learner(learner, *fit_batch[2:], average_spk_emb=True)
                    predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
                    synth_samples(
                        fit_batch, fit_preds, self.vocoder, config,
                        figure_fit_dir, audio_fit_dir, f"step_{self.test_global_step}-FTstep_{ft_step}"
                    )
                    synth_samples(
                        qry_batch, predictions, self.vocoder, config,
                        figure_dir, audio_dir, f"step_{self.test_global_step}-FTstep_{ft_step}"
                    )
            learner.train()
            self.model.train()
        del learner

        return outputs

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
