import numpy as np
import torch
from functools import partial
from collections import defaultdict

from src.utils.tools import pad_1D, pad_2D


def reprocess(data, idxs):
    datas = [data[idx] for idx in idxs]

    ids = [data_i["id"] for data_i in datas]
    speakers = [data_i["speaker"] for data_i in datas]
    texts = [data_i["text"] for data_i in datas]
    raw_texts = [data_i["raw_text"] for data_i in datas]
    mels = [data_i["mel"] for data_i in datas]
    pitches = [data_i["pitch"] for data_i in datas]
    energies = [data_i["energy"] for data_i in datas]
    durations = [data_i["duration"] for data_i in datas]

    text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array([mel.shape[0] for mel in mels])

    speakers = np.array(speakers)
    texts = pad_1D(texts)
    mels = pad_2D(mels)
    pitches = pad_1D(pitches)
    energies = pad_1D(energies)
    durations = pad_1D(durations)

    if "spk_ref_mel_slices" in datas[0]:
        spk_ref_mels = [data_i["spk_ref_mel_slices"] for data_i in datas]
        # spk_ref_mel_lens = np.array([len(spk_ref_mel) for spk_ref_mel in spk_ref_mels])
        start = 0
        spk_ref_slices = []
        for spk_ref_mel in spk_ref_mels:
            end = start + spk_ref_mel.shape[0]
            spk_ref_slices.append(slice(start, end))
            start = end

        spk_ref_mels = np.concatenate(spk_ref_mels, axis=0)
        speaker_args = (
            torch.from_numpy(spk_ref_mels).float(),
            spk_ref_slices
        )
    else:
        speaker_args = torch.from_numpy(speakers).long()

    # return (
    #     ids,
    #     raw_texts,
    #     speaker_args,
    #     torch.from_numpy(texts).long(),
    #     torch.from_numpy(text_lens),
    #     max(text_lens),
    #     torch.from_numpy(mels).float(),
    #     torch.from_numpy(mel_lens),
    #     max(mel_lens),
    #     torch.from_numpy(pitches).float(),
    #     torch.from_numpy(energies),
    #     torch.from_numpy(durations).long(),
    # )
    batch = {
        "ids": ids,
        "raw_texts": raw_texts,
        "speaker_args": speaker_args,
        "texts": torch.from_numpy(texts).long(),
        "src_lens": torch.from_numpy(text_lens),
        "max_src_len": max(text_lens),
        "mels": torch.from_numpy(mels).float(),
        "mel_lens": torch.from_numpy(mel_lens),
        "max_mel_len": max(mel_lens),
        "p_targets": torch.from_numpy(pitches).float(),
        "e_targets": torch.from_numpy(energies),
        "d_targets": torch.from_numpy(durations).long(),
        "speakers": torch.from_numpy(speakers).long(),
    }

    if "accent" in datas[0]:
        #TODO
        accents = [data_i["accent"] for data_i in datas]
        regions = [data_i["region"] for data_i in datas]
        accents = np.array(accents)
        regions = np.array(regions)
        batch.update({
            "accents": torch.from_numpy(accents).long(),
            "regions": torch.from_numpy(regions).long(),
        })

    if "wav" in datas[0]:
        wavs = [data_i["wav"] for data_i in datas]
        wav_lens = np.array([wav.shape[0] for wav in wavs])
        wavs = pad_1D(wavs)
        batch.update({
            "wavs": torch.from_numpy(wavs).float(),
            "wav_lens": torch.from_numpy(wav_lens),
            "max_wav_len": max(wav_lens),
        })

    return batch


def split_reprocess(batch, idxs):
    # (
    #     ids,
    #     raw_texts,
    #     speaker_args,
    #     texts,
    #     text_lens,
    #     max_text_lens,
    #     mels,
    #     mel_lens,
    #     max_mel_lens,
    #     pitches,
    #     energies,
    #     durations,
    # ) = batch
    ids = batch["ids"]
    raw_texts = batch["raw_texts"]
    speaker_args = batch["speaker_args"]
    texts = batch["texts"]
    text_lens = batch["src_lens"]
    max_text_lens = batch["max_src_len"]
    mels = batch["mels"]
    mel_lens = batch["mel_lens"]
    pitches = batch["p_targets"]
    energies = batch["e_targets"]
    durations = batch["d_targets"]
    speakers = batch["speakers"]

    if isinstance(idxs, list):
        idxs = np.array(idxs)

    sub_ids = [ids[idx] for idx in idxs]
    sub_raw_texts = [raw_texts[idx] for idx in idxs]
    if isinstance(speaker_args, tuple):
        spk_ref_mels, spk_ref_slices = speaker_args
        start = 0
        sub_spk_ref_slices = []
        sub_spk_ref_mels = [spk_ref_mels[spk_ref_slices[idx]] for idx in idxs]
        for spk_ref_mel in sub_spk_ref_mels:
            end = start + spk_ref_mel.shape[0]
            sub_spk_ref_slices.append(slice(start, end))
            start = end
        sub_spk_ref_mels = torch.cat(sub_spk_ref_mels, dim=0)
        sub_speaker_args = (sub_spk_ref_mels, sub_spk_ref_slices)
    else:
        sub_speaker_args = speaker_args[idxs]
    sub_text_lens = text_lens[idxs]
    sub_max_text_lens = sub_text_lens.max()
    sub_texts = texts[idxs][:, :sub_max_text_lens]
    sub_mel_lens = mel_lens[idxs]
    sub_max_mel_lens = sub_mel_lens.max()
    sub_mels = mels[idxs][:, :sub_max_mel_lens]
    if pitches.shape[1] == max_text_lens:
        sub_pitches = pitches[idxs][:, :sub_max_text_lens]
    else:
        sub_pitches = pitches[idxs][:, :sub_max_mel_lens]
    if energies.shape[1] == max_text_lens:
        sub_energies = energies[idxs][:, :sub_max_text_lens]
    else:
        sub_energies = energies[idxs][:, :sub_max_mel_lens]
    sub_durations = durations[idxs][:, :sub_max_text_lens]

    # return (
    #     sub_ids,
    #     sub_raw_texts,
    #     sub_speaker_args,
    #     sub_texts,
    #     sub_text_lens,
    #     sub_max_text_lens,
    #     sub_mels,
    #     sub_mel_lens,
    #     sub_max_mel_lens,
    #     sub_pitches,
    #     sub_energies,
    #     sub_durations,
    # )
    sub_batch = {
        "ids": sub_ids,
        "raw_texts": sub_raw_texts,
        "speaker_args": sub_speaker_args,
        "texts": sub_texts,
        "src_lens": sub_text_lens,
        "max_src_len": sub_max_text_lens,
        "mels": sub_mels,
        "mel_lens": sub_mel_lens,
        "max_mel_len": sub_max_mel_lens,
        "p_targets": sub_pitches,
        "e_targets": sub_energies,
        "d_targets": sub_durations,
        "speakers": speakers[idxs],
    }

    if "accents" in batch:
        accents = batch["accents"]
        regions = batch["regions"]
        sub_accents = accents[idxs]
        sub_regions = regions[idxs]
        sub_batch.update({
            "accents": sub_accents,
            "regions": sub_regions,
        })

    if "wavs" in batch:
        wavs = batch["wavs"]
        wav_lens = batch["wav_lens"]
        sub_wav_lens = wav_lens[idxs]
        sub_max_wav_lens = sub_wav_lens.max()
        sub_wavs = wavs[idxs][:, :sub_max_wav_lens]
        sub_batch.update({
            "wavs": sub_wavs,
            "wav_lens": sub_wav_lens,
            "max_wav_len": sub_max_wav_lens,
        })

    return sub_batch


def get_single_collate(sort=True):
    """Used with BaselineDataModule"""
    def collate_fn(data):
        data_size = len(data)

        if sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        output = reprocess(data, idx_arr)

        return output
    return collate_fn


class SpeakerTaskCollate:
    """TaskDataset.task_collate

    Task: 1 way(spk), K shots, Q queries
    data: len(data) = K + Q     [GD]
    """

    def __init__(self):
        pass


    def get_meta_collate(self, shots, queries, sort=False, split=True):
        return partial(self.meta_collate_fn, shots=shots, queries=queries, sort=sort, split=split)


    def meta_collate_fn(self, data, shots, queries, sort=False, split=True):
        """
        split: split to sup/qry for meta-loss. If False, `get_meta_collate` is
            still different from `get_single_collate`, where the data
            distributions can be different.
        """
        batch_size = shots + queries
        data_size = len(data)
        assert data_size == batch_size, "n_batch=1 for speaker adaptation"

        if sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        idx_arr = idx_arr.reshape((-1, batch_size))

        if split:
            sup_idx = np.zeros(batch_size, dtype=bool)
            sup_idx[np.arange(shots)] = True
            # sup_idx[np.random.choice(batch_size, shots)] = True
            qry_idx = ~sup_idx

            # n_batch * 12 data elements
            sup_out = [reprocess(data, idx) for idx in idx_arr[:, sup_idx]]
            qry_out = [reprocess(data, idx) for idx in idx_arr[:, qry_idx]]

            # 2 * n_batch * 12
            output = (sup_out, qry_out)

        else:
            # n_batch * 12
            output = [reprocess(data, idx) for idx in idx_arr]

        return output


class LanguageTaskCollate:
    """TaskDataset.task_collate

    Task: N spks (N >= 1), 1 way(lang), K shots, Q queries, B batch_size
    data: len(data) = K + Q     [SGD, K%B=0]
    """

    def __init__(self, config):
        """
            config: dict,
        """
        self.lang_id2symbols = config["lang_id2symbols"]
        self.d_representation = config["representation_dim"]


    def get_meta_collate(self, shots, queries):
        return partial(self.meta_collate_fn, shots=shots, queries=queries)


    def meta_collate_fn(self, data, shots, queries):
        """ multi-speaker with multi-task inner-loop training:
                random split, use global speaker_id
        """
        # import time
        # st = time.time()
        batch_size = shots + queries
        data_size = len(data)
        # assert data_size % batch_size == 0, "Assume batch_size = 1 way * (shots + queries)"
        # assert data_size // batch_size > 0, "Assume batch_size = 1 way * (shots + queries)"
        assert data_size == batch_size, "len(data) = K + Q     [SGD, K%B=0]"

        idx_arr = np.arange(data_size)
        idx_arr = idx_arr.reshape((-1, batch_size))

        sup_out = list()
        qry_out = list()
        ref_phn_repr = None
        for idxs in idx_arr:
            sup_ids, qry_ids = self.split_sup_qry(data, idxs, shots, queries)
            # print("S/Q ids", sup_ids, qry_ids)

            # st1 = time.time()
            sup_out.append(reprocess(data, sup_ids))
            # pad_sup = time.time() - st1

            qry_out.append(reprocess(data, qry_ids))
            # pad_qry = time.time() - st1

            ref_phn_repr = self.calc_phn_repr(data, sup_ids)
            # calc_ref = time.time() - st1

        return (sup_out, qry_out, ref_phn_repr)


    def split_sup_qry(self, data, idxs, shots, queries):
        assert len(idxs) == shots + queries
        phn2idxs = defaultdict(list)
        for idx in idxs:
            phn_set = set(data[idx]["text"])
            for phn in phn_set:
                phn2idxs[phn].append(idx)

        sup_ids = []
        qry_ids = []
        for idx in idxs:
            if len(qry_ids) < queries:
                phn_set = set(data[idx]["text"])
                for phn in phn_set:
                    if len(phn2idxs[phn]) == 1:
                        sup_ids.append(idx)
                        break
                else:
                    qry_ids.append(idx)
                    for phn in phn_set:
                        phn2idxs[phn].remove(idx)
            else:
                sup_ids.append(idx)

        assert len(sup_ids) == shots and len(qry_ids) == queries
        return np.array(sup_ids), np.array(qry_ids)


    def calc_phn_repr(self, data, idxs):
        lang_id = data[idxs[0]]["language"]
        n_symbols = len(self.lang_id2symbols[lang_id])
        texts = [data[idx]["text"] for idx in idxs]
        representations = [data[idx]["representation"] for idx in idxs]

        table = {i: [] for i in range(n_symbols)}
        for text, representation in zip(texts, representations):
            # NOTE: len(text) == len(representation) ?
            for t, r in zip(text, representation):
                table[int(t)].append(r)

        phn_repr = np.zeros((n_symbols, self.d_representation), dtype=float)
        for i in range(n_symbols):
            if len(table[i]) == 0:
                phn_repr[i] = np.zeros(self.d_representation)
            else:
                phn_repr[i] = np.mean(np.stack(table[i], axis=0), axis=0)

        return torch.from_numpy(phn_repr).float()

