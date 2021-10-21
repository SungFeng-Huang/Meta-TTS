import numpy as np
import torch
from functools import partial

from utils.tools import pad_1D, pad_2D


def reprocess(data, idxs):
    ids = [data[idx]["id"] for idx in idxs]
    speakers = [data[idx]["speaker"] for idx in idxs]
    texts = [data[idx]["text"] for idx in idxs]
    raw_texts = [data[idx]["raw_text"] for idx in idxs]
    mels = [data[idx]["mel"] for idx in idxs]
    pitches = [data[idx]["pitch"] for idx in idxs]
    energies = [data[idx]["energy"] for idx in idxs]
    durations = [data[idx]["duration"] for idx in idxs]

    text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array([mel.shape[0] for mel in mels])

    speakers = np.array(speakers)
    texts = pad_1D(texts)
    mels = pad_2D(mels)
    pitches = pad_1D(pitches)
    energies = pad_1D(energies)
    durations = pad_1D(durations)

    if "ref_mel_slices" in data[0]:
        ref_mels = [data[idx]["ref_mel_slices"] for idx in idxs]
        # ref_mel_lens = np.array([len(ref_mel) for ref_mel in ref_mels])
        start = 0
        ref_slices = []
        for ref_mel in ref_mels:
            end = start + ref_mel.shape[0]
            ref_slices.append(slice(start, end))
            start = end

        ref_mels = np.concatenate(ref_mels, axis=0)
        speaker_args = (
            torch.from_numpy(ref_mels).float(),
            ref_slices
        )
    else:
        speaker_args = torch.from_numpy(speakers).long()

    return (
        ids,
        raw_texts,
        speaker_args,
        torch.from_numpy(texts).long(),
        torch.from_numpy(text_lens),
        max(text_lens),
        torch.from_numpy(mels).float(),
        torch.from_numpy(mel_lens),
        max(mel_lens),
        torch.from_numpy(pitches).float(),
        torch.from_numpy(energies),
        torch.from_numpy(durations).long(),
    )


def get_meta_collate(shots, queries, sort=True):
    """ data: N * (K + K)"""
    batch_size = shots + queries

    def collate_fn(data):
        data_size = len(data)
        assert data_size % batch_size == 0, "Assum batch_size = ways * (shots + queries)"
        assert data_size // batch_size > 0, "Assum batch_size = ways * (shots + queries)"

        if sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        idx_arr = idx_arr.reshape((-1, batch_size))
        # idx_arr = idx_arr.reshape((-1, batch_size)).tolist()

        sup_idx = np.zeros(batch_size, dtype=bool)
        sup_idx[np.arange(shots)] = True
        # sup_idx[np.random.choice(batch_size, shots)] = True
        qry_idx = ~sup_idx

        sup_out = list()
        qry_out = list()
        for idx in idx_arr[:, sup_idx]:
            sup_out.append(reprocess(data, idx))
        for idx in idx_arr[:, qry_idx]:
            qry_out.append(reprocess(data, idx))

        # 2(sup+qry) * n_batch * 12(data_ele)
        return (sup_out, qry_out)
        # output = list()
        # for idx in idx_arr:
        # output.append(reprocess(data, idx))

        return output
    return collate_fn


def get_multi_collate(shots, queries, sort=True):
    """ data: N * (K + K)"""
    batch_size = shots + queries

    def collate_fn(data):
        data_size = len(data)
        assert data_size % batch_size == 0, "Assum batch_size = ways * (shots + queries)"
        assert data_size // batch_size > 0, "Assum batch_size = ways * (shots + queries)"

        if sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        idx_arr = idx_arr.reshape((-1, batch_size))

        output = list()
        for idx in idx_arr:
            output.append(reprocess(data, idx))

        return output
    return collate_fn


def get_single_collate(sort=True):
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


class MultiLingualCollate(object):
    def __init__(self, config):
        self.lang_id2symbols = config["lang_id2symbols"]
        self.d_representation = config["representation_dim"]

    def get_multi_collate(self, shots, queries, sort=True):
        return partial(self.multi_collate_fn, shots=shots, queries=queries, sort=sort)

    def get_meta_collate(self, shots, queries, sort=True):
        return partial(self.meta_collate_fn, shots=shots, queries=queries, sort=sort)

    def meta_collate_fn(self, data, shots, queries, sort=True):
        import time
        st = time.time()
        """ data: N * (K + K)"""
        batch_size = shots + queries
        data_size = len(data)
        assert data_size % batch_size == 0, "Assum batch_size = ways * (shots + queries)"
        assert data_size // batch_size > 0, "Assum batch_size = ways * (shots + queries)"

        if sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        idx_arr = idx_arr.reshape((-1, batch_size))
        sup_idx = np.zeros(batch_size, dtype=bool)
        sup_idx[np.arange(shots)] = True
        qry_idx = ~sup_idx

        sup_out = list()
        qry_out = list()
        ref_p_embeddings = list()
        for (sup_idxs, qry_idxs) in zip(idx_arr[:, sup_idx], idx_arr[:, qry_idx]):
            st1 = time.time()
            sup_out.append(self.reprocess(data, sup_idxs))
            pad_sup = time.time() - st1
            filtered_qry_idxs = self.filter_query(data, sup_idxs, qry_idxs)
            filter = time.time() - st1
            ref_p_embeddings.append(
                self.calc_p_embedding(data, filtered_qry_idxs))
            calc_ref = time.time() - st1
            qry_out.append(self.reprocess(data, filtered_qry_idxs))
            pad_qry = time.time() - st1
            print(
                f"Pad sup: {pad_sup:.2f}s, Filter: {filter:.2f}s, Calc ref: {calc_ref:.2f}s, Pad qry: {pad_qry:.2f}s")
            print(
                f"Samples (sup, qry): ({len(sup_idxs)}, {len(filtered_qry_idxs)})")
        print(f"Multilingual collate: {time.time() - st:.2f}s")

        return (sup_out, qry_out, ref_p_embeddings)

    def multi_collate_fn(self, data, shots, queries, sort=True):
        batch_size = shots + queries
        data_size = len(data)
        assert data_size % batch_size == 0, "Assum batch_size = ways * (shots + queries)"
        assert data_size // batch_size > 0, "Assum batch_size = ways * (shots + queries)"

        if sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        idx_arr = idx_arr.reshape((-1, batch_size))

        output = list()
        for idx in idx_arr:
            output.append(reprocess(data, idx))

        return output

    def filter_query(self, data, sup_idxs, qry_idxs):
        appeared = {}
        filtered_qry_idxs = []
        for idx in sup_idxs:
            appeared.update({ph: True for ph in data[idx]["text"]})
        for idx in qry_idxs:
            flag = True
            for ph in data[idx]["text"]:
                if ph not in appeared:
                    flag = False
                    break
            if flag:
                filtered_qry_idxs.append(idx)
        return filtered_qry_idxs

    def calc_p_embedding(self, data, idxs):
        lang_id = data[idxs[0]]["language"]
        n_symbols = len(self.lang_id2symbols[lang_id])
        texts = [data[idx]["text"] for idx in idxs]
        representations = [data[idx]["representation"] for idx in idxs]
        table = {i: [] for i in range(n_symbols)}
        for text, representation in zip(texts, representations):
            for t, r in zip(text, representation):
                table[int(t)].append(r)
        p_embedding = np.zeros(
            (n_symbols, self.d_representation), dtype=np.float32)
        for i in range(n_symbols):
            if len(table[i]) == 0:
                p_embedding[i] = np.zeros(self.d_representation)
            else:
                p_embedding[i] = np.mean(np.stack(table[i], axis=0), axis=0)
        return p_embedding

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        if "ref_mel_slices" in data[0]:
            ref_mels = [data[idx]["ref_mel_slices"] for idx in idxs]
            # ref_mel_lens = np.array([len(ref_mel) for ref_mel in ref_mels])
            start = 0
            ref_slices = []
            for ref_mel in ref_mels:
                end = start + ref_mel.shape[0]
                ref_slices.append(slice(start, end))
                start = end

            ref_mels = np.concatenate(ref_mels, axis=0)
            speaker_args = (
                torch.from_numpy(ref_mels).float(),
                ref_slices
            )
        else:
            speaker_args = torch.from_numpy(speakers).long()

        return (
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
            torch.from_numpy(mels).float(),
            torch.from_numpy(mel_lens),
            max(mel_lens),
            torch.from_numpy(pitches).float(),
            torch.from_numpy(energies),
            torch.from_numpy(durations).long(),
        )
