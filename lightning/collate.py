import numpy as np
import torch

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

    return (
        ids,
        raw_texts,
        torch.from_numpy(speakers).long(),
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

