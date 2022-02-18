import os
import json

from torch.utils.data import ConcatDataset
from learn2learn.data import MetaDataset, TaskDataset
from learn2learn.data.transforms import FusedNWaysKShots, LoadData
from learn2learn.data.task_dataset import DataDescription
from learn2learn.utils.lightning import EpisodicBatcher

from lightning.collate import SpeakerTaskCollate, LanguageTaskCollate
from .define import LANG_ID2SYMBOLS


def few_shot_task_dataset(_dataset, ways, shots, queries, n_tasks_per_label=-1, epoch_length=-1, type="spk"):
    """
        _dataset is already a `ConcatDataset`
    """
    if type == "spk":
        id2lb = get_multispeaker_id2lb(_dataset.datasets)
        _collate = SpeakerTaskCollate()
    else:
        id2lb = get_multilingual_id2lb(_dataset.datasets)
        _collate = LanguageTaskCollate({
            "lang_id2symbols": LANG_ID2SYMBOLS,
            "representation_dim": 1024,
        })

    meta_dataset = MetaDataset(_dataset, indices_to_labels=id2lb)

    if n_tasks_per_label > 0:
        # For val/test, constant number of tasks per label
        tasks = []
        for label, indices in meta_dataset.labels_to_indices.items():
            if len(indices) >= shots+queries:
                # 1-way-K-shots-Q-queries transforms per label
                transforms = [
                    FusedNWaysKShots(meta_dataset, n=ways, k=shots+queries,
                                     replacement=False, filter_labels=[label]),
                    LoadData(meta_dataset),
                ]
                # 1-way-K-shots-Q-queries task dataset
                _tasks = TaskDataset(
                    meta_dataset, task_transforms=transforms, num_tasks=n_tasks_per_label,
                    task_collate=_collate.get_meta_collate(shots, queries),
                )
                tasks.append(_tasks)
        tasks = ConcatDataset(tasks)

    else:
        # For train, dynamic tasks
        # 1-way-K-shots-Q-queries transforms
        transforms = [
            FusedNWaysKShots(meta_dataset, n=ways, k=shots+queries, replacement=True),
            LoadData(meta_dataset),
        ]
        # 1-way-K-shots-Q-queries task dataset
        tasks = TaskDataset(
            meta_dataset, task_transforms=transforms,
            task_collate=_collate.get_meta_collate(shots, queries),
        )
        if epoch_length > 0:
            # Epochify task dataset, for periodic validation
            tasks = EpisodicBatcher(tasks, epoch_length=epoch_length).train_dataloader()

    return tasks


def load_descriptions(tasks, filename):
    with open(filename, 'r') as f:
        loaded_descriptions = json.load(f)
    assert len(tasks.datasets) == len(loaded_descriptions), "TaskDataset count mismatch"

    for i, _tasks in enumerate(tasks.datasets):
        descriptions = loaded_descriptions[i]
        assert len(descriptions) == _tasks.num_tasks, "num_tasks mismatch"
        for j in descriptions:
            data_descriptions = [DataDescription(index) for index in descriptions[j]]
            task_descriptions = _tasks.task_transforms[-1](data_descriptions)
            _tasks.sampled_descriptions[int(j)] = task_descriptions


def write_descriptions(tasks, filename):
    descriptions = []
    for ds in tasks.datasets:
        data_descriptions = {}
        for i in ds.sampled_descriptions:
            data_descriptions[i] = [desc.index for desc in ds.sampled_descriptions[i]]
        descriptions.append(data_descriptions)

    with open(filename, 'w') as f:
        json.dump(descriptions, f, indent=4)


def load_SQids2Tid(SQids_filename, tag):
    with open(SQids_filename, 'r') as f:
        SQids = json.load(f)
    SQids2Tid = {}
    for i, SQids_dict in enumerate(SQids):
        sup_ids, qry_ids = SQids_dict['sup_id'], SQids_dict['qry_id']
        SQids2Tid[f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"] = f"{tag}_{i:03d}"
    return SQids, SQids2Tid


def get_SQids2Tid(tasks, tag):
    SQids = []
    SQids2Tid = {}
    for i, task in enumerate(tasks):
        sup_ids, qry_ids = task[0][0][0], task[1][0][0]
        SQids.append({'sup_id': sup_ids, 'qry_id': qry_ids})
        SQids2Tid[f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"] = f"{tag}_{i:03d}"
    return SQids, SQids2Tid


def prefetch_tasks(tasks, tag='val', log_dir=''):
    if (os.path.exists(os.path.join(log_dir, f'{tag}_descriptions.json'))
            and os.path.exists(os.path.join(log_dir, f'{tag}_SQids.json'))):
        # Recover descriptions
        load_descriptions(tasks, os.path.join(log_dir, f'{tag}_descriptions.json'))
        SQids, SQids2Tid = load_SQids2Tid(os.path.join(log_dir, f'{tag}_SQids.json'), tag)

    else:
        os.makedirs(log_dir, exist_ok=True)

        # Run through tasks to get descriptions
        SQids, SQids2Tid = get_SQids2Tid(tasks, tag)
        with open(os.path.join(log_dir, f"{tag}_SQids.json"), 'w') as f:
            json.dump(SQids, f, indent=4)
        write_descriptions(tasks, os.path.join(log_dir, f"{tag}_descriptions.json"))

    return SQids2Tid


def get_multispeaker_id2lb(datasets):
    id2lb = {}
    total = 0
    for dataset in datasets:
        l = len(dataset)
        id2lb.update({k: f"corpus_{dataset.lang_id}-spk_{dataset.speaker[k - total]}"
                     for k in range(total, total + l)})
        total += l

    return id2lb


def get_multilingual_id2lb(datasets):
    id2lb = {}
    total = 0
    for dataset in datasets:
        l = len(dataset)
        id2lb.update({k: dataset.lang_id for k in range(total, total + l)})
        total += l

    return id2lb
