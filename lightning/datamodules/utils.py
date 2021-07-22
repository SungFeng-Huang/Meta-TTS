import os
import json

from torch.utils.data import ConcatDataset
from learn2learn.data import MetaDataset, TaskDataset
from learn2learn.data.transforms import FusedNWaysKShots, LoadData
from learn2learn.data.task_dataset import DataDescription
from learn2learn.utils.lightning import EpisodicBatcher

from lightning.collate import get_meta_collate


def few_shot_task_dataset(_dataset, ways, shots, queries, task_per_speaker=-1, epoch_length=-1):
    # Make meta-dataset, to apply 1-way-5-shots tasks
    id2lb = {k:v for k,v in enumerate(_dataset.speaker)}
    meta_dataset = MetaDataset(_dataset, indices_to_labels=id2lb)

    if task_per_speaker > 0:
        # constant number of 1-way-5-shots tasks for each speaker
        tasks = []
        for label, indices in meta_dataset.labels_to_indices.items():
            if len(indices) >= shots+queries:
                # 1-way-5-shots transforms of a label
                transforms = [
                    FusedNWaysKShots(meta_dataset, n=ways, k=shots+queries,
                                     replacement=False, filter_labels=[label]),
                    LoadData(meta_dataset),
                ]
                # 1-way-5-shot task dataset
                _tasks = TaskDataset(
                    meta_dataset, task_transforms=transforms, num_tasks=task_per_speaker,
                    task_collate=get_meta_collate(shots, queries, False),
                )
                tasks.append(_tasks)
        tasks = ConcatDataset(tasks)

    else:
        # 1-way-5-shots transforms
        transforms = [
            FusedNWaysKShots(meta_dataset, n=ways, k=shots+queries, replacement=True),
            LoadData(meta_dataset),
        ]
        # 1-way-5-shot task dataset
        tasks = TaskDataset(
            meta_dataset, task_transforms=transforms,
            task_collate=get_meta_collate(shots, queries, False),
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


def get_SQids2Tid(tasks, tag):
    SQids = []
    SQids2Tid = {}
    for i, task in enumerate(tasks):
        sup_ids, qry_ids = task[0][0][0], task[1][0][0]
        SQids.append({'sup_id': sup_ids, 'qry_id': qry_ids})
        SQids2Tid[f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"] = f"{tag}_{i:03d}"
    return SQids, SQids2Tid


def prefetch_tasks(tasks, tag='val', log_dir=''):
    if os.path.exists(os.path.join(log_dir, f'{tag}_descriptions.json')):
        # Recover descriptions
        load_descriptions(tasks, os.path.join(log_dir, f'{tag}_descriptions.json'))

    # Check whether loaded successfully
    SQids, SQids2Tid = get_SQids2Tid(tasks, tag)
    if os.path.exists(os.path.join(log_dir, f"{tag}_SQids.json")):
        with open(os.path.join(log_dir, f"{tag}_SQids.json"), 'r') as f:
            origin_SQids = json.load(f)
        assert origin_SQids == SQids
    else:
        with open(os.path.join(log_dir, f"{tag}_SQids.json"), 'w') as f:
            json.dump(SQids, f, indent=4)

    if not os.path.exists(os.path.join(log_dir, f'{tag}_descriptions.json')):
        write_descriptions(tasks, os.path.join(log_dir, f"{tag}_descriptions.json"))

    return SQids2Tid
