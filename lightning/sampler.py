import torch
import numpy as np
from torch.utils.data import Dataset, BatchSampler, DistributedSampler


class GroupBatchSampler(BatchSampler):
    def __init__(self, sampler, group_size, batch_size, drop_last, sort):
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = sampler.data_source
        self.group_size = group_size
        self.sort = sort

    def sort_batches(self, batches):
        gbidx = [idx for batch in batches for idx in batch]
        texts = [np.array(text_to_sequence(self.dataset.text[idx], self.dataset.cleaners)) for idx in gbidx]
        len_arr = np.array([text.shape[0] for text in texts])
        idx_arr = np.argsort(-len_arr)
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        assert len(idx_arr) == len(batches)
        return idx_arr

    def __iter__(self):
        batches = []
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
                if len(batches) == self.group_size:
                    if self.sort:
                        sorted_batches = self.sort_batches(batches)
                    else:
                        sorted_batches = batches
                    for b in sorted_batches:
                        yield b
                    batches = []
        if len(batches) > 0:
            if self.sort:
                sorted_batches = self.sort_batches(batches)
            else:
                sorted_batches = batches
            for b in sorted_batches:
                yield b
        if len(batch) > 0 and not self.drop_last:
            yield batch


class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]

    Reference:
        torchnlp.samplers.distributed_batch_sampler
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)
