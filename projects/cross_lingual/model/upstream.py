from typing import List, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


# class MelExtractor(pl.LightningModule):
#     def __init__(self):
#         super().__init__()

#     def extract(self, wav_data: Union[List[torch.Tensor], List[np.ndarray]], norm=False) -> Tuple[torch.FloatTensor, List[int]]:
#         """
#         Easier user interface for using s3prl extractors.
#         Input is viewed as single batch and then forwarded, so be cautious about GPU usage!
#         Args:
#             wavs: List of wavs represented as numpy arrays or torch tensors.
#             norm: Normalize representation or not.
#         Return:
#             All hidden states and n_frames (for client to remove padding). Hidden states shape are (B, L, n_layers, dim).
#         """
#         wavs = []
#         n_frames = []
#         if isinstance(wav_data[0], np.ndarray):
#             for wav in wav_data:
#                 wavs.append(torch.from_numpy(wav).float().to(self.device))
#                 n_frames.append(len(wav) // (Constants.S3PRL_SAMPLE_RATE // 1000 * self._fp))
#         else:
#             for wav in wav_data:
#                 wavs.append(wav.float().to(self.device))
#                 n_frames.append(len(wav) // (Constants.S3PRL_SAMPLE_RATE // 1000 * self._fp))
        
#         representation = self._extract(wavs, norm=norm)
              
#         return representation, n_frames


class Padding(pl.LightningModule):
    def __init__(self):
        pass

    def extract(self, data: Union[List[torch.Tensor], List[np.ndarray]], norm=False) -> Tuple[torch.FloatTensor, List[int]]:
        """
        Perform simple padding and normalization.
        Args:
            data: List of data represented as numpy arrays or torch tensors.
            norm: Normalize representation or not.
        Return:
            All hidden states and n_frames (for client to remove padding). Hidden states shape are (B, L, dim).
        """
        n_frames = [d.shape[0] for d in data]
        if isinstance(data[0], np.ndarray):
            temp = [torch.from_numpy(d).float().to(self.device) for d in data]
        else:
            temp = data        
        representation = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True)  # B, L, dim
        if norm:
            representation = torch.nn.functional.normalize(representation, dim=-1)
              
        return representation, n_frames
