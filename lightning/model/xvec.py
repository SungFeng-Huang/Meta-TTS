"""
https://github.com/manojpamk/pytorch_xvectors/blob/master/models.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

from lightning.augments import SpecAugmentMM


class DropoutMixin:
    def set_dropout_rate(self, p_drop):
        for x in self.modules():
            if isinstance(x, nn.Dropout):
                x.p = p_drop


class SimpleTDNN(DropoutMixin, pl.LightningModule):

    def __init__(self, num_classes: int, p_dropout: float):
        super().__init__()
        self.tdnn1 = nn.Conv1d(in_channels=80, out_channels=128, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, dilation=1)
        self.bn_tdnn3 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(2*128,128)
        self.bn_fc1 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(128,64)
        self.bn_fc2 = nn.BatchNorm1d(64, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(64,num_classes)

        self.aug = SpecAugmentMM(time_mix_width=100, time_stripes_num=2,
                                 freq_mix_width=20, freq_stripes_num=2)

    def forward(self, x, eps=1e-5):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.aug(x)

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.aug(x)

        if self.training:
            x = x + torch.randn(x.size(), device=self.device)*eps
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x


class XvecTDNN(DropoutMixin, pl.LightningModule):

    def __init__(self, num_classes: int, p_dropout: float = 0.5):
        """Initializer for XvecTDNN.

        Args:
            num_classes: Number of speakers/regions/accents.
            p_dropout: Dropout rate.
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        self.tdnn1 = nn.Conv1d(in_channels=80, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000,512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512,512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512,num_classes)

        self.aug = SpecAugmentMM(time_mix_width=100, time_stripes_num=2,
                                 freq_mix_width=20, freq_stripes_num=2)

    def forward(self, x, eps=1e-5):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.aug(x)

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))
        x = self.aug(x)

        if self.training:
            x = x + torch.randn(x.size()).to(x)*eps
            # shape = x.size()
            # noise = torch.cuda.FloatTensor(shape)
            # torch.randn(shape, out=noise)
            # x += noise*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x
