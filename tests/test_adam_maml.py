import sys
import os
import torch
import pytest

from torch.testing import assert_allclose, assert_close
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer

class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

num_samples = 10000

class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        output = self.layer(x)
        # self.print(self.trainer.state.stage)
        # self.print(self.layer.weight)
        # self.print(self.layer.bias)
        # self.print()
        print(self.layer.weight)
        print(self.layer.bias)
        print()
        return output

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.1, weight_decay=0.1)
        return torch.optim.AdamW(self.parameters(), lr=0.1, weight_decay=0.1)
    

@pytest.mark.parametrize('mode', ['', 'copy', 'clone'])
def test_adam(mode: str):
    model_0 = BoringModel()
    model_1 = BoringModel()
    model_1.load_state_dict(model_0.state_dict())

    from lightning.algorithms.MAML import AdamWMAML
    adam_0 = torch.optim.AdamW(model_0.parameters(), lr=0.1, weight_decay=0.1)
    _adam_1 = AdamWMAML(model_1, lr=0.1, weight_decay=0.1, allow_unused=True)
    if mode == 'copy':
        adam_1 = _adam_1.copy()
    elif mode == 'clone':
        adam_1 = _adam_1.clone()
    elif mode == '':
        adam_1 = _adam_1

    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    state_dict_0 = model_0.state_dict()
    state_dict_1 = adam_1.module.state_dict()
    for n in state_dict_0:
        assert_close(state_dict_1[n], state_dict_0[n], msg=f"{n}\n{state_dict_1[n]}\n{state_dict_0[n]}")

    batch = list(train_data)[0]
    output_0 = model_0.training_step(batch, 0)
    output_1 = adam_1.module.training_step(batch, 0)
    assert_close(output_1['loss'], output_0['loss'])

    adam_0.zero_grad()
    output_0['loss'].backward()
    adam_0.step()
    adam_1.adapt_(loss=output_1['loss'])

    adam_0_states = adam_0.state_dict()['state']
    for i in adam_0_states:
        adam_0_state = adam_0_states[i]
        adam_1_state = adam_1.compute_update.transforms_modules[i].transform.state_dict()
        for k in adam_0_state:
            assert_close(adam_1_state[k].view(-1), adam_0_state[k].view(-1))

    state_dict_0 = model_0.state_dict()
    state_dict_1 = adam_1.module.state_dict()
    for n in state_dict_0:
        assert_close(state_dict_1[n], state_dict_0[n], msg=f"{n}\n{state_dict_1[n]}\n{state_dict_0[n]}")
