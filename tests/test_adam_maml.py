import sys
import os
import torch
import pytest

from torch.testing import assert_allclose, assert_close
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from lightning.algorithms.MAML import AdamWMAML

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
        # print(self.layer.weight)
        # print(self.layer.bias)
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
    

def _check_model_close(model_0: BoringModel, model_1: BoringModel, **kwargs):
    state_dict_0 = model_0.state_dict()
    state_dict_1 = model_1.state_dict()
    for n in state_dict_0:
        assert_close(state_dict_1[n], state_dict_0[n], **kwargs)

def _check_adam_close(adam_0: torch.optim.AdamW, adam_1: AdamWMAML, **kwargs):
    adam_0_states = adam_0.state_dict()['state']
    for i in adam_0_states:
        adam_0_state = adam_0_states[i]
        adam_1_state = adam_1.compute_update.transforms_modules[i].transform.state_dict()
        for k in adam_0_state:
            assert_close(adam_1_state[k].view(-1), adam_0_state[k].view(-1), **kwargs)


def _construct(mode: str, weight_decay: float = 0.1):
    """
    Construct a model, optimizer, and AdamWMAML.

    Parameters
    ----------
    mode : str
        One of 'copy', 'clone', or ''.
    weight_decay : float
        Weight decay for AdamWMAML.

    Returns
    -------
    model_0 : BoringModel
    model_1 : BoringModel
    adam_0 : torch.optim.AdamW
    adam_1 : AdamWMAML
    
    """
    model_0 = BoringModel()
    model_1 = BoringModel()
    model_0.load_state_dict(model_0.state_dict())
    model_1.load_state_dict(model_0.state_dict())

    adam_0 = torch.optim.AdamW(model_0.parameters(), lr=0.1, weight_decay=weight_decay)
    _adam_1 = AdamWMAML(model_1, lr=0.1, weight_decay=weight_decay, allow_unused=True)
    if mode == 'copy':
        adam_1 = _adam_1.copy()
    elif mode == 'clone':
        adam_1 = _adam_1.clone()
    elif mode == '':
        adam_1 = _adam_1

    return model_0, model_1, adam_0, adam_1


@pytest.mark.parametrize('mode', ['', 'copy', 'clone'])
@pytest.mark.parametrize('weight_decay', [0.1, 0.0])
def test_adam_construct(mode: str, weight_decay: float):
    model_0, model_1, adam_0, adam_1 = _construct(mode, weight_decay)
    _check_model_close(model_0, adam_1.module)

def _forward(model_0: BoringModel, model_1: BoringModel, batch: torch.Tensor):
    output_0: dict[str, torch.Tensor] = model_0.training_step(batch, 0)
    output_1: dict[str, torch.Tensor] = model_1.training_step(batch, 0)
    return output_0['loss'], output_1['loss']


@pytest.mark.parametrize('mode', ['', 'copy', 'clone'])
@pytest.mark.parametrize('weight_decay', [0.1, 0.0])
@pytest.mark.parametrize('steps', [0, 1, 2])
def test_adam_adapt(mode: str, weight_decay: float, steps: int):
    model_0, model_1, adam_0, adam_1 = _construct(mode, weight_decay)

    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    dataset = list(train_data)
    for i in range(steps):
        batch = dataset[i]

        # Test forward
        loss_0, loss_1 = _forward(model_0, adam_1.module, batch)
        assert_close(loss_1, loss_0, atol=(1+i)*1e-5, rtol=1e-5)

        # Test backward + AdamW update
        adam_0.zero_grad()
        loss_0.backward()
        adam_0.step()
        adam_1.adapt_(loss=loss_1)
        _check_adam_close(adam_0, adam_1, atol=(1+i)*1e-5, rtol=1e-5)
        _check_model_close(model_0, adam_1.module, atol=(1+i)*1e-5, rtol=1e-5)
        if mode == '':
            _check_model_close(model_1, adam_1.module, atol=(1+i)*1e-5, rtol=1e-5)

    # Test query loss
    batch = dataset[steps]
    loss_0, loss_1 = _forward(model_0, adam_1.module, batch)
    assert_close(loss_1, loss_0, atol=(1+steps)*1e-5, rtol=1e-5)

    # Test backward
    grads_0 = torch.autograd.grad(loss_0, model_0.parameters(), allow_unused=True, retain_graph=True)
    grads_1 = torch.autograd.grad(loss_1, model_1.parameters(), allow_unused=True, retain_graph=True)
    grads_2 = torch.autograd.grad(loss_1, adam_1.module.parameters(), allow_unused=True, retain_graph=True)
    for g0, g1, g2 in zip(grads_0, grads_1, grads_2):
        # Test 1-step backward
        assert_close(g0, g2, atol=(1+steps)*1e-5, rtol=1e-5)

        # Test backward through multi-step updates to initial model_1
        if steps > 0 and mode != '' and weight_decay > 0:
            # steps > 0 and model != '': model updated, so grads should be different from initial model
            # weight_decay > 0: to prevent L1 loss leads to consistent gradients
            assert not torch.equal(g1, g2)
        else:
            assert_close(g1, g2, atol=(1+steps)*1e-5, rtol=1e-5)
    
    # Compare torch.grad to loss backward
    loss_1.backward()
    for (n, p), g in zip(model_1.named_parameters(), grads_1):
        if steps == 0:
            assert torch.equal(p, adam_1.module.state_dict()[n])
            assert_close(p.grad, g, atol=(1+steps)*1e-5, rtol=1e-5)
        elif mode == '':
            assert torch.equal(p, adam_1.module.state_dict()[n])
            assert p.grad is None, "model_1.parameters() are currently intermediate nodes instead of leaf nodes, so grads should be None."
        else:
            # non-updated v.s. updated
            assert not torch.equal(p, adam_1.module.state_dict()[n])
            assert_close(p.grad, g, atol=(1+steps)*1e-5, rtol=1e-5)