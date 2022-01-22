import torch
from torch.autograd import grad as torch_grad
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch import Tensor
from torch.nn import Module
from typing import List, Callable

import learn2learn as l2l
from learn2learn.utils import clone_module, update_module, detach_module, clone_parameters

import hypergrad as hg
from hypergrad import CG_torch, get_outer_gradients, update_tensor_grads

from lightning.collate import split_reprocess


class MAML(l2l.algorithms.MAML):
    def __init__(self,
                 model,
                 lr,
                 first_order=False,
                 allow_unused=None,
                 allow_nograd=False):
        super().__init__(model, lr, first_order, allow_unused, allow_nograd)

    def adapt(self,
              loss,
              first_order=None,
              allow_unused=None,
              allow_nograd=None):
        """
        Different from l2l.algorithms.MAML, we make this function out of space.
        """
        copied = self.copy()
        # super(MAML, copied).adapt(loss, first_order, allow_unused, allow_nograd)
        copied.adapt_(loss, first_order, allow_unused, allow_nograd)
        return copied

    def adapt_(self,
              loss,
              first_order=None,
              allow_unused=None,
              allow_nograd=None):
        """
        In-place adapt. Same as l2l.algorithms.MAML.adapt().
        """
        return super().adapt(loss, first_order, allow_unused, allow_nograd)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)

    def copy(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        Return a copied model instance pointing to the same parameters/buffers.
        Useful to keep track of the parameters before update_module().
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(copy_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)


class Task:
    """
    Handles the train and valdation loss for a single task
    """
    def __init__(self, sup_data, qry_data, batch_size=None, shuffle=True):
        self.sup_data = sup_data
        self.qry_data = qry_data
        sampler = RandomSampler(range(len(sup_data[0]))) if shuffle else range(len(sup_data[0]))
        self.sup_sampler = BatchSampler(sampler,
                                        batch_size=batch_size,
                                        drop_last=True)
        self.sup_it = iter(self.sup_sampler)

        self.batch_size = batch_size

    def reset_iterator(self):
        self.sup_it = iter(self.sup_sampler)

    def next_batch(self):
        try:
            idxs = next(self.sup_it)
        except:
            self.reset_iterator()
            idxs = next(self.sup_it)
        batch = split_reprocess(self.sup_data, idxs)
        return batch

    def __iter__(self):
        self.reset_iterator()
        return self

    def __next__(self):
        try:
            idxs = next(self.sup_it)
            batch = split_reprocess(self.sup_data, idxs)
            return batch
        except:
            raise StopIteration


def CG(module: MAML,
       hmodel: Module,
       K: int ,
       fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
       outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
       tol=1e-10,
       set_grad=True,
       stochastic=False) -> List[Tensor]:
    """
     Module version of `hypergrad.CG`.
     Computes the hypergradient by applying K steps of the conjugate gradient method (CG).
     It can end earlier when tol is reached.
     Args:
         module: the MAML learner of the inner solver procedure.
         hmodel: the outer model (or hyper-/meta-module), each element needs requires_grad=True
         K: the maximum number of conjugate gradient iterations
         fp_map: the fixed point map which defines the inner problem
         outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
         tol: end the method earlier when the norm of the residual is less than tol
         set_grad: if True set t.grad to the hypergradient for every t in hparams
         stochastic: set this to True when fp_map is not a deterministic function of its inputs
     Returns:
         the list of hypergradients for each element in hparams
    """
    # `detach_module` behaves in-place, so we need to `clone_module` first.
    module = module.clone()
    params_requires_grad = [n for n, p in module.module.named_parameters()
                            if p.requires_grad]
    detach_module(module)

    params = [p.requires_grad_(True) for n, p in module.module.named_parameters()
              if n in params_requires_grad]
    hparams = [p for n, p in hmodel.named_parameters()
               if n in params_requires_grad]
    module.train()
    for p, hp in zip(params, hparams):
        if p.shape != hp.shape:
            print(p.shape, hp.shape)

    o_loss, valid_error, predictions = outer_loss(module, hmodel)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(module, hmodel)

    def dfp_map_dw(xs):
        if stochastic:
            w_mapped_in = fp_map(module, hmodel)
            for w, p in zip(w_mapped_in, params):
                if w.shape != p.shape:
                    print(w, p)
            Jfp_mapTv = torch_grad(w_mapped_in, params, grad_outputs=xs, retain_graph=False)
        else:
            Jfp_mapTv = torch_grad(w_mapped, params, grad_outputs=xs, retain_graph=True)
        return [v - j for v, j in zip(xs, Jfp_mapTv)]

    vs = CG_torch.cg(dfp_map_dw, grad_outer_w, max_iter=K, epsilon=tol)  # K steps of conjugate gradient

    if stochastic:
        # learner' = forward(learner, system.model)
        w_mapped = fp_map(module, hmodel)

    # grads = torch.grad(learner', system.model)
    grads = torch_grad(w_mapped, hparams, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads, valid_error, predictions


def copy_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    Modified from l2l.utils.clone_module().
    
    **Description**

    Creates a copied module instance, whose parameters/buffers/submodules
    are linked to their origin data_ptr.
    While update_module() updates the model parameters in-place, it generates
    new parameters then assign it back to the model instance, this implies that
    we cannot access the original parameters from the computational graph anymore.
    Through this function, we can keep track of the original parameters without
    additional clonings.

    **Arguments**

    * **module** (Module) - Module to be copied.

    **Return**

    * (Module) - The copied module.

    **Example**

    ~~~python
    # create new model instance pointing to the original parameters/buffers
    copied = copy_module(model)   
    error = loss(copied(X), y)
    grads = torch.autograd.grad(
        error,
        copied.parameters(),
        create_graph=True,
    )
    updates = [-lr * g for g in grads]
    l2l.update_module(copied, updates=updates)
    # copied.parameters are newly created nodes on the computational graph, while
    # we can access the original parameters through model.parameters
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.

    if memo is None:
        # Maps data_ptr to the tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    copied = module.__new__(type(module))
    copied.__dict__ = module.__dict__.copy()
    copied._parameters = copied._parameters.copy()
    copied._buffers = copied._buffers.copy()
    copied._modules = copied._modules.copy()

    # Second, link all parameters
    if hasattr(copied, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    copied._parameters[param_key] = memo[param_ptr]
                else:
                    copied._parameters[param_key] = param
                    memo[param_ptr] = param

    # Third, handle the buffers if necessary
    if hasattr(copied, '_buffers'):
        for buffer_key in module._buffers:
            if copied._buffers[buffer_key] is not None and \
                    copied._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    copied._buffers[buffer_key] = memo[buff_ptr]
                else:
                    copied._buffers[buffer_key] = buff
                    memo[param_ptr] = buff

    # Then, recurse for each submodule
    if hasattr(copied, '_modules'):
        for module_key in copied._modules:
            copied._modules[module_key] = copy_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(copied, 'flatten_parameters'):
        copied = copied._apply(lambda x: x)
    return copied
