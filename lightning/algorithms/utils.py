import torch
from torch import Tensor
from torch.nn import Module
from typing import List, Callable
import traceback

import learn2learn as l2l


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


class GBML(l2l.algorithms.GBML):
    """
    GBML with some modifications:
    - Add "inner_freeze" attribute, s.t. freeze throughout inner-loop and updated in outer-loop.
    - Add `copy()` to return a new `GBML` instance with original parameters, instead of cloned parameters in `clone()`.
    - Change original in-place `adapt()` into `adapt_()`.
    - Add out-of-space `adapt()`.
    """
    module: Module

    def inner_freeze(self, module_name=""):
        for p in self.module.get_submodule(module_name).parameters():
            p.inner_freeze = True

    def inner_unfreeze(self, module_name=""):
        for p in self.module.get_submodule(module_name).parameters():
            p.inner_freeze = False

    def outer_freeze(self, module_name=""):
        for p in self.module.get_submodule(module_name).parameters():
            p.requires_grad = False

    def outer_unfreeze(self, module_name=""):
        for p in self.module.get_submodule(module_name).parameters():
            p.requires_grad = True

    def clone(
            self,
            first_order=None,
            allow_unused=None,
            allow_nograd=None,
            adapt_transform=None,
            ):
        """
        **Description**

        Similar to `MAML.clone()`.

        **Arguments**

        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        **Returns**

        * **clone** (GBML) - A new `GBML` instance with the same parameters, submodules and functionalities as `self`.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        if adapt_transform is None:
            adapt_transform = self.adapt_transform
        module_clone = l2l.clone_module(self.module)
        update_clone = l2l.clone_module(self.compute_update)
        return GBML(
            module=module_clone,
            transform=self.transform,
            lr=self.lr,
            adapt_transform=adapt_transform,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
            compute_update=update_clone,
        )

    def copy(
        self,
        first_order=None,
        allow_unused=None,
        allow_nograd=None,
        adapt_transform=None,
    ):
        """
        Similar to `clone()`, but with original parameters instead of cloned parameters.

        Return a copied model instance pointing to the same parameters/buffers.
        Useful to keep track of the parameters before update_module().
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        if adapt_transform is None:
            adapt_transform = self.adapt_transform
        module_clone = copy_module(self.module)
        update_clone = copy_module(self.compute_update)
        return GBML(
            module=module_clone,
            transform=self.transform,
            lr=self.lr,
            adapt_transform=adapt_transform,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
            compute_update=update_clone,
        )

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
        In-place adapt. Same as original `adapt(), but deal with "inner_freeze" attr.
        """
        # return super().adapt(loss, first_order, allow_unused, allow_nograd)
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if self.adapt_transform and self._params_updated:
            # Update the learnable update function
            update_grad = torch.autograd.grad(
                loss,
                self.compute_update.parameters(),
                create_graph=second_order,
                retain_graph=second_order,
                allow_unused=allow_unused,
            )
            self.diff_sgd(self.compute_update, update_grad)
            self._params_updated = False

        # Update the module
        updates = self.compute_update(
            loss,
            self.module.parameters(),
            create_graph=second_order or self.adapt_transform,
            retain_graph=second_order or self.adapt_transform,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
        )
        for i, p in enumerate(self.module.parameters()):
            if not p.requires_grad or getattr(p, "inner_freeze", False):
                updates[i] = None
        self.diff_sgd(self.module, updates)
        self._params_updated = True

