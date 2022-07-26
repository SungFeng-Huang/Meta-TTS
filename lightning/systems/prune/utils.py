import torch
from torch.nn.utils.prune import PruningContainer, CustomFromMask, BasePruningMethod
from collections.abc import Iterable


def global_learnable_unstructured(parameters, pruning_method, importance_scores=None, **kwargs):
    r"""Basically the same as `torch.nn.utils.prune.global_unstructured`, while
    with learnable masks.

    Globally prunes tensors corresponding to all parameters in ``parameters``
    by applying the specified ``pruning_method``.
    Modifies modules in place by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter's importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Raises:
        TypeError: if ``PRUNING_TYPE != 'unstructured'``

    Note:
        Since global structured pruning doesn't make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.

    Examples:
        >>> net = nn.Sequential(OrderedDict([
                ('first', nn.Linear(10, 4)),
                ('second', nn.Linear(4, 1)),
            ]))
        >>> parameters_to_prune = (
                (net.first, 'weight'),
                (net.second, 'weight'),
            )
        >>> prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=10,
            )
        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        tensor(10, dtype=torch.uint8)

    """
    # ensure parameters is a list or generator of tuples
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    amount = kwargs.get("amount", None)
    if not isinstance(amount, float):
        raise TypeError("Not dealing with int amount yet")

    # flatten importance scores to consider them all at once in global pruning
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = torch.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = PruningContainer()
    container._tensor_name = "temp"  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # to make it match that of `container`
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            "the `pruning_method`. Found method {} of type {}".format(
                pruning_method, method.PRUNING_TYPE
            )
        )

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module, name in parameters:

        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # Assign the correct pre-computed mask to each parameter and add it
        # to the forward_pre_hooks like any other pruning method
        custom_from_learnable_mask(module, name, mask=param_mask)

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param


class CustomLearnableMask(CustomFromMask):

    def apply_mask(self, module):
        r"""Simply handles the multiplication between the parameter being
        pruned and the generated mask.
        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.

        Args:
            module (nn.Module): module containing the tensor to prune

        Returns:
            pruned_tensor (torch.Tensor): pruned version of the input tensor
        """
        # to carry out the multiplication, the mask needs to have been computed,
        # so the pruning method must know what tensor it's operating on
        assert self._tensor_name is not None, "Module {} has to be pruned".format(
            module
        )  # this gets set in apply()
        if module.training and getattr(module, "_update_mask", True):
            if not hasattr(module, self._tensor_name + "_logits"):
                mask = getattr(module, self._tensor_name + "_mask")
                mask.fill_(torch.mean(mask))
                logits = torch.nn.Parameter(
                    torch.distributions.utils.probs_to_logits(mask, is_binary=True),
                    requires_grad=True,
                )
                module.register_parameter(self._tensor_name + "_logits", logits)
            else:
                logits = getattr(module, self._tensor_name + "_logits")
            m = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=torch.tensor([1.0], device=logits.device),
                logits=logits
            )   # temp smaller the sharper, but sharpness could be decided by logits
            soft_mask = m.rsample()
            hard_mask = torch.ones_like(soft_mask).masked_fill_(soft_mask < 0.5, 0)
            new_mask = hard_mask - soft_mask.detach() + soft_mask
            module.register_buffer(self._tensor_name + "_mask", new_mask)

        return super().apply_mask(module)


def custom_from_learnable_mask(module, name, mask):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by applying the pre-computed mask in ``mask``.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.
        mask (Tensor): binary mask to be applied to the parameter.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> m = prune.custom_from_mask(
                nn.Linear(5, 3), name='bias', mask=torch.tensor([0, 1, 0])
            )
        >>> print(m.bias_mask)
        tensor([0., 1., 0.])

    """
    CustomLearnableMask.apply(module, name, mask)
    return module
