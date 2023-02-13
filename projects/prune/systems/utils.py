import torch
from torch.nn.utils.prune import PruningContainer, CustomFromMask
import pytorch_lightning as pl
from collections.abc import Iterable
import numpy as np
import random
import copy


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
        mask = getattr(module, self._tensor_name + "_mask")
        if not hasattr(module, self._tensor_name + "_logits"):
            # would be called at hook construction
            mask.fill_(torch.mean(mask))
            # mean = torch.full_like(mask, 0.9)
            # mean = torch.full_like(mask, torch.mean(mask))
            logits = torch.nn.Parameter(
                torch.distributions.utils.probs_to_logits(mask, is_binary=True),
                requires_grad=True,
            )
            module.register_parameter(self._tensor_name + "_logits", logits)
        logits = getattr(module, self._tensor_name + "_logits")
        if torch.isnan(logits).sum() > 0:
            print("logits contain nan")
            print(module)

        if getattr(module, "_update_mask", True):
            if module.training:
                m = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                    temperature=torch.tensor([1.0], device=logits.device),
                    logits=logits
                )   # temp smaller the sharper, but sharpness could be decided by logits
                soft_mask = m.rsample()
                hard_mask = torch.ones_like(soft_mask).masked_fill_(soft_mask < 0.5, 0)
                new_mask = hard_mask - soft_mask.detach() + soft_mask
                module.register_buffer(self._tensor_name + "_mask", new_mask)
            else:
                # new_mask = torch.sigmoid(logits)
                new_mask = torch.ones_like(logits).masked_fill_(logits < 0, 0)
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


class LogitsAndMasks(pl.LightningModule):
    def __init__(self, model, mode, init_prob=1.0,
                 **mask_opt):
        super().__init__()
        self.mode = mode
        self.init_prob = init_prob

        d_hidden = model.encoder.d_model
        self.d_hidden_logits = self.create_logits(
            d_hidden, learnable=mask_opt.get("mask_hidden", False))

        # encoder/decoder logits
        for module_name in {"encoder","decoder"}:
            module = model.get_submodule(module_name)
            mod_name = module_name[:3]
            for i in range(len(module.layer_stack)):
                d_k = module.layer_stack[i].slf_attn.d_k
                d_v = module.layer_stack[i].slf_attn.d_v
                n_head = module.layer_stack[i].slf_attn.n_head
                self.register_parameter(
                    f"{mod_name}_{i}_n_head_logits",
                    self.create_logits(n_head, 1,
                                       learnable=mask_opt.get("mask_mha", False))
                )
                self.register_parameter(
                    f"{mod_name}_{i}_d_k_logits",
                    self.create_logits(1, d_k,
                                       learnable=mask_opt.get("mask_mha", False))
                )
                self.register_parameter(
                    f"{mod_name}_{i}_d_v_logits",
                    self.create_logits(1, d_v,
                                       learnable=mask_opt.get("mask_mha", False))
                )
                self.register_parameter(
                    f"{mod_name}_{i}_w_qs_bias_logits",
                    self.create_logits(1,
                                       learnable=mask_opt.get("mask_mha", False))
                )
                self.register_parameter(
                    f"{mod_name}_{i}_w_ks_bias_logits",
                    self.create_logits(1,
                                       learnable=mask_opt.get("mask_mha", False))
                )
                self.register_parameter(
                    f"{mod_name}_{i}_w_vs_bias_logits",
                    self.create_logits(1,
                                       learnable=mask_opt.get("mask_mha", False))
                )
                self.register_parameter(
                    f"{mod_name}_{i}_fc_bias_logits",
                    self.create_logits(1,
                                       learnable=mask_opt.get("mask_mha", False))
                )
                d_ffn_hid = module.layer_stack[i].pos_ffn.d_hid
                self.register_parameter(
                    f"{mod_name}_{i}_d_ffn_hid_logits",
                    self.create_logits(d_ffn_hid,
                                       learnable=mask_opt.get("mask_pos_ffn", False))
                )
                self.register_parameter(
                    f"{mod_name}_{i}_ffn_w_1_bias_logits",
                    self.create_logits(1,
                                       learnable=mask_opt.get("mask_pos_ffn", False))
                )
                self.register_parameter(
                    f"{mod_name}_{i}_ffn_w_2_bias_logits",
                    self.create_logits(1,
                                       learnable=mask_opt.get("mask_pos_ffn", False))
                )

        # variance_adaptor logits
        for mode_name in {"duration", "pitch", "energy"}:
            module = model.variance_adaptor.get_submodule(mode_name+"_predictor")
            d_conv = module.filter_size
            self.register_parameter(
                f"{mode_name}_conv_1_logits",
                self.create_logits(d_conv,
                                   learnable=mask_opt.get(f"mask_{mode_name}", False))
            )
            self.register_parameter(
                f"{mode_name}_conv_2_logits",
                self.create_logits(d_conv,
                                   learnable=mask_opt.get(f"mask_{mode_name}", False))
            )
            self.register_parameter(
                f"{mode_name}_conv_1_bias_logits",
                self.create_logits(1,
                                   learnable=mask_opt.get(f"mask_{mode_name}", False))
            )
            self.register_parameter(
                f"{mode_name}_conv_2_bias_logits",
                self.create_logits(1,
                                   learnable=mask_opt.get(f"mask_{mode_name}", False))
            )
            self.register_parameter(
                f"{mode_name}_linear_bias_logits",
                self.create_logits(1,
                                   learnable=mask_opt.get(f"mask_{mode_name}", False))
            )

        # mel_linear logits
        self.register_parameter(
            "mel_linear_bias_logits",
            self.create_logits(1,
                               learnable=mask_opt.get(f"mask_mel_linear", False))
        )

        # postnet logits
        d_embedding = model.postnet.d_embedding
        for i in range(len(model.postnet.convolutions)):
            if i != 0:
                self.register_parameter(
                    f"postnet_conv_{i}_d_conv_logits",
                    self.create_logits(d_embedding,
                                       learnable=mask_opt.get(f"mask_postnet", False))
                )
            self.register_parameter(
                f"postnet_conv_{i}_bias_logits",
                self.create_logits(1,
                                   learnable=mask_opt.get(f"mask_postnet", False))
            )

        # Setup properties
        for i in range(len(model.encoder.layer_stack)):
            enc_layer_i_masks = self.get_FFT_layer_i_property("enc", i)
            for mask_name, mask in enc_layer_i_masks.items():
                setattr(
                    self.__class__, f"encoder_layer_stack_{i}_{mask_name}",
                    property(mask)
                )
        for mode_name in {"duration", "pitch", "energy"}:
            VA_masks = self.get_variance_predictor_property(mode_name)
            for mask_name, mask in VA_masks.items():
                setattr(
                    self.__class__, f"variance_adaptor_{mode_name}_predictor_{mask_name}",
                    property(mask)
                )
        for i in range(len(model.decoder.layer_stack)):
            dec_layer_i_masks = self.get_FFT_layer_i_property("dec", i)
            for mask_name, mask in dec_layer_i_masks.items():
                setattr(
                    self.__class__, f"decoder_layer_stack_{i}_{mask_name}",
                    property(mask)
                )
        for i in range(len(model.postnet.convolutions)):
            postnet_layer_i_masks = self.get_postnet_layer_i_property(
                i, last_layer=(i+1 == len(model.postnet.convolutions)))
            for mask_name, mask in postnet_layer_i_masks.items():
                setattr(
                    self.__class__, f"postnet_{mask_name}",
                    property(mask)
                )

        # self.setup()

    # def setup(self):
    #     # sample once for sanity_check
    #     self.sample(training=True)

    def create_logits(self, *shape, learnable=True):
        if learnable:
            return torch.nn.Parameter(
                torch.distributions.utils.probs_to_logits(
                    torch.ones(*shape).to(self.device) * self.init_prob,
                    is_binary=True,
                ),
                requires_grad=True,
            )
        else:
            return torch.nn.Parameter(
                torch.distributions.utils.probs_to_logits(
                    torch.ones(*shape).to(self.device),
                    is_binary=True,
                ),
                requires_grad=False,
            )

    def create_dist(self, logits):
        return torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=torch.tensor([1.0], device=self.device),
            logits=logits,
        )

    def sample_masks(self, training=True, mode=""):
        """Sample masks from distributions(logits)
        """
        if mode == "":
            mode = self.mode

        # Create distributions
        self.dists = {}
        for n, p in self.named_parameters():
            basename, postfix = n.rsplit('_', 1)
            assert postfix == "logits", n
            # self.dists[basename] = self.create_dist(p)
            # dist = self.dists[basename]

        # for n, dist in self.dists.items():
            if training:
                if mode == "sigmoid":
                    logits = getattr(self, basename+"_logits")
                    new_mask = torch.sigmoid(logits)
                else:
                    dist = self.create_dist(p)
                    soft_mask = dist.rsample()
                    if self.mode == "hard":
                        hard_mask = torch.ones_like(soft_mask)
                        hard_mask = hard_mask.masked_fill_(soft_mask < 0.5, 0)
                        new_mask = hard_mask - soft_mask.detach() + soft_mask
                    elif self.mode == "soft":
                        new_mask = soft_mask
            else:
                logits = getattr(self, basename+"_logits")
                # new_mask = torch.sigmoid(logits)
                new_mask = torch.ones_like(logits)
                new_mask = new_mask.masked_fill_(logits < 0, 0)
            self.register_buffer(basename+"_mask", new_mask)

    def mask_model(self, model, detach_mask=False):
        """Setup model's mask to up-to-date"""
        for n, p in list(model.named_parameters()):
            #TODO: check whether exceptions
            if n in {"variance_adaptor.pitch_bins",
                     "variance_adaptor.energy_bins"}:
                continue
            elif "_orig" not in n:
                print(n)
                raise
            n = n[:-5]

            module_name, tensor_name = n.rsplit('.', 1)
            module = model.get_submodule(module_name)

            mask_name = '_'.join(n.split('.')) + "_mask"
            mask = getattr(self, mask_name).to(model.device)
            if detach_mask:
                mask = mask.detach()
            # print(mask)

            try:
                setattr(module, tensor_name, p * mask)
                # setattr(module, tensor_name+"_mask", mask.broadcast_to(p.shape))
                module.register_buffer(tensor_name+"_mask", mask.broadcast_to(p.shape))
            except:
                print(n)
                print(p)
                raise
        return

    def detach_mask(self):
        for n, b in self.named_buffers():
            b.detach_()

    @property
    def encoder_src_word_emb_weight_mask(self):
        return self.d_hidden_mask.view(1, -1)

    @property
    def encoder_position_enc_mask(self):
        return self.d_hidden_mask.view(1, 1, -1)

    def get_FFT_layer_i_property(self, module, i):
        def w_qs_weight_mask(self):
            in_channel_mask = self.d_hidden_mask    # [d_hidden]
            out_channel_mask = (
                getattr(self, f"{module}_{i}_n_head_mask")
                * getattr(self, f"{module}_{i}_d_k_mask")
            )   # [n_head, d_k]
            return in_channel_mask.view(1, -1) * out_channel_mask.view(-1, 1)
        def w_qs_bias_mask(self):
            in_channel_mask = getattr(self, f"{module}_{i}_w_qs_bias_mask")
            out_channel_mask = (
                getattr(self, f"{module}_{i}_n_head_mask")
                * getattr(self, f"{module}_{i}_d_k_mask")
            )   # [n_head, d_k]
            return in_channel_mask.view(-1) * out_channel_mask.view(-1)

        w_ks_weight_mask = w_qs_weight_mask
        def w_ks_bias_mask(self):
            in_channel_mask = getattr(self, f"{module}_{i}_w_ks_bias_mask")
            out_channel_mask = (
                getattr(self, f"{module}_{i}_n_head_mask")
                * getattr(self, f"{module}_{i}_d_k_mask")
            )   # [n_head, d_k]
            return in_channel_mask.view(-1) * out_channel_mask.view(-1)

        def w_vs_weight_mask(self):
            in_channel_mask = self.d_hidden_mask    # [d_hidden]
            out_channel_mask = (
                getattr(self, f"{module}_{i}_n_head_mask")
                * getattr(self, f"{module}_{i}_d_v_mask")
            )   # [n_head, d_v]
            return in_channel_mask.view(1, -1) * out_channel_mask.view(-1, 1)
        def w_vs_bias_mask(self):
            in_channel_mask = getattr(self, f"{module}_{i}_w_vs_bias_mask")
            out_channel_mask = (
                getattr(self, f"{module}_{i}_n_head_mask")
                * getattr(self, f"{module}_{i}_d_v_mask")
            )   # [n_head, d_v]
            return in_channel_mask.view(-1) * out_channel_mask.view(-1)

        def fc_weight_mask(self):
            in_channel_mask = (
                getattr(self, f"{module}_{i}_n_head_mask")
                * getattr(self, f"{module}_{i}_d_v_mask")
            )   # [n_head, d_v]
            out_channel_mask = self.d_hidden_mask   # [d_hidden]
            return in_channel_mask.view(1, -1) * out_channel_mask.view(-1, 1)
        def fc_bias_mask(self):
            in_channel_mask = getattr(self, f"{module}_{i}_fc_bias_mask")
            out_channel_mask = self.d_hidden_mask   # [d_hidden]
            return in_channel_mask.view(-1) * out_channel_mask.view(-1)

        def layer_norm_mask(self):
            channel_mask = self.d_hidden_mask   # [d_hidden]
            return channel_mask.view(-1)

        def w_1_weight_mask(self):
            in_channel_mask = self.d_hidden_mask    # [d_hidden]
            out_channel_mask = getattr(self, f"{module}_{i}_d_ffn_hid_mask")
            return (
                in_channel_mask.view(1, -1, 1) * out_channel_mask.view(-1, 1, 1)
            )
        def w_1_bias_mask(self):
            in_channel_mask = getattr(self, f"{module}_{i}_ffn_w_1_bias_mask")
            out_channel_mask = getattr(self, f"{module}_{i}_d_ffn_hid_mask")
            return in_channel_mask.view(-1) * out_channel_mask.view(-1)

        def w_2_weight_mask(self):
            in_channel_mask = getattr(self, f"{module}_{i}_d_ffn_hid_mask")
            out_channel_mask = self.d_hidden_mask   # [d_hidden]
            return (
                in_channel_mask.view(1, -1, 1) * out_channel_mask.view(-1, 1, 1)
            )
        def w_2_bias_mask(self):
            in_channel_mask = getattr(self, f"{module}_{i}_ffn_w_2_bias_mask")
            out_channel_mask = self.d_hidden_mask   # [d_hidden]
            return in_channel_mask.view(-1) * out_channel_mask.view(-1)

        return {
            "slf_attn_w_qs_weight_mask": w_qs_weight_mask,
            "slf_attn_w_qs_bias_mask": w_qs_bias_mask,
            "slf_attn_w_ks_weight_mask": w_ks_weight_mask,
            "slf_attn_w_ks_bias_mask": w_ks_bias_mask,
            "slf_attn_w_vs_weight_mask": w_vs_weight_mask,
            "slf_attn_w_vs_bias_mask": w_vs_bias_mask,
            "slf_attn_fc_weight_mask": fc_weight_mask,
            "slf_attn_fc_bias_mask": fc_bias_mask,
            "slf_attn_layer_norm_weight_mask": layer_norm_mask,
            "slf_attn_layer_norm_bias_mask": layer_norm_mask,
            "pos_ffn_w_1_weight_mask": w_1_weight_mask,
            "pos_ffn_w_1_bias_mask": w_1_bias_mask,
            "pos_ffn_w_2_weight_mask": w_2_weight_mask,
            "pos_ffn_w_2_bias_mask": w_2_bias_mask,
            "pos_ffn_layer_norm_weight_mask": layer_norm_mask,
            "pos_ffn_layer_norm_bias_mask": layer_norm_mask,
        }

    @property
    def speaker_emb_model_weight_mask(self):
        return self.d_hidden_mask.view(1, -1)

    @property
    def variance_adaptor_pitch_embedding_weight_mask(self):
        return self.d_hidden_mask.view(1, -1)

    @property
    def variance_adaptor_energy_embedding_weight_mask(self):
        return self.d_hidden_mask.view(1, -1)

    @property
    def decoder_position_enc_mask(self):
        return self.d_hidden_mask.view(1, 1, -1)

    def get_variance_predictor_property(self, module):
        def conv_layer_conv1d_1_conv_weight_mask(self):
            in_channel_mask = self.d_hidden_mask    # [d_hidden]
            out_channel_mask = getattr(self, f"{module}_conv_1_mask")
            return (
                in_channel_mask.view(1, -1, 1) * out_channel_mask.view(-1, 1, 1)
            )
        def conv_layer_conv1d_1_conv_bias_mask(self):
            in_channel_mask = getattr(self, f"{module}_conv_1_bias_mask")
            out_channel_mask = getattr(self, f"{module}_conv_1_mask")
            return in_channel_mask.view(-1) * out_channel_mask.view(-1)

        def conv_layer_layer_norm_1_mask(self):
            channel_mask = getattr(self, f"{module}_conv_1_mask")
            return channel_mask.view(-1)

        def conv_layer_conv1d_2_conv_weight_mask(self):
            in_channel_mask = getattr(self, f"{module}_conv_1_mask")
            out_channel_mask = getattr(self, f"{module}_conv_2_mask")
            return (
                in_channel_mask.view(1, -1, 1) * out_channel_mask.view(-1, 1, 1)
            )
        def conv_layer_conv1d_2_conv_bias_mask(self):
            in_channel_mask = getattr(self, f"{module}_conv_2_bias_mask")
            out_channel_mask = getattr(self, f"{module}_conv_2_mask")
            return in_channel_mask.view(-1) * out_channel_mask.view(-1)

        def conv_layer_layer_norm_2_mask(self):
            channel_mask = getattr(self, f"{module}_conv_2_mask")
            return channel_mask.view(-1)

        def linear_layer_weight_mask(self):
            in_channel_mask = getattr(self, f"{module}_conv_2_mask")
            # d_out=1, no out_channel_mask
            return in_channel_mask.view(1, -1)
        def linear_layer_bias_mask(self):
            in_channel_mask = getattr(self, f"{module}_linear_bias_mask")
            # d_out=1, no out_channel_mask
            return in_channel_mask.view(-1)

        return {
            "conv_layer_conv1d_1_conv_weight_mask":
                conv_layer_conv1d_1_conv_weight_mask,
            "conv_layer_conv1d_1_conv_bias_mask":
                conv_layer_conv1d_1_conv_bias_mask,
            "conv_layer_layer_norm_1_weight_mask":
                conv_layer_layer_norm_1_mask,
            "conv_layer_layer_norm_1_bias_mask":
                conv_layer_layer_norm_1_mask,
            "conv_layer_conv1d_2_conv_weight_mask":
                conv_layer_conv1d_2_conv_weight_mask,
            "conv_layer_conv1d_2_conv_bias_mask":
                conv_layer_conv1d_2_conv_bias_mask,
            "conv_layer_layer_norm_2_weight_mask":
                conv_layer_layer_norm_2_mask,
            "conv_layer_layer_norm_2_bias_mask":
                conv_layer_layer_norm_2_mask,
            "linear_layer_weight_mask": linear_layer_weight_mask,
            "linear_layer_bias_mask": linear_layer_bias_mask,
        }

    @property
    def mel_linear_weight_mask(self):
        in_channel_mask = self.d_hidden_mask    # [d_hidden]
        # d_out = d_mel, no out_channel_mask
        return in_channel_mask.view(1, -1)

    # mel_linear_bias_mask: already created by mel_linear_bias_logits
    # d_out = d_mel, no out_channel_mask

    def get_postnet_layer_i_property(self, i, last_layer=False):
        # Conv1D
        def convolutions_i_0_conv_weight_mask(self):
            if i == 0:
                # no input masking
                in_channel_mask = torch.ones(1).to(self.device)
            else:
                in_channel_mask = getattr(self, f"postnet_conv_{i}_d_conv_mask")
            if last_layer:
                out_channel_mask = torch.ones(1).to(self.device)
            else:
                out_channel_mask = getattr(self, f"postnet_conv_{i+1}_d_conv_mask")
            return (
                in_channel_mask.view(1, -1, 1) * out_channel_mask.view(-1, 1, 1)
            )
        def convolutions_i_0_conv_bias_mask(self):
            in_channel_mask = getattr(self, f"postnet_conv_{i}_bias_mask")
            if last_layer:
                out_channel_mask = torch.ones(1).to(self.device)
            else:
                out_channel_mask = getattr(self, f"postnet_conv_{i+1}_d_conv_mask")
            return in_channel_mask.view(-1) * out_channel_mask.view(-1)
        #BatchNorm1D
        def convolutions_i_1_BN_mask(self):
            if last_layer:
                channel_mask = torch.ones(1).to(self.device)
            else:
                channel_mask = getattr(self, f"postnet_conv_{i+1}_d_conv_mask")
            return channel_mask.view(-1)

        return {
            f"convolutions_{i}_0_conv_weight_mask":
                convolutions_i_0_conv_weight_mask,
            f"convolutions_{i}_0_conv_bias_mask":
                convolutions_i_0_conv_bias_mask,
            f"convolutions_{i}_1_weight_mask": convolutions_i_1_BN_mask,
            f"convolutions_{i}_1_bias_mask": convolutions_i_1_BN_mask,
            f"convolutions_{i}_1_running_mean_mask": convolutions_i_1_BN_mask,
            f"convolutions_{i}_1_running_var_mask": convolutions_i_1_BN_mask,
        }


def setup_structured_prune(model):

    def move_param_to_orig(module, tensor_name):
        if not hasattr(module, tensor_name+"_orig"):
            tensor = getattr(module, tensor_name)
            module.register_parameter(tensor_name+"_orig", tensor)
            del module._parameters[tensor_name]

    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        #TODO: check whether exceptions
        if n in {"variance_adaptor.pitch_bins",
                    "variance_adaptor.energy_bins"}:
            continue
        if n.endswith("num_batches_tracked"):
            continue
        # elif "_orig" in n:
        #     continue
        module_name, tensor_name = n.rsplit('.', 1)
        submodule = model.get_submodule(module_name)
        move_param_to_orig(submodule, tensor_name)

    # TODO: BN running_mean/running_var
    # Doesn't affect forward (would be masked by next layer's input channel mask)

class ManualPCGrad():
    """
    Borrowed from https://github.com/WeiChengTseng/Pytorch-PCGrad.
    Specifically for LearnableStructuredPruneSystem.
    """
    def __init__(self, model, mask_opt, reduction='mean'):
        self._model = model
        self._optim, self._reduction = mask_opt, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            self.model.manual_backward(obj, retain_graph=True)
            # obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

