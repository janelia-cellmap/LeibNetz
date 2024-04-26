# %%
import logging
from abc import ABC, abstractmethod
import torch

from leibnetz import LeibNet


class LearningRule(ABC):
    """
    This code is taken from https://github.com/Joxis/pytorch-hebbian.git
    The code is licensed under the MIT license.

    Please reference the following paper if you use this code:
    @inproceedings{talloen2020pytorchhebbian,
    author       = {Jules Talloen and Joni Dambre and Alexander Vandesompele},
    location     = {Online},
    title        = {PyTorch-Hebbian: facilitating local learning in a deep learning framework},
    year         = {2020},
    }
    """

    name: str = "LearningRule"

    def __init__(self):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def init_layers(self, model):
        pass

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def update(self, x, w):
        pass


class GeometricConsistencyRule(LearningRule):
    """
    Implements a geometric consistency local learning rule per module. Only implemented for convolutional layers.
    """

    name: str = "GeometricConsistencyRule"
    requires_grad: bool = True

    def __init__(
        self,
        learning_rate=0.1,
        optimizer="RAdam",
        optimizer_kwargs={},
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

    def __str__(self):
        return f"GeometricConsistencyRule(learning_rate={self.learning_rate}, optimizer={self.optimizer}, optimizer_kwargs={self.optimizer_kwargs})"

    def init_layers(self, layer):
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(mean=0.0, std=1.0)

    def update(self, module, args, kwargs, output):
        if module.training:
            if not hasattr(module, "kernel_size"):
                # Only implemented for convolutional layers
                return
            module.zero_grad()
            if not hasattr(module, "optimizer"):
                optimizer = getattr(torch.optim, self.optimizer)
                setattr(
                    module,
                    "optimizer",
                    optimizer(
                        [module.weight], lr=self.learning_rate, **self.optimizer_kwargs
                    ),
                )
            with torch.no_grad():
                inputs = args[0]
                ndims = len(module.kernel_size)
                # Randomly permute the input tensor in the spatial dimensions
                non_spatial_dims = len(inputs.shape) - ndims
                dim_permutations = torch.randperm(ndims) + non_spatial_dims
                dim_permutations = (
                    list(range(non_spatial_dims)) + dim_permutations.tolist()
                )
                perm_outputs = output.permute(*dim_permutations)
                perm_inputs = inputs.permute(*dim_permutations)
                outputs_of_perm_inputs = module.forward(perm_inputs)
            loss = torch.nn.functional.mse_loss(outputs_of_perm_inputs, perm_outputs)
            loss.backward()
            module.optimizer.step()
            # TODO: Weights are exploding, need to normalize them


class HebbsRule(LearningRule):
    name: str = "HebbsRule"
    requires_grad: bool = False

    def __init__(self, learning_rate=0.1, normalize_kwargs={"dim": 0}):
        super().__init__()
        self.learning_rate = learning_rate
        self.normalize_kwargs = normalize_kwargs
        if normalize_kwargs is not None:
            self.normalize_fcn = lambda x: torch.nn.functional.normalize(
                x, **normalize_kwargs
            )
        else:
            self.normalize_fcn = lambda x: x

    def __str__(self):
        return f"HebbsRule(learning_rate={self.learning_rate}, normalize_kwargs={self.normalize_kwargs})"

    def init_layers(self, layer):
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(mean=0.0, std=1.0)

    @torch.no_grad()
    def update(self, module, args, kwargs, output):
        if module.training:
            with torch.no_grad():
                inputs = args[0]
                if hasattr(module, "kernel_size"):
                    ndims = len(module.kernel_size)
                    # Extract patches for convolutional layers
                    X = extract_kernel_patches(
                        inputs,
                        module.in_channels,
                        module.kernel_size,
                        module.stride,
                        module.dilation,
                    )  # = c1 x 3 x3 x N
                    Y = extract_image_patches(
                        output, module.out_channels, ndims
                    ).T  # = N x c2
                    d_W = X @ Y  # = c1 x 3 x 3 x 3 x c2
                    if ndims == 2:
                        d_W = d_W.permute(3, 0, 1, 2)
                    if ndims == 3:
                        d_W = d_W.permute(4, 0, 1, 2, 3)
                    elif ndims == 4:
                        d_W = d_W.permute(5, 0, 1, 2, 3, 4)
                else:
                    d_W = inputs * output  # = c1 x c2

                d_W = self.normalize_fcn(d_W)
                module.weight.data += d_W * self.learning_rate


class KrotovsRule(LearningRule):
    """Krotov-Hopfield Hebbian learning rule fast implementation.
    Original source: https://github.com/DimaKrotov/Biological_Learning

    Args:
        precision: Numerical precision of the weight updates.
        delta: Anti-hebbian learning strength.
        norm: Lebesgue norm of the weights.
        k_ratio: Ranking parameter
    """

    name: str = "KrotovsRule"
    requires_grad: bool = False

    def __init__(
        self,
        learning_rate: float = 0.1,
        k_ratio: float = 0.5,
        delta: float = 0.4,
        norm: int = 2,
        normalize_kwargs: dict = {"dim": 0},
        precision: float = 1e-30,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        assert k_ratio <= 1, "k_ratio should be smaller or equal to 1"
        self.k_ratio = k_ratio
        self.delta = delta
        self.norm = norm
        self.normalize_kwargs = normalize_kwargs
        if normalize_kwargs is not None:
            self.normalize_fcn = lambda x: torch.nn.functional.normalize(
                x, **normalize_kwargs
            )
        else:
            self.normalize_fcn = lambda x: x
        self.precision = precision

    def __str__(self):
        return f"KrotovsRule(k_ratio={self.k_ratio}, delta={self.delta}, norm={self.norm}, normalize={self.normalize})"

    def init_layers(self, layer):
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(mean=0.0, std=1.0)

    @torch.no_grad()
    def update(self, module, args, kwargs, output):
        pass
        # if module.training:
        #     with torch.no_grad():
        #         inputs = args[0]
        #         N = inputs.shape[0]
        #         weights = module.weight
        #         num_hidden_units = weights.shape[0]
        #         k = int(self.k_ratio * num_hidden_units)
        #         if hasattr(module, "kernel_size"):
        #             ndims = len(module.kernel_size)

        #             # Extract patches for convolutional layers
        #             X = extract_kernel_patches(
        #                 inputs,
        #                 module.in_channels,
        #                 module.kernel_size,
        #                 module.stride,
        #                 module.dilation,
        #             )  # = c1 x 3 x 3 x 3 x N

        #             input_size = X.shape[-1]

        #             # Calculate overlap for each hidden unit and input sample
        #             tot_input = torch.tensordot(
        #                 torch.sign(weights) * torch.abs(weights) ** (self.norm - 1), X
        #             )

        #             # Get the top k activations for each input sample (hidden units ranked per input sample)
        #             _, indices = torch.topk(tot_input, k=k, dim=0)

        #             # Apply the activation function for each input sample
        #             activations = torch.zeros(
        #                 (num_hidden_units, N), device=weights.device
        #             )
        #             activations[indices[0], ...] = 1.0
        #             activations[indices[k - 1], ...] = -self.delta
        #             # ================== WIP ==================
        #             Y = extract_image_patches(
        #                 output, module.out_channels, ndims
        #             ).T  # = N x c2

        #             d_W = X @ Y  # = c1 x 3 x 3 x 3 x c2
        #             if ndims == 2:
        #                 d_W = d_W.permute(3, 0, 1, 2)
        #             if ndims == 3:
        #                 d_W = d_W.permute(4, 0, 1, 2, 3)
        #             elif ndims == 4:
        #                 d_W = d_W.permute(5, 0, 1, 2, 3, 4)

        #             # TODO: WIP
        #             weights = module.weight.view(
        #                 -1, torch.prod(torch.as_tensor(module.kernel_size))
        #             )
        #         else:
        #             # TODO: WIP
        #             # ndims = None
        #             # d_W = inputs * output  # = c1 x c2
        #             # weights = module.weight

        #         inputs = inputs.view(inputs.size(0), -1)

        #         inputs[0].shape[0]

        #         inputs = torch.t(inputs)

        #         # Calculate overlap for each hidden unit and input sample
        #         tot_input = torch.dot(
        #             torch.sign(weights) * torch.abs(weights) ** (self.norm - 1), inputs
        #         )

        #         # Get the top k activations for each input sample (hidden units ranked per input sample)
        #         _, indices = torch.topk(tot_input, k=k, dim=0)

        #         # Apply the activation function for each input sample
        #         activations = torch.zeros(
        #             (num_hidden_units, batch_size), device=weights.device
        #         )
        #         activations[indices[0], torch.arange(batch_size)] = 1.0
        #         activations[indices[k - 1], torch.arange(batch_size)] = -self.delta

        #         # Sum the activations for each hidden unit, the batch dimension is removed here
        #         xx = torch.sum(torch.mul(activations, tot_input), 1)

        #         # Apply the actual learning rule, from here on the tensor has the same dimension as the weights
        #         norm_factor = torch.mul(
        #             xx.view(xx.shape[0], 1).repeat((1, input_size)), weights
        #         )
        #         ds = torch.dot(activations, torch.t(inputs)) - norm_factor

        #         # Normalize the weight updates so that the largest update is 1 (which is then multiplied by the learning rate)
        #         nc = torch.max(torch.abs(ds))
        #         if nc < self.precision:
        #             nc = self.precision
        #         d_W = torch.true_divide(ds, nc)

        #         d_W = self.normalize_fcn(d_W)
        #         module.weight.data += d_W * self.learning_rate


class OjasRule(LearningRule):
    name: str = "OjasRule"
    requires_grad: bool = False

    def __init__(self, learning_rate=0.1, normalize_kwargs={"dim": 0}):
        super().__init__()
        self.learning_rate = learning_rate
        self.normalize_kwargs = normalize_kwargs
        if normalize_kwargs is not None:
            self.normalize_fcn = lambda x: torch.nn.functional.normalize(
                x, **normalize_kwargs
            )
        else:
            self.normalize_fcn = lambda x: x

    def __str__(self):
        return f"OjasRule(learning_rate={self.learning_rate}, normalize_kwargs={self.normalize_kwargs})"

    def init_layers(self, layer):
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(mean=0.0, std=1.0)

    @torch.no_grad()
    def update(self, module, args, kwargs, output):
        if module.training:
            with torch.no_grad():
                inputs = args[0]
                if hasattr(module, "kernel_size"):
                    # d_W = Y*(X - Y*W)
                    ndims = len(module.kernel_size)
                    # Extract patches for convolutional layers
                    X = extract_kernel_patches(
                        inputs,
                        module.in_channels,
                        module.kernel_size,
                        module.stride,
                        module.dilation,
                    )  # = c1 x 3 x3 x N
                    Y = extract_image_patches(
                        output, module.out_channels, ndims
                    )  # = c2 x N
                    W = module.weight  # = c2 x c1 x 3 x 3 x 3
                    if ndims == 2:
                        W = W.permute(1, 2, 3, 0)
                    if ndims == 3:
                        W = W.permute(1, 2, 3, 4, 0)
                    elif ndims == 4:
                        W = W.permute(1, 2, 3, 4, 5, 0)

                    d_W = (X - W @ Y) @ Y.T  # = c1 x 3 x 3 x 3 x c2
                    if ndims == 2:
                        d_W = d_W.permute(3, 0, 1, 2)
                    if ndims == 3:
                        d_W = d_W.permute(4, 0, 1, 2, 3)
                    elif ndims == 4:
                        d_W = d_W.permute(5, 0, 1, 2, 3, 4)
                else:
                    W = module.weight
                    d_W = output @ (inputs - output @ W)

                d_W = self.normalize_fcn(d_W)
                module.weight.data += d_W * self.learning_rate


def extract_kernel_patches(x, channels, kernel_size, stride, dilation, padding=0):
    # TODO: implement dilation and padding
    #   does the order in which the patches are returned matter?
    # [b, c, h, w] OR [b, c, d, h, w] OR [b, c, t, d, h, w]

    # Extract patches
    d = 2
    patches = x
    for k, s in zip(kernel_size, stride):
        patches = patches.unfold(d, k, s)
        d += 1

    if len(kernel_size) == 2:
        patches = patches.permute(1, 4, 5, 0, 2, 3).contiguous()
    if len(kernel_size) == 3:
        patches = patches.permute(1, 5, 6, 7, 0, 2, 3, 4).contiguous()
    elif len(kernel_size) == 4:
        patches = patches.permute(1, 6, 7, 8, 9, 0, 2, 3, 4, 5).contiguous()

    return patches.view(channels, *kernel_size, -1)


def extract_image_patches(x, channels, ndims):
    #   does the order in which the patches are returned matter?
    # [b, c, h, w] OR [b, c, d, h, w] OR [b, c, t, d, h, w]

    if ndims == 2:
        x = x.permute(1, 0, 2, 3).contiguous()
    if ndims == 3:
        x = x.permute(1, 0, 2, 3, 4).contiguous()
    elif ndims == 4:
        x = x.permute(1, 0, 2, 3, 4, 5).contiguous()

    return x.view(channels, -1)


def _add_learning_parts(
    model, rule: LearningRule, hook: torch.utils.hooks.RemovableHandle | list
):
    if not hasattr(model, "learning_hooks"):
        setattr(model, "learning_hooks", [])
    if not hasattr(model, "learning_rules"):
        setattr(model, "learning_rules", [])
    if isinstance(hook, list):
        model.learning_hooks.extend(hook)
    else:
        model.learning_hooks.append(hook)
    model.learning_rules.append(rule)


def convert_to_bio(model: LeibNet, learning_rule: LearningRule, init_layers=True):
    """Converts a LeibNet model to use local bio-inspired learning rules.

    Args:
        model (LeibNet): Initial LeibNet model to convert.
        learning_rule (LearningRule): Learning rule to apply to the model. Can be `HebbsRule`, `KrotovsRule` or `OjasRule`.
        learning_rate (float, optional): Learning rate for the learning rule. Defaults to 1.0.
        init_layers (bool, optional): Whether to initialize the model's layers. Defaults to True. This will discard existing weights.

    Returns:
        LeibNet: Model with local learning rules applied in forward hooks.
    """

    hook = learning_rule.update

    hooks = []
    for module in model.modules():
        if hasattr(module, "weight"):
            hooks.append(module.register_forward_hook(hook, with_kwargs=True))
            if init_layers:
                if hasattr(learning_rule, "init_layers"):
                    learning_rule.init_layers(module)
                else:
                    torch.nn.init.sparse_(module.weight, sparsity=0.5)
            module.weight.requires_grad = learning_rule.requires_grad
            _add_learning_parts(module, learning_rule, hooks[-1])

    _add_learning_parts(model, learning_rule, hooks)
    model.requires_grad_(learning_rule.requires_grad)

    return model


def convert_to_backprop(model: LeibNet):
    """Converts a LeibNet model to use backpropagation for training.

    Args:
        model (LeibNet): Initial LeibNet model to convert.

    Returns:
        LeibNet: Model with backpropagation applied in forward hooks.
    """

    for module in model.modules():
        if hasattr(module, "weight"):
            module.weight.requires_grad = True

    try:
        for hook in model.learning_hooks:
            hook.remove()
    except AttributeError:
        UserWarning(
            "Model does not have learning hooks. It is already using backpropagation."
        )

    return model


# # # %%
# from leibnetz.nets import build_unet

# unet = build_unet()
# inputs = unet.get_example_inputs()["input"]
# batch_size = 2
# inputs = torch.cat(
#     [
#         inputs,
#     ]
#     * batch_size,
#     dim=0,
# )
# for module in unet.modules():
#     if hasattr(module, "weight"):
#         break
# output = module(inputs)
# X = extract_kernel_patches(
#     inputs, module.in_channels, module.kernel_size, module.stride, module.dilation
# )
# Y = extract_image_patches(output, module.out_channels, len(module.kernel_size))
# W = module.weight  # == c2 x c1 x 3 x 3 x 3
# print(X.shape)
# print(Y.shape)
# # d_W = Y*(X - Y*W)
# (X - (W.permute(1, 2, 3, 4, 0) @ Y))  # == c1 x 3 x 3 x 3 x N
# ((X - (W.permute(1, 2, 3, 4, 0) @ Y)) @ Y.T).permute(
#     4, 0, 1, 2, 3
# )  # == c2 x c1 x 3 x 3 x 3
# # %%
# from leibnetz.nets import build_unet

# unet = build_unet()
# model = convert_to_bio(unet, OjasRule())
# batch = model(model.get_example_inputs())
# # %%
# %%
