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

    def __init__(self):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def init_layers(self, model):
        pass

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def update(self, x, w):
        pass


class HebbsRule(LearningRule):

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
                    ndims = None
                    d_W = inputs * output  # = c1 x c2

                d_W = self.normalize_fcn(d_W)
                module.weight.data += d_W * self.learning_rate


class RhoadesRule(LearningRule):
    """Rule modifying Krotov-Hopfield Hebbian learning rule fast implementation.

    Args:
        precision: Numerical precision of the weight updates.
        delta: Anti-hebbian learning strength.
        norm: Lebesgue norm of the weights.
        k_ratio: Ranking parameter
    """

    def __init__(
        self, k_ratio=0.5, delta=0.4, norm=2, normalize=False, precision=1e-30
    ):
        super().__init__()
        self.precision = precision
        self.delta = delta
        self.norm = norm
        assert k_ratio <= 1, "k_ratio should be smaller or equal to 1"
        self.k_ratio = k_ratio
        self.normalize = normalize

    def __str__(self):
        return f"RhoadesRule(k_ratio={self.k_ratio}, delta={self.delta}, norm={self.norm}, normalize={self.normalize})"

    def init_layers(self, layer):
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(mean=0.0, std=1.0)

    def update(self, inputs: torch.Tensor, module: torch.nn.Module):
        # TODO: WIP
        if hasattr(module, "kernel_size"):
            # Extract patches for convolutional layers
            inputs = extract_kernel_patches(
                inputs, module.kernel_size, module.stride, module.dilation
            )
            weights = module.weight.view(
                -1, torch.prod(torch.as_tensor(module.kernel_size))
            )
        else:
            weights = module.weight
        inputs = inputs.view(inputs.size(0), -1)

        # TODO: needs re-implementation
        batch_size = inputs.shape[0]
        num_hidden_units = torch.prod(torch.as_tensor(weights.shape))
        input_size = inputs[0].shape[0]
        k = int(self.k_ratio * num_hidden_units)

        # TODO: WIP
        if self.normalize:
            norm = torch.norm(inputs, dim=1)
            norm[norm == 0] = 1
            inputs = torch.div(inputs, norm.view(-1, 1))

        inputs = torch.t(inputs)

        # Calculate overlap for each hidden unit and input sample
        tot_input = torch.matmul(
            torch.sign(weights) * torch.abs(weights) ** (self.norm - 1), inputs
        )

        # Get the top k activations for each input sample (hidden units ranked per input sample)
        _, indices = torch.topk(tot_input, k=k, dim=0)

        # Apply the activation function for each input sample
        activations = torch.zeros((num_hidden_units, batch_size), device=weights.device)
        activations[indices[0], torch.arange(batch_size)] = 1.0
        activations[indices[k - 1], torch.arange(batch_size)] = -self.delta

        # Sum the activations for each hidden unit, the batch dimension is removed here
        xx = torch.sum(torch.mul(activations, tot_input), 1)

        # Apply the actual learning rule, from here on the tensor has the same dimension as the weights
        norm_factor = torch.mul(
            xx.view(xx.shape[0], 1).repeat((1, input_size)), weights
        )
        ds = torch.matmul(activations, torch.t(inputs)) - norm_factor

        # Normalize the weight updates so that the largest update is 1 (which is then multiplied by the learning rate)
        nc = torch.max(torch.abs(ds))
        if nc < self.precision:
            nc = self.precision
        d_w = torch.true_divide(ds, nc)

        return d_w


class KrotovsRule(LearningRule):
    """Krotov-Hopfield Hebbian learning rule fast implementation.
    This code is taken from https://github.com/Joxis/pytorch-hebbian.git
    The code is licensed under the MIT license.

    Please reference the following paper if you use this code:
    @inproceedings{talloen2020pytorchhebbian,
    author       = {Jules Talloen and Joni Dambre and Alexander Vandesompele},
    location     = {Online},
    title        = {PyTorch-Hebbian: facilitating local learning in a deep learning framework},
    year         = {2020},
    }
    Original source: https://github.com/DimaKrotov/Biological_Learning

    Args:
        precision: Numerical precision of the weight updates.
        delta: Anti-hebbian learning strength.
        norm: Lebesgue norm of the weights.
        k_ratio: Ranking parameter
    """

    def __init__(
        self, k_ratio=0.5, delta=0.4, norm=2, normalize=False, precision=1e-30
    ):
        super().__init__()
        self.precision = precision
        self.delta = delta
        self.norm = norm
        assert k_ratio <= 1, "k_ratio should be smaller or equal to 1"
        self.k_ratio = k_ratio
        self.normalize = normalize

    def __str__(self):
        return f"KrotovsRule(k_ratio={self.k_ratio}, delta={self.delta}, norm={self.norm}, normalize={self.normalize})"

    def init_layers(self, layer):
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(mean=0.0, std=1.0)

    def update(self, inputs: torch.Tensor, module: torch.nn.Module):
        if hasattr(module, "kernel_size"):
            # Extract patches for convolutional layers
            inputs = extract_kernel_patches(
                inputs, module.kernel_size, module.stride, module.dilation
            )
            weights = module.weight.view(
                -1, torch.prod(torch.as_tensor(module.kernel_size))
            )
        else:
            weights = module.weight
        inputs = inputs.view(inputs.size(0), -1)

        batch_size = inputs.shape[0]
        num_hidden_units = weights.shape[0]
        input_size = inputs[0].shape[0]
        k = int(self.k_ratio * num_hidden_units)

        # TODO: WIP
        if self.normalize:
            norm = torch.norm(inputs, dim=1)
            norm[norm == 0] = 1
            inputs = torch.div(inputs, norm.view(-1, 1))

        inputs = torch.t(inputs)

        # Calculate overlap for each hidden unit and input sample
        tot_input = torch.matmul(
            torch.sign(weights) * torch.abs(weights) ** (self.norm - 1), inputs
        )

        # Get the top k activations for each input sample (hidden units ranked per input sample)
        _, indices = torch.topk(tot_input, k=k, dim=0)

        # Apply the activation function for each input sample
        activations = torch.zeros((num_hidden_units, batch_size), device=weights.device)
        activations[indices[0], torch.arange(batch_size)] = 1.0
        activations[indices[k - 1], torch.arange(batch_size)] = -self.delta

        # Sum the activations for each hidden unit, the batch dimension is removed here
        xx = torch.sum(torch.mul(activations, tot_input), 1)

        # Apply the actual learning rule, from here on the tensor has the same dimension as the weights
        norm_factor = torch.mul(
            xx.view(xx.shape[0], 1).repeat((1, input_size)), weights
        )
        ds = torch.matmul(activations, torch.t(inputs)) - norm_factor

        # Normalize the weight updates so that the largest update is 1 (which is then multiplied by the learning rate)
        nc = torch.max(torch.abs(ds))
        if nc < self.precision:
            nc = self.precision
        d_w = torch.true_divide(ds, nc)

        return d_w


class OjasRule(LearningRule):

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
                    ndims = None
                    d_W = inputs * output  # = c1 x c2

                d_W = self.normalize_fcn(d_W)
                module.weight.data += d_W * self.learning_rate
        d_W = self.c * ((inputs**2) * module.weight - (inputs**2) * (module.weight**3))
        return d_W


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
            module.weight.requires_grad = False
            setattr(module, "learning_rule", learning_rule)
            setattr(module, "learning_rate", learning_rule.learning_rate)
            setattr(module, "learning_hook", hooks[-1])

    setattr(model, "learning_rule", learning_rule)
    setattr(model, "learning_rate", learning_rule.learning_rate)
    setattr(model, "learning_hooks", hooks)
    model.requires_grad_(False)

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


# %%
from leibnetz.nets import build_unet

unet = build_unet()
inputs = unet.get_example_inputs()["input"]
batch_size = 2
inputs = torch.cat(
    [
        inputs,
    ]
    * batch_size,
    dim=0,
)
for module in unet.modules():
    if hasattr(module, "weight"):
        break
output = module(inputs)
X = extract_kernel_patches(
    inputs, module.in_channels, module.kernel_size, module.stride, module.dilation
)
Y = extract_image_patches(output, module.out_channels, len(module.kernel_size)).T
weights = module.weight
print(X.shape)
print(Y.shape)
# weights = weights.view(
#     *module.weight.shape[: -len(module.kernel_size)],
#     torch.prod(torch.as_tensor(module.kernel_size)),
# )

# print(inputs.shape)
# print(weights.shape)
# ((inputs**2) * weights).shape

# %%
from leibnetz.nets import build_unet

unet = build_unet()
model = convert_to_bio(unet, HebbsRule())
batch = model(model.get_example_inputs())
# %%
import numpy as np

c = 0.1
pad = (np.array(kernel_size) - 1) // 2
c * output * (
    inputs[..., pad[0] : -pad[0], pad[1] : -pad[1], pad[2] : -pad[2]] - output * weights
)
# %%
