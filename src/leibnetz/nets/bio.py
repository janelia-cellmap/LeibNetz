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

# %%
import logging
from abc import ABC, abstractmethod
import torch

from leibnetz import LeibNet


class LearningRule(ABC):

    def __init__(self):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def init_layers(self, model):
        pass

    @abstractmethod
    def update(self, x, w):
        pass


class HebbsRule(LearningRule):

    def __init__(self, c=0.1):
        super().__init__()
        self.c = c

    def update(self, inputs: torch.Tensor, weights: torch.Tensor):
        # TODO: Needs re-implementation
        d_ws = torch.zeros(inputs.size(0))
        for idx, x in enumerate(inputs):
            y = torch.dot(weights, x)

            d_w = torch.zeros(weights.shape)
            for i in range(y.shape[0]):
                for j in range(x.shape[0]):
                    d_w[i, j] = self.c * x[j] * y[i]

            d_ws[idx] = d_w

        return torch.mean(d_ws, dim=0)


class KrotovsRule(LearningRule):
    """Krotov-Hopfield Hebbian learning rule fast implementation.

    Original source: https://github.com/DimaKrotov/Biological_Learning

    Args:
        precision: Numerical precision of the weight updates.
        delta: Anti-hebbian learning strength.
        norm: Lebesgue norm of the weights.
        k: Ranking parameter
    """

    def __init__(self, precision=1e-30, delta=0.4, norm=2, k=2, normalize=False):
        super().__init__()
        self.precision = precision
        self.delta = delta
        self.norm = norm
        self.k = k
        self.normalize = normalize

    def init_layers(self, layer):
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(mean=0.0, std=1.0)

    def update(self, inputs: torch.Tensor, weights: torch.Tensor):
        batch_size = inputs.shape[0]
        num_hidden_units = weights.shape[0]
        input_size = inputs[0].shape[0]
        assert (
            self.k <= num_hidden_units
        ), "The amount of hidden units should be larger or equal to k!"

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
        _, indices = torch.topk(tot_input, k=self.k, dim=0)

        # Apply the activation function for each input sample
        activations = torch.zeros((num_hidden_units, batch_size), device=weights.device)
        activations[indices[0], torch.arange(batch_size)] = 1.0
        activations[indices[self.k - 1], torch.arange(batch_size)] = -self.delta

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

    def __init__(self, c=0.1):
        super().__init__()
        self.c = c

    def update(self, inputs: torch.Tensor, weights: torch.Tensor):
        # TODO: needs re-implementation
        d_ws = torch.zeros(inputs.size(0), *weights.shape)
        for idx, x in enumerate(inputs):
            y = torch.mm(weights, x.unsqueeze(1))

            d_w = torch.zeros(weights.shape)
            for i in range(y.shape[0]):
                for j in range(x.shape[0]):
                    d_w[i, j] = self.c * y[i] * (x[j] - y[i] * weights[i, j])

            d_ws[idx] = d_w

        return torch.mean(d_ws, dim=0)


def extract_image_patches(x, kernel_size, stride, dilation, padding=0):
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
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
    if len(kernel_size) == 3:
        # TODO: Not sure if this is right
        patches = patches.permute(0, 5, 6, 7, 1, 2, 3, 4).contiguous()
    elif len(kernel_size) == 4:
        patches = patches.permute(0, 6, 7, 8, 9, 1, 2, 3, 4, 5).contiguous()

    return patches.view(-1, *kernel_size)


def convert_to_bio(
    model: LeibNet, learning_rule: LearningRule, learning_rate=1.0, init_layers=True
):
    """Converts a LeibNet model to use local bio-inspired learning rules.

    Args:
        model (LeibNet): Initial LeibNet model to convert.
        learning_rule (LearningRule): Learning rule to apply to the model. Can be `HebbsRule`, `KrotovsRule` or `OjasRule`.
        learning_rate (float, optional): Learning rate for the learning rule. Defaults to 1.0.
        init_layers (bool, optional): Whether to initialize the model's layers. Defaults to True. This will discard existing weights.

    Returns:
        LeibNet: Model with local learning rules applied in forward hooks.
    """

    def hook(module, args, kwargs, output):
        if module.training:
            with torch.no_grad():
                if hasattr(module, "kernel_size"):
                    # Extract patches for convolutional layers
                    inputs = extract_image_patches(
                        args[0], module.kernel_size, module.stride, module.dilation
                    )
                    weights = module.weight.view(
                        -1, torch.prod(torch.as_tensor(module.kernel_size))
                    )
                else:
                    inputs = args[0]
                    weights = module.weight
                inputs = inputs.view(inputs.size(0), -1)
                d_w = learning_rule.update(inputs, weights)
                d_w = d_w.view(module.weight.size())
                module.weight.data += d_w * learning_rate

    hooks = []
    for module in model.modules():
        if hasattr(module, "weight"):
            hooks.append(module.register_forward_hook(hook, with_kwargs=True))
            module.weight.requires_grad = False
            if init_layers:
                learning_rule.init_layers(module)

    setattr(model, "learning_rule", learning_rule)
    setattr(model, "learning_rate", learning_rate)
    setattr(model, "learning_hooks", hooks)

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
