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

    def update(self, inputs, w):
        # TODO: Needs re-implementation
        d_ws = torch.zeros(inputs.size(0))
        for idx, x in enumerate(inputs):
            y = torch.dot(w, x)

            d_w = torch.zeros(w.shape)
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

    def init_layers(self, layers: list):
        for layer in [lyr.layer for lyr in layers]:
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.Conv2d:
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
        activations = torch.zeros((num_hidden_units, batch_size))
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

    def update(self, inputs, w):
        # TODO: needs re-implementation
        d_ws = torch.zeros(inputs.size(0), *w.shape)
        for idx, x in enumerate(inputs):
            y = torch.mm(w, x.unsqueeze(1))

            d_w = torch.zeros(w.shape)
            for i in range(y.shape[0]):
                for j in range(x.shape[0]):
                    d_w[i, j] = self.c * y[i] * (x[j] - y[i] * w[i, j])

            d_ws[idx] = d_w

        return torch.mean(d_ws, dim=0)


def convert_to_bio(model: LeibNet, learning_rule: LearningRule, **kwargs):
    """Converts a LeibNet model to use local bio-inspired learning rules.

    Args:
        model (LeibNet): Initial LeibNet model to convert.
        learning_rule (LearningRule): Learning rule to apply to the model. Can be `HebbsRule`, `KrotovsRule` or `OjasRule`.

    Returns:
        _type_: _description_
    """

    def hook(module, args, kwargs, output): ...

    for module in model.modules():
        if len(module._parameters) > 0:
            module.register_forward_hook(hook, with_kwargs=True)

    return model
