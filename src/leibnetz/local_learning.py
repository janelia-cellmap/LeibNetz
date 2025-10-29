# %%
import logging
from abc import ABC, abstractmethod
import torch

from leibnetz import LeibNet


class LearningRule(ABC):
    name: str = "LearningRule"
    requires_grad: bool = False

    def __init__(self):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def init_layers(self, layer):
        pass

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def update(self, module, args, kwargs, output):
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
        if not module.training:
            return None
        if not hasattr(module, "kernel_size"):
            # Only implemented for convolutional layers
            return None
        if not args or len(args) == 0:
            return None
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
            dim_permutations = list(range(non_spatial_dims)) + dim_permutations.tolist()
            perm_outputs = output.permute(*dim_permutations)
            perm_inputs = inputs.permute(*dim_permutations)
            outputs_of_perm_inputs = module.forward(perm_inputs)
        loss = torch.nn.functional.mse_loss(outputs_of_perm_inputs, perm_outputs)
        loss.backward()
        module.optimizer.step()
        # TODO: Weights are exploding, need to normalize or regularize them


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
        if not module.training:
            return None
        if not args or len(args) == 0 or output is None:
            return None
        inputs = args[0]
        if hasattr(module, "kernel_size") and isinstance(
            module.kernel_size, (list, tuple)
        ):
            ndims = len(module.kernel_size)
            # Extract patches for convolutional layers
            X = extract_kernel_patches(
                inputs,
                module.in_channels,
                module.kernel_size,
                module.stride,
                module.dilation,
            )  # = c1 x 3 x 3 x N
            Y = extract_image_patches(output, module.out_channels, ndims).T  # = N x c2
            d_W = X @ Y  # = c1 x 3 x 3 x 3 x c2
            if ndims == 2:
                d_W = d_W.permute(3, 0, 1, 2)
            if ndims == 3:
                d_W = d_W.permute(4, 0, 1, 2, 3)
            elif ndims == 4:
                d_W = d_W.permute(5, 0, 1, 2, 3, 4)

            d_W = self.normalize_fcn(d_W)
            module.weight.data += d_W * self.learning_rate
        else:
            # Linear or non-conv layers
            # d_W = Y*X
            Y = output
            X = inputs
            d_W = Y.t() @ X
            d_W = self.normalize_fcn(d_W)
            module.weight.data += self.learning_rate * d_W
        return None


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
        normalize_kwargs: dict | None = {"dim": 0},
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
        return f"KrotovsRule(k_ratio={self.k_ratio}, delta={self.delta}, norm={self.norm}, normalize_kwargs={self.normalize_kwargs})"

    def init_layers(self, layer):
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(mean=0.0, std=1.0)

    @torch.no_grad()
    def update(self, module, args, kwargs, output):
        if not module.training or not args or len(args) == 0:
            return None
        inputs = args[0]
        batch_size = inputs.shape[0]
        weights = module.weight

        if hasattr(module, "kernel_size"):
            # Convolutional layers
            return self._update_conv_layer(module, inputs, output, batch_size)
        else:
            # Linear layers
            return self._update_linear_layer(module, inputs, output, batch_size)

    def _update_conv_layer(self, module, inputs, output, batch_size):
        """Update weights for convolutional layers using Krotov-Hopfield rule."""
        ndims = len(module.kernel_size)

        # Extract patches for convolutional layers
        X = extract_kernel_patches(
            inputs,
            module.in_channels,
            module.kernel_size,
            module.stride,
            module.dilation,
        )  # Shape: c_in x k1 x k2 x ... x N_patches

        # Flatten kernel dimensions for easier computation
        kernel_size_prod = torch.prod(torch.tensor(module.kernel_size)).item()
        X_flat = X.view(
            module.in_channels * kernel_size_prod, -1
        )  # (c_in * k_size, N_patches)

        # Flatten weights to match
        W_flat = module.weight.view(
            module.out_channels, -1
        )  # (out_channels, c_in * k_size)

        # Apply Krotov-Hopfield learning rule
        d_W_flat = self._krotov_update(W_flat, X_flat)

        # Reshape back to original weight shape
        d_W = d_W_flat.view_as(module.weight)

        # Apply weight update
        d_W = self.normalize_fcn(d_W)
        module.weight.data += d_W * self.learning_rate

        return None

    def _update_linear_layer(self, module, inputs, output, batch_size):
        """Update weights for linear layers using Krotov-Hopfield rule."""
        # For linear layers, inputs shape: (batch_size, in_features)
        # Weights shape: (out_features, in_features)

        X = inputs.T  # (in_features, batch_size)
        W = module.weight  # (out_features, in_features)

        # Apply Krotov-Hopfield learning rule
        d_W = self._krotov_update(W, X)

        # Apply weight update
        d_W = self.normalize_fcn(d_W)
        module.weight.data += d_W * self.learning_rate

        return None

    def _krotov_update(self, weights, inputs):
        """
        Core Krotov-Hopfield learning rule implementation.

        Args:
            weights: (num_units, input_dim) weight matrix
            inputs: (input_dim, num_samples) input matrix

        Returns:
            d_W: Weight updates of same shape as weights
        """
        num_units = weights.shape[0]
        input_dim = weights.shape[1]
        num_samples = inputs.shape[1]

        # Calculate k (number of active units)
        k = max(1, int(self.k_ratio * num_units))

        # Calculate overlaps between weights and inputs
        # weights: (num_units, input_dim), inputs: (input_dim, num_samples)
        # overlaps: (num_units, num_samples)
        if self.norm == 2:
            # L2 norm case (most common)
            overlaps = weights @ inputs  # (num_units, num_samples)
        else:
            # General Lp norm case
            weight_norms = torch.sign(weights) * torch.abs(weights) ** (self.norm - 1)
            overlaps = weight_norms @ inputs

        # Initialize weight updates
        d_W = torch.zeros_like(weights)

        # Process each sample independently
        for sample_idx in range(num_samples):
            sample_input = inputs[:, sample_idx]  # (input_dim,)
            sample_overlaps = overlaps[:, sample_idx]  # (num_units,)

            # Find top-k active units for this sample
            _, top_k_indices = torch.topk(sample_overlaps, k)

            # Create activation pattern
            activations = torch.zeros(num_units, device=weights.device)

            # Set activations for top-k units
            for i, unit_idx in enumerate(top_k_indices):
                if i == 0:
                    # Winner (most active unit) gets positive activation
                    activations[unit_idx] = 1.0
                elif i == k - 1:
                    # Last active unit gets anti-Hebbian activation
                    activations[unit_idx] = -self.delta
                # Other units remain at 0

            # Apply learning rule: d_W = activations * input - activations * weights * activation_strength
            for unit_idx in range(num_units):
                if activations[unit_idx] != 0:
                    activation = activations[unit_idx]
                    current_weight = weights[unit_idx, :]

                    # Hebbian term: activation * input
                    hebbian_term = activation * sample_input

                    # Anti-Hebbian term: activation * current_weight * overlap
                    overlap_strength = sample_overlaps[unit_idx]
                    anti_hebbian_term = activation * overlap_strength * current_weight

                    # Combine terms
                    unit_update = hebbian_term - anti_hebbian_term

                    # Accumulate update (average over samples)
                    d_W[unit_idx, :] += unit_update / num_samples

        # Normalize updates to prevent explosion
        update_norm = torch.max(torch.abs(d_W))
        if update_norm > self.precision:
            d_W = d_W / update_norm

        return d_W


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
        if not module.training:
            return None
        if not args or len(args) == 0 or output is None:
            return None
        inputs = args[0]
        if hasattr(module, "kernel_size") and isinstance(
            module.kernel_size, (list, tuple)
        ):
            ndims = len(module.kernel_size)
            # d_W = Y*(X - Y*W)
            # Extract patches for convolutional layers
            X = extract_kernel_patches(
                inputs,
                module.in_channels,
                module.kernel_size,
                module.stride,
                module.dilation,
            )  # = c1 x 3 x3 x N
            Y = extract_image_patches(output, module.out_channels, ndims)  # = c2 x N
            W = module.weight.data  # = c2 x c1 x 3 x 3 x 3
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

            d_W = self.normalize_fcn(d_W)
            module.weight.data += d_W * self.learning_rate
        else:
            # Linear or non-conv layers
            # Oja's rule: d_W = Y*(X - Y*W)
            # For linear layer: W is (out_features, in_features), X is (batch, in_features), Y is (batch, out_features)
            Y = output  # (batch, out_features)
            X = inputs  # (batch, in_features)
            W = module.weight.data  # (out_features, in_features)

            # Calculate X - Y @ W where Y @ W has shape (batch, in_features)
            # Then multiply by Y.T to get shape (in_features, out_features)
            # Finally transpose to get (out_features, in_features) to match W
            d_W = Y.T @ (X - Y @ W)  # (out_features, in_features)

            d_W = self.normalize_fcn(d_W)
            module.weight.data += d_W * self.learning_rate
        return None


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
    # Ensure hooks is always a list
    if not hasattr(model, "learning_hooks") or not isinstance(
        model.learning_hooks, list
    ):
        model.learning_hooks = []
    if not hasattr(model, "learning_rules") or not isinstance(
        model.learning_rules, list
    ):
        model.learning_rules = []

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
