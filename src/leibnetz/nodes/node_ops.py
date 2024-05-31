from typing import Callable
from torch import nn
from torch import Tensor
from torchvision.utils import _log_api_usage_once
from torchvision.ops import SqueezeExcitation

# from torchvision.ops import DeformConv2d
import numpy as np

# from funlib.learn.torch.models.conv4d import Conv4d


class ConvPass(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        kernel_sizes,
        activation="ReLU",
        final_activation=None,
        padding="valid",
        residual=False,
        padding_mode="reflect",
        norm_layer=None,
        dropout_prob=None,
        squeeze_excitation=False,
        squeeze_ratio=2,
        # deformable=False,
        # TODO: Consider adding DropBlock as an option
    ):
        """Convolution pass block

        Args:
            input_nc (int): Number of input channels
            output_nc (int): Number of output channels
            kernel_sizes (list(int) or array_like): Kernel sizes for convolution layers.
            activation (str or callable): Name of activation function in 'nn' or the function itself.
            final_activation (str or callable, optional): Name of activation function in 'nn' or the function itself, to be applied to final output values only. Defaults to the same as activation.
            padding (str, optional): What type of padding to use in convolutions. Defaults to 'valid'.
            residual (bool, optional): Whether to make the blocks calculate the residual. Defaults to False.
            padding_mode (str, optional): What values to use in padding (i.e. 'zeros', 'reflect', 'wrap', etc.). Defaults to 'reflect'.
            norm_layer (callable or None, optional): Whether to use a normalization layer and if so (i.e. if not None), the layer to use. Defaults to None.
            dropout_prob (float, optional): Dropout probability. Defaults to None.
            squeeze_excitation (bool, optional): Whether to use a squeeze-and-excitation block. Defaults to False.
            squeeze_ratio (int, optional): Ratio of squeeze channels to input channels. Defaults to 2.
            # deformable (bool, optional): Whether to use deformable convolutions. Defaults to False.

        Returns:
            ConvPass: Convolution block
        """
        super(ConvPass, self).__init__()

        if activation is not None:
            if isinstance(activation, str):
                self.activation = getattr(nn, activation)()
            else:
                self.activation = activation()  # assume is function
        else:
            self.activation = nn.Identity()

        if final_activation is not None:
            if isinstance(final_activation, str):
                self.final_activation = getattr(nn, final_activation)()
            else:
                self.final_activation = final_activation()  # assume is function
        else:
            self.final_activation = self.activation

        self.residual = residual
        self.padding = padding
        self.padding_mode = padding_mode
        self.norm_layer = norm_layer
        self.dropout_prob = dropout_prob
        self.squeeze_excitation = squeeze_excitation
        self.squeeze_ratio = squeeze_ratio
        # self.deformable = deformable
        if isinstance(norm_layer, str):
            try:
                if norm_layer == "batch":
                    norm_layer = {2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[
                        len(kernel_sizes[0])
                    ]
                elif norm_layer == "instance":
                    norm_layer = {2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[
                        len(kernel_sizes[0])
                    ]
                elif norm_layer == "group":
                    norm_layer = {2: nn.GroupNorm, 3: nn.GroupNorm}[
                        len(kernel_sizes[0])
                    ]
                else:
                    raise ValueError("Unknown normalization layer")
            except KeyError:
                raise RuntimeError(
                    "%dD normalization not implemented" % len(kernel_sizes[0])
                )

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.kernel_sizes = kernel_sizes

        layers = []

        for i, kernel_size in enumerate(kernel_sizes):
            # TODO: Use of BatchNorm does not work with bio-inspired learning rules
            if norm_layer is not None:
                layers.append(norm_layer(input_nc))

            if dropout_prob is not None and dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))

            self.dims = len(kernel_size)

            try:
                # TODO: Implement Conv4d
                # conv = {2: nn.Conv2d, 3: nn.Conv3d, 4: Conv4d}[self.dims]
                conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dims]
            except KeyError:
                raise ValueError(
                    # f"Only 2D, 3D and 4D convolutions are supported, not {self.dims}D"
                    f"Only 2D and 3D convolutions are supported not {self.dims}D"
                )

            layers.append(
                conv(
                    input_nc,
                    output_nc,
                    kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                )
            )
            if squeeze_excitation and i == len(kernel_sizes) // squeeze_ratio:
                try:
                    se_layer = {2: SqueezeExcitation, 3: SqueezeExcitation3d}[self.dims]
                except KeyError:
                    raise ValueError(
                        f"Only 2D and 3D squeeze-and-excitation blocks are supported not {self.dims}D"
                    )
                layers.append(se_layer(output_nc, output_nc // squeeze_ratio))
            if residual and i == 0:
                if input_nc < output_nc and output_nc % input_nc == 0:
                    groups = input_nc
                elif input_nc % output_nc == 0:
                    groups = output_nc
                else:
                    groups = 1
                self.x_init_map = conv(
                    input_nc,
                    output_nc,
                    np.ones(self.dims, dtype=int),
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=False,
                    groups=groups,
                )
            elif i < len(kernel_sizes) - 1:
                layers.append(self.activation)

            input_nc = output_nc

        self.conv_pass = nn.Sequential(*layers)

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.shape[: -self.dims] + tuple(shape)

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, x):
        if not self.residual:
            x = self.conv_pass(x)
            return self.final_activation(x)
        else:
            res = self.conv_pass(x)
            if self.padding.lower() == "valid":
                init_x = self.crop(self.x_init_map(x), res.size()[-self.dims :])
            else:
                init_x = self.x_init_map(x)
            return self.final_activation(res + init_x)


class SqueezeExcitation3d(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv3d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input
