from torch import nn
import numpy as np
from funlib.learn.torch.models.conv4d import Conv4d


class ConvPass(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        kernel_sizes,
        activation="ReLU",
        padding="valid",
        residual=False,
        padding_mode="reflect",
        norm_layer=None,
        dropout_prob=None,
    ):
        """Convolution pass block

        Args:
            input_nc (int): Number of input channels
            output_nc (int): Number of output channels
            kernel_sizes (list(int) or array_like): Kernel sizes for convolution layers.
            activation (str or callable): Name of activation function in 'nn' or the function itself.
            padding (str, optional): What type of padding to use in convolutions. Defaults to 'valid'.
            residual (bool, optional): Whether to make the blocks calculate the residual. Defaults to False.
            padding_mode (str, optional): What values to use in padding (i.e. 'zeros', 'reflect', 'wrap', etc.). Defaults to 'reflect'.
            norm_layer (callable or None, optional): Whether to use a normalization layer and if so (i.e. if not None), the layer to use. Defaults to None.
            dropout_prob (float, optional): Dropout probability. Defaults to None.

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

        self.residual = residual
        self.padding = padding
        self.padding_mode = padding_mode
        self.norm_layer = norm_layer
        self.dropout_prob = dropout_prob
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
            if norm_layer is not None:
                layers.append(norm_layer(input_nc))

            layers.append(self.activation)

            if dropout_prob is not None and dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))

            self.dims = len(kernel_size)

            conv = {2: nn.Conv2d, 3: nn.Conv3d, 4: Conv4d}[self.dims]

            try:
                layers.append(
                    conv(
                        input_nc,
                        output_nc,
                        kernel_size,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                )
                if residual and i == 0:
                    if input_nc < output_nc:
                        groups = input_nc
                    else:
                        groups = output_nc
                    self.x_init_map = conv(
                        input_nc,
                        output_nc,
                        np.ones(self.dims, dtype=int),
                        padding=padding,
                        padding_mode=padding_mode,
                        bias=False,
                        groups=groups,
                    )
            except KeyError:
                raise RuntimeError("%dD convolution not implemented" % self.dims)

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
            return self.conv_pass(x)
        else:
            res = self.conv_pass(x)
            if self.padding.lower() == "valid":
                init_x = self.crop(self.x_init_map(x), res.size()[-self.dims :])
            else:
                init_x = self.x_init_map(x)
            return res + init_x
