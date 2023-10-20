# adapted from funlib.learn.torch.models

from funlib.learn.torch.models.conv4d import Conv4d
import math
import numpy as np
import torch
import torch.nn as nn



class UNet(torch.nn.Module):
    def __init__(
        self,
        input_nc,
        ngf,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down=None,
        kernel_size_up=None,
        activation="ReLU",
        output_nc=None,
        num_heads=1,
        constant_upsample=False,
        downsample_method="max",
        padding_type="valid",
        residual=False,
        norm_layer=None,
        add_noise=False,
    ):
        """Create a U-Net::

            f_in --> f_left --------------------------->> f_right--> f_out
                        |                                   ^
                        v                                   |
                     g_in --> g_left ------->> g_right --> g_out
                                 |               ^
                                 v               |
                                       ...

        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively.

        The U-Net expects 3D or 4D tensors shaped like::

            ``(batch=1, channels, [length,] depth, height, width)``.

        It will perform 4D convolutions as long as ``length`` is greater than 1.
        As soon as ``length`` is 1 due to a valid convolution, the time dimension will be
        dropped and tensors with ``(b, c, z, y, x)`` will be use (and returned)
        from there on.

        Args:

            input_nc:

                The number of input channels.

            ngf:

                The number of feature maps in the first layer. By default, this is also the
                number of output feature maps. Stored in the ``channels``
                dimension.

            fmap_inc_factor:

                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``.

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

            kernel_size_down (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the left side. Kernel sizes
                can be given as tuples or integer. If not given, each
                convolutional pass will consist of two 3x3x3 convolutions.

            kernel_size_up (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the right side. Within one
                of the lists going from left to right. Kernel sizes can be
                given as tuples or integer. If not given, each convolutional
                pass will consist of two 3x3x3 convolutions.

            activation:

                Which activation to use after a convolution. Accepts the name
                of any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).

            output_nc (optional):

                The number of feature maps in the output layer. By default, this is the same as the
                number of feature maps of the input layer. Stored in the ``channels``
                dimension.

            num_heads (optional):

                Number of decoders. The resulting U-Net has one single encoder
                path and num_heads decoder paths. This is useful in a
                multi-task learning context.

            constant_upsample (optional):

                If set to true, perform a constant upsampling instead of a
                transposed convolution in the upsampling layers.

            downsample_method (optional):

                Whether to use max pooling ('max') or strided convolution ('convolve') for downsampling layers. Default is max pooling.

            padding_type (optional):

                How to pad convolutions. Either 'same' or 'valid' (default).

            residual (optional):

                Whether to train convolutional layers to output residuals to add to inputs (as in ResNet) or to directly convolve input data to output. Either 'True' or 'False' (default).

            norm_layer (optional):

                What, if any, normalization to layer after network layers. Default is none.

            add_noise (optional):

                Whether to add gaussian noise with 0 mean and unit variance ('True'), mean and variance parameterized by the network ('param'), or no noise ('False' <- default).

        """

        super(UNet, self).__init__()

        self.ndims = len(downsample_factors[0])
        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.input_nc = input_nc
        self.output_nc = output_nc if output_nc else ngf
        self.residual = residual
        if add_noise == "param":  # add noise feature if necessary
            self.noise_layer = ParameterizedNoiseBlock()
        elif add_noise:
            self.noise_layer = NoiseBlock()
        else:
            self.noise_layer = None
        # default arguments

        if kernel_size_down is None:
            kernel_size_down = [
                [(3,) * self.ndims, (3,) * self.ndims]
            ] * self.num_levels
        if kernel_size_up is None:
            kernel_size_up = [[(3,) * self.ndims, (3,) * self.ndims]] * (
                self.num_levels - 1
            )

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if padding_type.lower() == "valid":
                if factor_product is None:
                    factor_product = list(factor)
                else:
                    factor_product = list(
                        f * ff for f, ff in zip(factor, factor_product)
                    )
            elif padding_type.lower() == "same":
                factor_product = None
            else:
                raise f"Invalid padding_type option: {padding_type}"
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        # left convolutional passes
        self.l_conv = nn.ModuleList(
            [
                ConvPass(
                    input_nc
                    if level == 0
                    else ngf
                    * fmap_inc_factor ** (level - (downsample_method.lower() == "max")),
                    ngf * fmap_inc_factor**level,
                    kernel_size_down[level],
                    activation=activation,
                    padding=padding_type,
                    residual=self.residual,
                    norm_layer=norm_layer,
                )
                for level in range(self.num_levels)
            ]
        )
        self.dims = self.l_conv[0].dims

        # left downsample layers
        if downsample_method.lower() == "max":

            self.l_down = nn.ModuleList(
                [
                    MaxDownsample(downsample_factors[level])
                    for level in range(self.num_levels - 1)
                ]
            )

        elif downsample_method.lower() == "convolve":

            self.l_down = nn.ModuleList(
                [
                    ConvDownsample(
                        ngf * fmap_inc_factor**level,
                        ngf * fmap_inc_factor ** (level + 1),
                        kernel_size_down[level][0],
                        downsample_factors[level],
                        activation=activation,
                        padding=padding_type,
                        norm_layer=norm_layer,
                    )
                    for level in range(self.num_levels - 1)
                ]
            )

        else:

            raise RuntimeError(
                f'Unknown downsampling method {downsample_method}. Use "max" or "convolve" instead.'
            )

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Upsample(
                            downsample_factors[level],
                            mode="nearest" if constant_upsample else "transposed_conv",
                            input_nc=ngf * fmap_inc_factor ** (level + 1)
                            + (
                                level == 1 and (add_noise is not False)
                            ),  # TODO Fix NoiseBlock addition...
                            output_nc=ngf * fmap_inc_factor ** (level + 1),
                            crop_factor=crop_factors[level],
                            next_conv_kernel_sizes=kernel_size_up[level],
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )

        # right convolutional passes
        self.r_conv = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvPass(
                            ngf * fmap_inc_factor**level
                            + ngf * fmap_inc_factor ** (level + 1),
                            ngf * fmap_inc_factor**level
                            if output_nc is None or level != 0
                            else output_nc,
                            kernel_size_up[level],
                            activation=activation,
                            padding=padding_type,
                            residual=self.residual,
                            norm_layer=norm_layer,
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )

    def rec_forward(self, level, f_in):

        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:

            if self.noise_layer is not None:
                f_left = self.noise_layer(f_left)
            fs_out = [f_left] * self.num_heads

        else:

            # down
            g_in = self.l_down[i](f_left)

            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)

            # up, concat, and crop
            fs_right = [
                self.r_up[h][i](f_left, gs_out[h]) for h in range(self.num_heads)
            ]

            # convolve
            fs_out = [self.r_conv[h][i](fs_right[h]) for h in range(self.num_heads)]

        return fs_out

    def forward(self, x):

        y = self.rec_forward(self.num_levels - 1, x)

        if self.num_heads == 1:
            return y[0]

        return y