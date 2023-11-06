import math
from torch import nn
import torch
from funlib.learn.torch.models.conv4d import Conv4d


class ConvDownsample(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        kernel_sizes,
        downsample_factor,
        activation="ReLU",
        padding="valid",
        padding_mode="reflect",
        norm_layer=None,
    ):
        """Convolution-based downsampling

        Args:
            input_nc (int): Number of input channels.
            output_nc (int): Number of output channels.
            kernel_sizes (list(int) or array_like): Kernel sizes for convolution layers.
            downsample_factor (int): Factor by which to downsample in all spatial dimensions.
            activation (str or callable): Name of activation function in 'nn' or the function itself.
            padding (str, optional): What type of padding to use in convolutions. Defaults to 'valid'.
            padding_mode (str, optional): What values to use in padding (i.e. 'zeros', 'reflect', 'wrap', etc.). Defaults to 'reflect'.
            norm_layer (callable or None, optional): Whether to use a normalization layer and if so (i.e. if not None), the layer to use. Defaults to None.

        Returns:
            Downsampling layer.
        """

        super(ConvDownsample, self).__init__()

        if activation is not None:
            if isinstance(activation, str):
                self.activation = getattr(nn, activation)()
            else:
                self.activation = activation()  # assume is function
        else:
            self.activation = nn.Identity()

        self.padding = padding

        layers = []

        self.dims = len(kernel_sizes)

        conv = {2: nn.Conv2d, 3: nn.Conv3d, 4: Conv4d}[self.dims]

        try:
            layers.append(
                conv(
                    input_nc,
                    output_nc,
                    kernel_sizes,
                    stride=downsample_factor,
                    padding="valid",
                    padding_mode=padding_mode,
                )
            )

        except KeyError:
            raise RuntimeError("%dD convolution not implemented" % self.dims)

        if norm_layer is not None:
            layers.append(norm_layer(output_nc))

        layers.append(self.activation)
        self.conv_pass = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_pass(x)


class MaxDownsample(nn.Module):
    def __init__(self, downsample_factor, flexible=True):
        """MaxPooling-based downsampling

        Args:
            downsample_factor (list(int) or array_like): Factors to downsample by in each dimension.
            flexible (bool, optional): True allows nn.MaxPoolNd to crop the right/bottom of tensors in order to allow pooling of tensors not evenly divisible by the downsample_factor. Alternative implementations could pass 'ceil_mode=True' or 'padding= {# > 0}' to avoid cropping of inputs. False forces inputs to be evenly divisible by the downsample_factor, which generally restricts the flexibility of model architectures. Defaults to True.

        Returns:
            Downsampling layer.
        """

        super(MaxDownsample, self).__init__()

        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor
        self.flexible = flexible

        pool = {
            2: nn.MaxPool2d,
            3: nn.MaxPool3d,
            4: nn.MaxPool3d,  # only 3D pooling, even for 4D input
        }[self.dims]

        self.down = pool(
            downsample_factor,
            stride=downsample_factor,
        )

    def forward(self, x):
        if self.flexible:
            try:
                return self.down(x)
            except:
                self.check_mismatch(x.size())
        else:
            self.check_mismatch(x.size())
            return self.down(x)

    def check_mismatch(self, size):
        for d in range(1, self.dims + 1):
            if size[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (size, self.downsample_factor, self.dims - d)
                )
        return


class Upsample(nn.Module):
    def __init__(
        self,
        scale_factor,
        mode=None,
        input_nc=None,
        output_nc=None,
        crop_factor=None,
        next_conv_kernel_sizes=None,
    ):
        super(Upsample, self).__init__()

        if crop_factor is not None:
            assert (
                next_conv_kernel_sizes is not None
            ), "crop_factor and next_conv_kernel_sizes have to be given together"

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes
        self.dims = len(scale_factor)

        if mode == "transposed_conv":
            up = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[self.dims]

            self.up = up(
                input_nc, output_nc, kernel_size=scale_factor, stride=scale_factor
            )

        else:
            self.up = nn.Upsample(scale_factor=tuple(scale_factor), mode=mode)

    def crop_to_factor(self, x, factor, kernel_sizes):
        """Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        """

        shape = x.size()
        spatial_shape = shape[-self.dims :]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes) for d in range(self.dims)
        )

        # we need (spatial_shape - convolution_crop) to be a multiple of
        # factor, i.e.:
        #
        # (s - c) = n*k
        #
        # we want to find the largest n for which s' = n*k + c <= s
        #
        # n = floor((s - c)/k)
        #
        # this gives us the target shape s'
        #
        # s' = n*k + c

        ns = (
            int(math.floor(float(s - c) / f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n * f + c for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:
            assert all(
                ((t > c) for t, c in zip(target_spatial_shape, convolution_crop))
            ), (
                "Feature map with shape %s is too small to ensure "
                "translation equivariance with factor %s and following "
                "convolutions %s" % (shape, factor, kernel_sizes)
            )

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.dims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, f_left, g_out):
        g_up = self.up(g_out)

        if self.crop_factor is not None:
            g_cropped = self.crop_to_factor(
                g_up, self.crop_factor, self.next_conv_kernel_sizes
            )
        else:
            g_cropped = g_up

        f_cropped = self.crop(f_left, g_cropped.size()[-self.dims :])

        return torch.cat([f_cropped, g_cropped], dim=1)
