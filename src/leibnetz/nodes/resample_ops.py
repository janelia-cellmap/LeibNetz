from logging import getLogger

from torch import nn

# from funlib.learn.torch.models.conv4d import Conv4d


logger = getLogger(__name__)


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

        try:
            # TODO: Implement Conv4d
            # conv = {2: nn.Conv2d, 3: nn.Conv3d, 4: Conv4d}[self.dims]
            conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dims]
        except KeyError:
            raise ValueError(
                # f"Only 2D, 3D and 4D convolutions are supported, not {self.dims}D"
                f"Only 2D and 3D convolutions are supported, not {self.dims}D"
            )

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

    def _forward_single(self, x):
        return self.conv_pass(x)

    def forward(self, x):
        if isinstance(x, dict):
            return [
                self._forward_single(x[key]) for key in x.keys() if x[key] is not None
            ]
        else:
            return self._forward_single(x)


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
            tuple(downsample_factor),
        )

    def _forward_single(self, x):
        if self.flexible:
            try:
                return self.down(x)
            except Exception as e:
                self.check_mismatch(x.size(), e)
        else:
            self.check_mismatch(x.size())
            return self.down(x)

    def forward(self, x):
        if isinstance(x, dict):
            return [
                self._forward_single(x[key]) for key in x.keys() if x[key] is not None
            ]
        else:
            return self._forward_single(x)

    def check_mismatch(self, size, e: Exception | None = None):
        for d in range(1, self.dims + 1):
            if size[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (size, self.downsample_factor, self.dims - d)
                )
        if e is not None:
            logger.error(f"failed MaxDownsample with array size {size}")
            raise e
        raise RuntimeError("Unknown error during downsampling")


class Upsample(nn.Module):
    def __init__(
        self,
        scale_factor,
        mode="nearest",
        input_nc=None,
        output_nc=None,
    ):
        super(Upsample, self).__init__()

        self.dims = len(scale_factor)

        if mode == "transposed_conv":
            up = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[self.dims]

            self.up = up(
                input_nc, output_nc, kernel_size=scale_factor, stride=scale_factor
            )

        else:
            self.up = nn.Upsample(scale_factor=tuple(scale_factor), mode=mode)

    def _forward_single(self, x):
        return self.up(x)

    def forward(self, x):
        if isinstance(x, dict):
            return [
                self._forward_single(x[key]) for key in x.keys() if x[key] is not None
            ]
        else:
            return self._forward_single(x)
