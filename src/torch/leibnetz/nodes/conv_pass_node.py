import math
import numpy as np
import torch
from . import Node, ConvPass


# defines baseclass for all nodes in the network
class ConvPassNode(Node):
    def __init__(
        self,
        input_keys,
        output_keys,
        input_nc,
        output_nc,
        kernel_sizes,
        output_key_channels=None,
        activation="ReLU",
        padding="valid",
        residual=False,
        padding_mode="reflect",
        norm_layer=None,
        identifier=None,
    ) -> None:
        super().__init__(input_keys, output_keys, identifier)
        self.output_key_channels = output_key_channels
        self._type = __name__.split(".")[-1]
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.padding = padding
        self.residual = residual
        self.padding_mode = padding_mode
        self.norm_layer = norm_layer
        self.model = ConvPass(
            input_nc,
            output_nc,
            kernel_sizes,
            activation=activation,
            padding=padding,
            residual=residual,
            padding_mode=padding_mode,
            norm_layer=norm_layer,
        )
        self.color = "#00FF00"
        self._convolution_crop = None

    def forward(self, **inputs):
        # implement any parsing of input/output buffers here
        # buffers are dictionaries

        # crop if necessary
        shapes = [inputs[key].shape for key in self.input_keys]
        smallest_shape = np.min(shapes, axis=0)
        for key in self.input_keys:
            if inputs[key].shape != smallest_shape:
                inputs[key] = self.crop(inputs[key], smallest_shape)
        # concatenate inputs to single tensor
        inputs = torch.cat([inputs[key] for key in self.input_keys], dim=1)

        # crop inputs to ensure translation equivariance
        inputs = self.crop_to_factor(inputs)
        outputs = self.model(inputs)

        # split outputs into separate tensors
        if self.output_key_channels is not None:
            outputs = torch.split(outputs, self.output_key_channels, dim=1)
        elif len(self.output_keys) > 1:
            outputs = torch.split(outputs, len(self.output_keys), dim=1)
        return {key: val for key, val in zip(self.output_keys, outputs)}

    # the crop that will already be done due to the convolutions
    @property
    def convolution_crop(self):
        if self._convolution_crop is None:
            lost_voxels = np.zeros(self.ndims, dtype=int)
            for module in self.model.modules():
                if hasattr(module, "padding") and module.padding == "same":
                    continue
                if hasattr(module, "kernel_size"):
                    lost_voxels += np.array(module.kernel_size) - 1
            self._convolution_crop = lost_voxels
        return self._convolution_crop

    def get_input_from_output_shape(self, output_shape):
        input_shape = output_shape + self.convolution_crop
        return {key: (input_shape, (1,) * self.ndims) for key in self.input_keys}

    def get_output_from_input_shape(self, input_shape):
        output_shape = (
            input_shape - self.factor_crop(input_shape) - self.convolution_crop
        )
        return {key: (output_shape, (1,) * self.ndims) for key in self.output_keys}

    def factor_crop(self, input_shape):
        """Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        """
        # we need (spatial_shape - self.convolution_crop) to be a multiple of
        # self.least_common_scale, i.e.:
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
            for s, c, f in zip(
                input_shape, self.convolution_crop, self.least_common_scale
            )
        )
        target_spatial_shape = tuple(
            n * f + c
            for n, c, f in zip(ns, self.convolution_crop, self.least_common_scale)
        )

        return input_shape - target_spatial_shape

    def crop_to_factor(self, x):
        shape = x.size()
        spatial_shape = shape[-self.ndims :]
        target_spatial_shape = spatial_shape - self.factor_crop(spatial_shape)
        if target_spatial_shape != spatial_shape:
            assert all(
                ((t > c) for t, c in zip(target_spatial_shape, self.convolution_crop))
            ), (
                "Feature map with shape %s is too small to ensure "
                "translation equivariance with self.least_common_scale %s and following "
                "convolutions %s" % (shape, self.least_common_scale, self.kernel_sizes)
            )

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.ndims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]
