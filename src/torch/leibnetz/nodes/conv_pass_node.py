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
        resolution=(1, 1, 1),
        identifier=None,
    ) -> None:
        super().__init__(output_keys, resolution, identifier)
        self.input_keys = input_keys
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
        self.compute_minimal_shapes()

    def forward(self, **inputs):
        # implement any parsing of input/output buffers here
        # buffers are dictionaries

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

    def compute_minimal_shapes(self):
        # TODO: inlcude crop_to_factor in this computation
        lost_voxels = np.zeros(self.ndims, dtype=int)
        kernel_sizes = []
        for i, module in enumerate(self.model.modules()):
            if i == 0:
                self.input_nc = module.input_nc
            if hasattr(module, "padding") and module.padding == "same":
                continue
            if hasattr(module, "kernel_size"):
                lost_voxels += np.array(module.kernel_size) - 1
                kernel_sizes.append(module.kernel_size)
            if hasattr(module, "stride"):
                # not sure what to do here...
                ...
            if hasattr(module, "dilation"):
                # not sure what to do here...
                ...
            if hasattr(module, "output_padding"):
                # not sure if this is correct...
                lost_voxels -= np.array(module.output_padding) * 2
            if hasattr(module, "out_channels"):
                self.output_nc = module.out_channels
        self.kernel_sizes = kernel_sizes
        self.min_input_voxels = lost_voxels + 1
        self.min_input_shape = self.min_input_voxels * self.resolution
        self.min_output_shape = self.step_valid_shape = self.resolution

    def is_valid_input_shape(self, input_shape):
        return (input_shape >= self.min_input_shape).all() and (
            (input_shape - self.min_input_shape) % self.step_valid_shape == 0
        ).all()

    def get_input_from_output(self, output_shape):
        input_shape = output_shape + self.min_input_shape - self.min_output_shape
        if not self.is_valid_input_shape(input_shape):
            raise ValueError(f"{output_shape} is not valid output_shape.")
        return input_shape

    def get_output_from_input(self, input_shape):
        if not self.is_valid_input_shape(input_shape):
            msg = f"{input_shape} is not a valid input shape."
            raise ValueError(msg)
        output_shape = input_shape - self.min_input_shape + self.min_output_shape
        return output_shape

    def crop_to_factor(self, x):
        """Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        """

        factor = self._least_common_resolution / self.resolution
        shape = x.size()
        spatial_shape = shape[-self.ndims :]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in self.kernel_sizes) for d in range(self.ndims)
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
                "convolutions %s" % (shape, factor, self.kernel_sizes)
            )

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.ndims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]
