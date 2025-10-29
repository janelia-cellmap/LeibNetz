import math
import numpy as np
import torch
from leibnetz.nodes import Node


class WrapperNode(Node):
    """Wrapper node for integrating existing PyTorch modules as LeibNetz nodes.

    Allows wrapping arbitrary PyTorch modules to use them within the LeibNetz framework.
    Provides shape calculation capabilities for wrapped modules and handles cropping operations.

    Args:
        model: PyTorch module to wrap.
        input_keys: Input tensor keys.
        output_keys: Output tensor keys.
        identifier: Optional unique identifier for the node.
        output_key_channels: Optional channel specification per output key.
    """

    def __init__(
        self, model, input_keys, output_keys, identifier=None, output_key_channels=None
    ):
        super().__init__(input_keys, output_keys, identifier)
        self.model = model
        self.output_key_channels = output_key_channels
        self._type = __name__.split(".")[-1]
        self.color = "#FF0000"

    # the crop that will already be done due to the convolutions
    @property
    def convolution_crop(self):
        if not hasattr(self, "_convolution_crop"):
            lost_voxels = np.zeros(self.ndims, dtype=int)
            for module in self.model.modules():
                if hasattr(module, "padding") and module.padding == "same":
                    continue
                if hasattr(module, "kernel_size"):
                    lost_voxels += np.array(module.kernel_size) - 1
            assert (lost_voxels >= 0).all() & (
                lost_voxels % 1 == 0
            ).all(), f"Non-integer lost voxels {lost_voxels} at node {self.id}"
            self._convolution_crop = lost_voxels
        return self._convolution_crop

    def forward(self, inputs):
        # implement any parsing of input/output buffers here
        # buffers are dictionaries

        # crop to same size if necessary
        inputs = self.get_min_crops(inputs)
        # concatenate inputs to single tensor in the channel dimension
        inputs_tensor = torch.cat([inputs[key] for key in self.input_keys], dim=1)

        # crop inputs_tensor to ensure translation equivariance
        inputs_tensor = self.crop_to_factor(inputs_tensor)
        outputs = self.model(inputs_tensor)

        # split outputs into separate tensors
        if self.output_key_channels is not None:
            outputs = torch.split(outputs, self.output_key_channels, dim=1)
        elif len(self.output_keys) > 1:
            outputs = torch.split(outputs, len(self.output_keys), dim=1)
        else:
            outputs = [outputs]
        return {key: val for key, val in zip(self.output_keys, outputs)}

    def get_min_crops(self, inputs):
        shapes = [
            inputs[key].shape[-self.ndims :]
            for key in self.input_keys
            if inputs[key] is not None
        ]
        assert (
            len(shapes) > 0
        ), f"No inputs for node {self.id}, with expected inputs {self.input_keys}"
        smallest_shape = np.min(shapes, axis=0)
        # smallest_shape = torch.min(torch.as_tensor(shapes), dim=0)
        assert len(smallest_shape) == self.ndims, (
            f"Input shapes {shapes} have wrong dimensionality for node {self.id}, "
            f"with expected inputs {self.input_keys} of dimensionality {self.ndims}"
        )
        for key in self.input_keys:
            # NOTE: "if" omitted to allow torch tracing
            # if any(inputs[key].shape[-self.ndims :] != smallest_shape):
            inputs[key] = self.crop(inputs[key], smallest_shape)
        return inputs

    def get_input_from_output_shape(self, output_shape):
        input_shape = output_shape + self.convolution_crop
        input_shape = input_shape * self.scale  # to world coordinates
        input_shape = (
            np.ceil(input_shape / self.least_common_scale) * self.least_common_scale
        )  # expanded to fit least common scale
        input_shape = input_shape / self.scale  # to voxel coordinates
        assert (np.ceil(input_shape) == input_shape).all()
        # return {key: (input_shape, (1,) * self.ndims) for key in self.input_keys}
        return {key: (input_shape, self.scale) for key in self.input_keys}

    def get_output_from_input_shape(self, input_shape):
        output_shape = (
            input_shape - self.factor_crop(input_shape) - self.convolution_crop
        )
        # return {key: (output_shape, (1,) * self.ndims) for key in self.output_keys}
        return {key: (output_shape, self.scale) for key in self.output_keys}

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
        spatial_shape = input_shape[-self.ndims :] * self.scale
        ns = (
            int(math.floor(float(s - c) / f))
            for s, c, f in zip(
                spatial_shape,
                self.convolution_crop * self.scale,
                self.least_common_scale,
            )
        )
        target_spatial_shape = tuple(
            n * f + c
            for n, c, f in zip(
                ns, self.convolution_crop * self.scale, self.least_common_scale
            )
        )

        return (spatial_shape - target_spatial_shape) / self.scale

    def crop_to_factor(self, x: torch.Tensor):
        shape = x.shape[-self.ndims :]
        target_shape = shape - self.factor_crop(shape)
        # NOTE: "if" omitted to allow torch tracing
        # if any(target_shape != shape):
        assert all(((t >= c) for t, c in zip(target_shape, self.convolution_crop))), (
            "Feature map with shape %s is too small to ensure "
            "translation equivariance with self.least_common_scale %s and following "
            "convolutions %s" % (x.size(), self.least_common_scale, self.kernel_sizes)
        )

        return self.crop(x, target_shape.astype(int))
        # return x

    def crop(self, x: torch.Tensor, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.shape[: -self.ndims] + tuple(shape)

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]
