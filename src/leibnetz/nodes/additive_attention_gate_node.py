import math
import numpy as np
import torch
from leibnetz.nodes import Node
from leibnetz.nodes.node_ops import ConvPass


# TODO: Not 2D compatible
class AdditiveAttentionGateNode(Node):
    """Additive attention gate node for selectively emphasizing features.

    Implements attention mechanism from https://arxiv.org/pdf/1804.03999.pdf.
    Currently only supports 3D operations.

    Note: Not 2D compatible.

    Args:
        output_keys (list[str]): Keys for the output tensors.
        gating_key (str): Key for the gating signal input tensor.
        input_key (str): Key for the input features tensor.
        output_nc (int): Number of output channels.
        gating_nc (int): Number of channels in the gating signal.
        input_nc (int): Number of channels in the input features.
        identifier (str, optional): Unique identifier for the node. Defaults to None.
        ndims (int): Number of spatial dimensions (default: 3).
    """

    def __init__(
        self,
        output_keys,
        gating_key,
        input_key,
        output_nc,
        gating_nc,
        input_nc,
        identifier=None,
        ndims=3,
    ) -> None:
        """Attention Block Module.

            Implemented from paper: https://arxiv.org/pdf/1804.03999.pdf

            The attention block takes two inputs: 'g' (gating signal) and 'x' (input features).

                [g] --> W_g --\                 /--> psi --> * --> [output]
                                \               /
                [x] --> W_x --> [+] --> relu --

        Where:
        - W_g and W_x are 1x1 Convolution followed by Batch Normalization
        - [+] indicates element-wise addition
        - relu is the Rectified Linear Unit activation function
        - psi is a sequence of 1x1 Convolution, Batch Normalization, and Sigmoid activation
        - * indicates element-wise multiplication between the output of psi and input feature 'x'
        - [output] has the same dimensions as input 'x', selectively emphasized by attention weights

        Args:
        f_gating (int): The number of feature channels in the gating signal (g).
                   This is the input channel dimension for the W_g convolutional layer.

        f_left (int): The number of feature channels in the input features (x).
                   This is the input channel dimension for the W_x convolutional layer.

        f_intermediate (int): The number of intermediate feature channels.
                     This represents the output channel dimension of the W_g and W_x convolutional layers
                     and the input channel dimension for the psi layer. Typically, F_int is smaller
                     than F_g and F_l, as it serves to compress the feature representations before
                     applying the attention mechanism.

        The AttentionBlock uses two separate pathways to process 'g' and 'x', combines them,
        and applies a sigmoid activation to generate an attention map. This map is then used
        to scale the input features 'x', resulting in an output that focuses on important
        features as dictated by the gating signal 'g'.

        """

        super().__init__([input_key, gating_key], output_keys, identifier)
        self.gating_key = gating_key
        self.input_key = input_key
        self.gating_nc = gating_nc
        self.input_nc = input_nc
        self.output_nc = output_nc
        self._ndims = ndims
        self.kernel_sizes = [(1,) * self._ndims]
        self._type = __name__.split(".")[-1]
        self.color = "F00FF0F"

        self.W_g = ConvPass(
            self.gating_nc,
            self.output_nc,
            kernel_sizes=self.kernel_sizes,
            activation=None,
            padding="same",
        )

        self.W_x = ConvPass(
            self.input_nc,
            self.output_nc,
            kernel_sizes=self.kernel_sizes,
            activation=None,
            padding="same",
        )

        self.psi = ConvPass(
            self.output_nc,
            1,
            kernel_sizes=self.kernel_sizes,
            activation="Sigmoid",
            padding="same",
        )

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

    def forward(self, inputs):
        # implement any parsing of input/output buffers here
        # buffers are dictionaries

        # crop to same size if necessary
        inputs = self.get_min_crops(inputs)

        g = self.crop_to_factor(inputs[self.gating_key])
        x = self.crop_to_factor(inputs[self.input_key])
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # change it to crop
        # if np.all(x1.shape[-self.ndims :] != g1.shape[-self.ndims :]):
        smallest_shape = np.min(
            [x1.shape[-self.ndims :], g1.shape[-self.ndims :]], axis=0
        )
        assert len(smallest_shape) == self.ndims, (
            f"Input shapes {[x1.shape,g1.shape]} have wrong dimensionality for node {self.id}, "
            f"with expected inputs {self.input_keys} of dimensionality {self.ndims}"
        )
        # if np.all(x1.shape[-self.ndims :] != smallest_shape):
        x1 = self.crop(x1, smallest_shape)
        # if np.all(g1.shape[-self.ndims :] != smallest_shape):
        g1 = self.crop(g1, smallest_shape)
        psi = torch.nn.functional.relu(g1 + x1)
        psi = self.psi(psi)
        psi = torch.nn.functional.softmax(psi, dim=1)
        output = torch.matmul(x, psi)
        return {key: val for key, val in zip(self.output_keys, [output])}

    def get_input_from_output_shape(self, output_shape):
        input_shape = output_shape * self.scale  # to world coordinates
        input_shape = (
            np.ceil(input_shape / self.least_common_scale) * self.least_common_scale
        )  # expanded to fit least common scale
        input_shape = input_shape / self.scale  # to voxel coordinates
        assert (np.ceil(input_shape) == input_shape).all()
        return {key: (input_shape, self.scale) for key in self.input_keys}

    def get_output_from_input_shape(self, input_shape):
        output_shape = input_shape - self.factor_crop(input_shape)
        return {key: (output_shape, self.scale) for key in self.output_keys}

    def factor_crop(self, input_shape):
        """Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the resamples with the given kernel sizes.

        The crop could be done after the resamples, but it is more efficient
        to do that before (feature maps will be smaller).
        """
        # we need (spatial_shape - self.resample_crop) to be a multiple of
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
            int(math.floor(float(s) / f))
            for s, f in zip(
                spatial_shape,
                self.least_common_scale,
            )
        )
        target_spatial_shape = tuple(n * f for n, f in zip(ns, self.least_common_scale))

        return (spatial_shape - target_spatial_shape) / self.scale

    def crop_to_factor(self, x):
        shape = x.shape[-self.ndims :]
        target_shape = shape - self.factor_crop(shape)
        # if (target_shape != shape).all():
        return self.crop(x, target_shape.astype(int))

        return x

    def crop(self, x: torch.Tensor, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.ndims] + tuple(shape)

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]
