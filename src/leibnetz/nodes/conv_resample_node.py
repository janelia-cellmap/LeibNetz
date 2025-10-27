import numpy as np
import torch
from leibnetz.nodes import Node
from leibnetz.nodes.resample_ops import Upsample, ConvDownsample
from leibnetz.nodes.node_ops import ConvPass


class ConvResampleNode(Node):
    def __init__(
        self,
        input_keys,
        output_keys,
        scale_factor=(1, 1, 1),
        kernel_sizes=None,
        input_nc=1,
        output_nc=1,
        identifier=None,
    ) -> None:
        super().__init__(input_keys, output_keys, identifier)
        self._type = __name__.split(".")[-1]
        self.kernel_sizes = kernel_sizes
        self.output_nc = output_nc
        self.scale_factor = np.array(scale_factor)
        self.input_nc = input_nc
        if np.all(self.scale_factor == 1):
            self.model = ConvPass(self.input_nc, self.output_nc, self.kernel_sizes)
            self.color = "#00FF00"
            self._type = "conv_pass"
        elif np.all(self.scale_factor <= 1):
            # ConvDownsample expects kernel_sizes as [3, 3, 3], not [[3, 3, 3]]
            if (
                self.kernel_sizes is not None
                and len(self.kernel_sizes) > 0
                and isinstance(self.kernel_sizes[0], list)
            ):
                downsample_kernel_sizes = self.kernel_sizes[0]
            else:
                downsample_kernel_sizes = self.kernel_sizes
            self.model = ConvDownsample(
                self.input_nc,
                self.output_nc,
                downsample_kernel_sizes,
                (1 / self.scale_factor).astype(int),
            )
            self.color = "#00FFFF"
            self._type = "conv_downsample"
        elif np.all(self.scale_factor >= 1):
            self.model = Upsample(
                self.scale_factor.astype(int),
                mode="transposed_conv",
                input_nc=self.input_nc,
                output_nc=self.output_nc,
            )
            self.color = "#FFFF00"
            self._type = "conv_upsample"
        else:
            raise NotImplementedError(
                "Simultaneous up- and downsampling not implemented"
            )

    def forward(self, inputs):
        # Extract the tensor from the inputs dict like other nodes do
        # ConvResampleNode expects a single input tensor
        input_tensor = inputs[self.input_keys[0]]
        outputs = self.model(input_tensor)

        # Handle the case where outputs might be a single tensor
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        return {key: val for key, val in zip(self.output_keys, outputs)}

    def get_input_from_output_shape(self, output_shape):
        """Calculate required input shape for given output shape"""
        # TODO: Implement proper shape calculation based on convolution and sampling
        # For now, return a placeholder implementation
        output_shape = np.array(output_shape)
        # Reverse the scaling operation
        input_shape = output_shape / self.scale_factor
        return {
            key: (input_shape, self.scale / self.scale_factor)
            for key in self.input_keys
        }

    def get_output_from_input_shape(self, input_shape):
        """Calculate output shape for given input shape"""
        # TODO: Implement proper shape calculation based on convolution and sampling
        # For now, return a placeholder implementation
        input_shape = np.array(input_shape)
        output_shape = input_shape * self.scale_factor
        return {key: (output_shape.astype(int), self.scale) for key in self.output_keys}
