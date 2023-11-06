import numpy as np
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
        outputs = self.model(**inputs)
        return {key: val for key, val in zip(self.output_keys, outputs)}

    def compute_minimal_shapes(self):
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
            msg = f"{output_shape} is not valid output_shape."
            raise ValueError(msg)
        return input_shape

    def get_output_from_input(self, input_shape):
        if not self.is_valid_input_shape(input_shape):
            msg = f"{input_shape} is not a valid input shape."
            raise ValueError(msg)
        output_shape = input_shape - self.min_input_shape + self.min_output_shape
        return output_shape
