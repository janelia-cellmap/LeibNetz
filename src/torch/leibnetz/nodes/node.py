import numpy as np
from torch.nn import Module


# defines baseclass for all nodes in the network
class Node(Module):
    def __init__(
        self, model, input_keys, output_keys, resolution=(1, 1, 1), identifier=None
    ) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self.input_keys = input_keys
        self.output_keys = output_keys
        self._type = __name__.split(".")[-1]
        self.model = model
        self.resolution = np.array(resolution)
        self.ndims = len(resolution)
        self.compute_minimal_shapes()
        self._least_common_resolution = None

    def forward(self, **inputs):
        # implement any parsing of input/output buffers here
        # buffers are dictionaries
        outputs = self.model(**inputs)
        return {key: val for key, val in zip(self.output_keys, outputs)}

    # @property
    # def least_common_resolution(self):
    #     if self._least_common_resolution is None:
    #         self._least_common_resolution = np.lcm.reduce(ance)
    #     return self._least_common_resolution

    def compute_minimal_shapes(self):
        # TODO: this is designed assuming a convolutional model
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
