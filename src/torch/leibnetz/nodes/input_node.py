from torch.nn import Module
import numpy as np


# defines baseclass for all input nodes in the network
# NOTE: Each InputNode should have a unique identifier and take only one tensor as input
class InputNode(Module):  # TODO: should be subclass of Node?
    def __init__(self, resolution=(1, 1, 1), identifier=None) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self._type = __name__.split(".")[-1]
        self.resolution = np.array(resolution)
        self.ndims = len(resolution)
        self.compute_minimal_shapes()

    def add_input(self, inputs):
        self.input_buffer.update(inputs)

    def forward(self):
        # implement any parsing of input/output buffers here
        # buffers are dictionaries
        self.output_buffer = self.input_buffer

    def clear_buffer(self):
        self.input_buffer = {}
        self.output_buffer = None

    def compute_minimal_shapes(self):
        self.min_input_voxels = (1,) * self.ndims
        self.min_input_shape = self.min_input_voxels * self.resolution
        self.min_output_shape = self.step_valid_shape = self.resolution

    def is_valid_input_shape(self, input_shape):
        return (input_shape >= self.min_input_shape).all() and (
            (input_shape - self.min_input_shape) % self.step_valid_shape == 0
        ).all()

    def get_input_from_output(self, output_shape):
        return output_shape

    def get_output_from_input(self, input_shape):
        return input_shape
