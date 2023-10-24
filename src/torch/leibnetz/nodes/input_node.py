from abc import abstractmethod
from torch.nn import Module


# defines baseclass for all nodes in the network
class InputNode(Module):
    def __init__(self, resolution=(1, 1, 1), identifier=None) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self.type = __name__.split(".")[-1]
        self.resolution = resolution
        self.ndims = len(resolution)
        (
            self.min_input_shape,
            self.step_valid_shape,
            self.min_output_shape,
        ) = self.compute_minimal_shapes()

    def forward(self):
        # implement any parsing of input/output buffers here
        # buffers are dictionaries
        self.output_buffer = self.input_buffer

    def compute_minimal_shapes(self):
        return (1,) * self.ndims, (1,) * self.ndims, (1,) * self.ndims
