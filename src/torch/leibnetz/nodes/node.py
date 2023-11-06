from abc import abstractmethod
import numpy as np
from torch.nn import Module


# defines baseclass for all nodes in the network
class Node(Module):
    output_nc: int

    def __init__(self, output_keys, resolution=(1, 1, 1), identifier=None) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self.output_keys = output_keys
        self._type = __name__.split(".")[-1]
        self.resolution = np.array(resolution)
        self.ndims = len(resolution)
        self._least_common_resolution = None

    @abstractmethod
    def forward(self, **inputs):
        raise NotImplementedError

    @abstractmethod
    def compute_minimal_shapes(self):
        raise NotImplementedError

    @abstractmethod
    def is_valid_input_shape(self, input_shape):
        raise NotImplementedError

    @abstractmethod
    def get_input_from_output(self, output_shape):
        raise NotImplementedError

    @abstractmethod
    def get_output_from_input(self, input_shape):
        raise NotImplementedError
