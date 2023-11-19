from abc import abstractmethod
from typing import Any, Iterable, Sequence, Tuple
import numpy as np
from torch.nn import Module


# defines baseclass for all nodes in the network
class Node(Module):
    id: Any
    output_keys: Iterable[str]
    _type: str

    def __init__(self, input_keys, output_keys, identifier=None) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.color = "#000000"
        self._type = __name__.split(".")[-1]
        self._scale = None
        self._ndims = None
        self._least_common_scale = None

    def set_scale(self, scale):
        self._scale = scale
        self._ndims = len(scale)

    def set_least_common_scale(self, least_common_scale):
        self._least_common_scale = least_common_scale
        if self._ndims is None:
            self._ndims = len(least_common_scale)

    @property
    def scale(self):
        if self._scale is not None:
            return self._scale
        else:
            raise RuntimeError("Scale not set. Make sure graph & node are initialized.")

    @property
    def ndims(self):
        if self._ndims is not None:
            return self._ndims
        else:
            raise RuntimeError("Ndims not set. Make sure graph & node are initialized.")

    @property
    def least_common_scale(self):
        if self._least_common_scale is not None:
            return self._least_common_scale
        else:
            raise RuntimeError(
                "Least common scale not set. Make sure graph & node are initialized."
            )

    @abstractmethod
    def forward(self, **inputs):
        raise NotImplementedError

    def get_input_from_output(
        self, outputs: dict[str, Sequence[Tuple]]
    ) -> dict[str, Sequence[Tuple]]:
        shapes, scales = zip(*outputs.values())
        factor = np.lcm.reduce([self.least_common_scale] + list(scales))
        output_shape = np.max(shapes, axis=0)
        output_shape = np.ceil(output_shape / factor) * factor
        inputs = self.get_input_from_output_shape(output_shape)
        return inputs

    @abstractmethod
    def get_input_from_output_shape(
        self, output_shape: Tuple
    ) -> dict[str, Sequence[Tuple]]:
        raise NotImplementedError

    def get_output_from_input(
        self, inputs: dict[str, Sequence[Tuple]]
    ) -> dict[str, Sequence[Tuple]]:
        shapes, scales = zip(*inputs.values())
        # factor = np.lcm.reduce([self.least_common_scale] + list(scales))
        input_shape = np.min(shapes, axis=0)
        # input_shape = np.floor(input_shape / factor) * factor
        outputs = self.get_output_from_input_shape(input_shape)
        return outputs

    @abstractmethod
    def get_output_from_input_shape(
        self, input_shape: Tuple
    ) -> dict[str, Sequence[Tuple]]:
        raise NotImplementedError

    def check_input_shapes(self, inputs: dict):
        # check if inputs are valid
        shapes_valid = True
        for input_key, val in inputs.items():
            shapes_valid &= self.is_valid_input_shape(input_key, val.shape)
        return shapes_valid

    @abstractmethod
    def is_valid_input_shape(self, input_key, input_shape):
        raise NotImplementedError
