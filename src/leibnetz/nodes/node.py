from abc import abstractmethod
from typing import Any, Iterable, Sequence, Tuple

import numpy as np
from torch.nn import Module


# defines baseclass for all nodes in the network
class Node(Module):
    """Base class for all neural network nodes in LeibNetz.

    All LeibNetz nodes inherit from this class, which provides the fundamental interface
    for shape calculation, cropping operations, and forward pass functionality.
    Nodes can be composed together to create complex neural network architectures.

    Args:
        input_keys: List of input key names this node expects.
        output_keys: List of output key names this node produces.
        identifier: Optional unique identifier for the node.
    """

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
        scale = np.array(scale)
        assert all(scale > 0)
        self._scale = scale
        self._ndims = len(scale)

    def set_least_common_scale(self, least_common_scale):
        least_common_scale = np.array(least_common_scale)
        assert all(least_common_scale > 0)
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
    def forward(self, inputs):
        raise NotImplementedError

    def get_input_from_output(
        self, outputs: dict[str, Sequence[Tuple]]
    ) -> dict[str, Sequence[Tuple]]:
        shapes = []
        scales = []
        for val in outputs.values():
            if val is None or val[1] is None or any(np.array(val[1]) <= 0):
                continue
            shapes.append(val[0])
            scales.append(val[1])
        # shapes, scales = zip(*outputs.values())
        # factor = np.lcm.reduce(
        #     [self.least_common_scale.astype(int)] + list(np.ceil(scales).astype(int))
        # )
        # assert np.all(factor > 0)
        output_shape = np.max(shapes, axis=0)
        # output_shape = np.ceil(output_shape / factor) * factor
        output_shape = output_shape * self.scale  # in world coordinates
        output_shape = (
            np.ceil(output_shape / self.least_common_scale) * self.least_common_scale
        )  # expanded to fit least common scale
        output_shape = output_shape / self.scale  # in voxel coordinates
        assert (
            np.ceil(output_shape) == output_shape
        ).all(), f"Output shape {output_shape} is not valid at node ID {self.id}."
        assert (
            len(output_shape) == self.ndims
        ), f"Input shape {output_shape} has wrong dimensionality. Expected to match spatial dimensions ({self.ndims})."
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
        shapes = []
        scales = []
        for k, (sh, sc) in inputs.items():
            assert (
                sh % 1 == 0
            ).all(), (
                f"Non-integer input shape {sh} for input key {k} at node ID {self.id}."
            )
            shapes.append(sh)
            scales.append(sc)
        # factor = np.lcm.reduce([self.least_common_scale] + list(scales))
        input_shape = np.min(shapes, axis=0)
        # input_shape = np.floor(input_shape / factor) * factor
        assert (
            len(input_shape) == self.ndims
        ), f"Input shape {input_shape} has wrong dimensionality. Expected to match spatial dimensions ({self.ndims})."
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
