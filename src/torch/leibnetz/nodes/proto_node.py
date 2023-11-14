from abc import abstractmethod
from typing import Any, Iterable, Union
import numpy as np
from torch.nn import Module


# defines class for fast prototyping of nodes
class ProtoNode(Module):
    id: Any
    output_keys: Iterable[str]
    _type: str

    def __init__(
        self,
        model,
        output_keys,
        output_key_channels=None,
        resolution=(1, 1, 1),
        identifier=None,
    ) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self.model = model
        self.output_keys = output_keys
        self.output_key_channels = output_key_channels
        self._type = __name__.split(".")[-1]
        self.resolution = np.array(resolution)
        self.ndims = len(resolution)
        self._least_common_resolution = None

    def forward(self, **inputs):
        # implement any parsing of input/output buffers here
        # buffers are dictionaries
        outputs = self.model(**inputs)

        # split outputs into separate tensors
        if self.output_key_channels is not None:
            outputs = torch.split(outputs, self.output_key_channels, dim=1)
        elif len(self.output_keys) > 1:
            outputs = torch.split(outputs, len(self.output_keys), dim=1)
        return {key: val for key, val in zip(self.output_keys, outputs)}

    @abstractmethod
    def compute_minimal_shapes(self):
        raise NotImplementedError

    @abstractmethod
    def is_valid_input_shape(self, input_key, input_shape):
        raise NotImplementedError

    @abstractmethod
    def get_input_from_output(self, output_shapes):
        raise NotImplementedError

    @abstractmethod
    def get_output_from_input(self, input_shapes):
        raise NotImplementedError
