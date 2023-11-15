from typing import Iterable, Sequence, Tuple
import numpy as np
import torch
from .resample_ops import Upsample, MaxDownsample
from . import Node


class ResampleNode(Node):
    def __init__(
        self, input_keys, output_keys, scale_factor=(1, 1, 1), identifier=None
    ) -> None:
        super().__init__(input_keys, output_keys, identifier)
        self._type = __name__.split(".")[-1]
        self._scale_factor = scale_factor

        if np.all(self.scale_factor == 1):
            self.model = torch.nn.Identity()
            self._type = "skip"
        elif np.all(self.scale_factor <= 1):
            self.model = MaxDownsample((1 / self.scale_factor).astype(int))
            self._type = "max_downsample"
        elif np.all(self.scale_factor >= 1):
            self.model = Upsample(self.scale_factor.astype(int))
            self._type = "upsample"
        else:
            raise NotImplementedError(
                "Simultaneous up- and downsampling not implemented"
            )

    @property
    def scale_factor(self):
        if self._scale_factor is not Iterable:
            self._scale_factor = (self._scale_factor,) * self.ndims
        if self._scale_factor is not np.ndarray:
            self._scale_factor = np.array(self._scale_factor)
        return self._scale_factor

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return {key: val for key, val in zip(self.output_keys, outputs)}

    def get_input_from_output_shape(self, output_shape) -> dict[str, Sequence]:
        output_shape = np.array(output_shape)
        return {
            key: (np.ceil(output_shape / self.scale_factor), self.scale_factor)
            for key in self.input_keys
        }

    def get_output_from_input_shape(self, input_shape) -> dict[str, Sequence]:
        input_shape = np.array(input_shape)
        assert np.all(
            (input_shape * self.scale_factor) % 1 == 0
        ), f"Input shape {input_shape} is not valid for scale factor {self.scale_factor}."
        return {
            key: ((input_shape * self.scale_factor).astype(int), self.scale_factor)
            for key in self.output_keys
        }
