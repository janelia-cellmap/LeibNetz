import torch
from .resample_ops import Upsample, MaxDownsample
from . import Node


class ResampleNode(Node):
    def __init__(
        self, input_keys, output_keys, resolution=(1, 1, 1), identifier=None
    ) -> None:
        # NOTE: resolution is the resolution of the output of this node
        super().__init__(output_keys, resolution, identifier)
        self._type = __name__.split(".")[-1]
        self.input_keys = input_keys
        self.scale_factor = None
        self.input_resolution = None

    def set_resample_params(self, input_resolution):
        self.input_resolution = input_resolution
        self.scale_factor = self.resolution / self.input_resolution
        if all(self.scale_factor == 1):
            self.model = torch.nn.Identity()
            self._type = "skip"
        elif all(self.scale_factor <= 1):
            self.model = MaxDownsample((1 / self.scale_factor).astype(int))
            self._type = "max_downsample"
        elif all(self.scale_factor >= 1):
            self.model = Upsample(self.scale_factor.astype(int))
            self._type = "upsample"
        else:
            raise NotImplementedError(
                "Simultaneous up- and downsampling not implemented"
            )

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return {key: val for key, val in zip(self.output_keys, outputs)}

    def compute_minimal_shapes(self):
        raise NotImplementedError

    def is_valid_input_shape(self, input_shape):
        raise NotImplementedError

    def get_input_from_output(self, output_shape):
        raise NotImplementedError

    def get_output_from_input(self, input_shape):
        raise NotImplementedError
