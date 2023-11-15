import numpy as np
import torch
from .resample_ops import Upsample, ConvDownsample
from .node_ops import ConvPass


class ConvResampleNode(torch.nn.Module):
    def __init__(
        self,
        input_keys,
        output_keys,
        scale_factor=(1, 1, 1),
        kernel_sizes=None,
        input_nc=1,
        output_nc=1,
        identifier=None,
    ) -> None:
        super().__init__(input_keys, output_keys, identifier)
        self._type = __name__.split(".")[-1]
        self.kernel_sizes = kernel_sizes
        self.output_nc = output_nc
        self.scale_factor = np.array(scale_factor)
        self.input_nc = input_nc
        if np.all(self.scale_factor == 1):
            self.model = ConvPass(self.input_nc, self.output_nc, self.kernel_sizes)
            self._type = "conv_pass"
        elif np.all(self.scale_factor <= 1):
            self.model = ConvDownsample(
                self.input_nc,
                self.output_nc,
                self.kernel_sizes,
                (1 / self.scale_factor).astype(int),
            )
            self._type = "conv_downsample"
        elif np.all(self.scale_factor >= 1):
            self.model = Upsample(
                self.scale_factor.astype(int),
                mode="transposed_conv",
                input_nc=self.input_nc,
                output_nc=self.output_nc,
            )
            self._type = "conv_upsample"
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
