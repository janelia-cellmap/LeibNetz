import torch
from .edge_ops import Upsample, ConvDownsample
from leibnetz.nodes.node_ops import ConvPass


# defines baseclass for all edges in the network
class ConvEdge(torch.nn.Module):
    # NOTE: Edges can change voxel resolution
    def __init__(
        self, input_node, output_node, kernel_sizes=None, identifier=None
    ) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self._type = __name__.split(".")[-1]
        self.input_node = input_node
        self.output_node = output_node
        self.kernel_sizes = kernel_sizes
        self.input_nc = input_node.output_nc
        self.output_nc = output_node.input_nc
        self.ndims = input_node.ndims
        self.scale_factor = output_node.resolution / input_node.resolution
        if all(self.scale_factor == 1):
            self.model = ConvPass(self.input_nc, self.output_nc, self.kernel_sizes)
            self._type = "conv_pass"
        elif all(self.scale_factor <= 1):
            self.model = ConvDownsample(
                self.input_nc,
                self.output_nc,
                self.kernel_sizes,
                (1 / self.scale_factor).astype(int),
            )
            self._type = "conv_downsample"
        elif all(self.scale_factor >= 1):
            self.next_conv_kernel_sizes = output_node.kernel_sizes
            self._model = lambda crop_factor: Upsample(
                self.scale_factor.astype(int),
                mode="transposed_conv",
                input_nc=self.input_nc,
                output_nc=self.output_nc,
                crop_factor=crop_factor,
                next_conv_kernel_sizes=self.next_conv_kernel_sizes,
            )
            self.model = None
            self._type = "conv_upsample"
        else:
            raise NotImplementedError(
                "Simultaneous up- and downsampling not implemented"
            )

    def set_crop_factor(self, crop_factor):
        self.model = self._model(crop_factor)

    def forward(self):
        if self.input_node.output_buffer is None:
            self.input_node.forward()
        output = {self.id: self.model(**self.input_node.output_buffer)}
        self.output_node.add_input(**output)
