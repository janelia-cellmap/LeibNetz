import torch
from edge_ops import Upsample, MaxDownsample


# defines baseclass for all edges in the network
class Edge(torch.nn.Module):
    # NOTE: Edges can change voxel resolution
    def __init__(self, input_node, output_node, identifier=None) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self._type = __name__.split(".")[-1]
        self.input_node = input_node
        self.output_node = output_node
        self.ndims = input_node.ndims
        self.scale_factor = output_node.resolution / input_node.resolution
        if all(self.scale_factor == 1):
            self.model = torch.nn.Identity()
            self._type = "skip"
        elif all(self.scale_factor <= 1):
            self.model = MaxDownsample((1 / self.scale_factor).astype(int))
            self._type = "max_downsample"
        elif all(self.scale_factor >= 1):
            self.next_conv_kernel_sizes = output_node.kernel_sizes
            self._model = lambda crop_factor: Upsample(
                self.scale_factor.astype(int),
                crop_factor=crop_factor,
                next_conv_kernel_sizes=self.next_conv_kernel_sizes,
            )
            self.model = None
            self._type = "upsample"
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
