from abc import abstractmethod
from torch.nn import Module


# defines baseclass for all edges in the network
class Edge(Module):
    # NOTE: Edges can change voxel resolution
    def __init__(self, input_node, output_node, model, identifier=None) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self._type = __name__.split(".")[-1]
        self.input_node = input_node
        self.output_node = output_node
        self.model = model
        self.ndims = self.input_node.ndims
        (
            self.min_input_shape,
            self.step_valid_shape,
            self.min_output_shape,
        ) = self.compute_minimal_shapes()

    def forward(self):
        if self.input_node.output_buffer is None:
            self.input_node.forward()
        output = {self.id: self.model(**self.input_node.output_buffer)}
        self.output_node.add_input(**output)

    @abstractmethod
    def compute_minimal_shapes(self):
        ...
        return min_input_shape, step, min_output_shape

    def is_valid_input_shape(self, input_shape):
        return (input_shape >= self.min_input_shape).all() and (
            (input_shape - self.min_input_shape) % self.step_valid_shape == 0
        ).all()

    def get_input_from_output(self, output_shape):
        input_shape = output_shape + self.min_input_shape - self.min_output_shape
        if not self.is_valid_input_shape(input_shape):
            msg = f"{output_shape} is not valid output_shape."
            raise ValueError(msg)
        return input_shape

    def get_output_from_input(self, input_shape):
        if not self.is_valid_input_shape(input_shape):
            msg = f"{input_shape} is not a valid input shape."
            raise ValueError(msg)
        output_shape = input_shape - self.min_input_shape + self.min_output_shape
        return output_shape
