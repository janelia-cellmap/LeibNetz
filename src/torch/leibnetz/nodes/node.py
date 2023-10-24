from abc import abstractmethod
from torch.nn import Module


# defines baseclass for all nodes in the network
class Node(Module):
    def __init__(self, model, resolution=(1, 1, 1), identifier=None) -> None:
        super().__init__()
        if identifier is None:
            identifier = id(self)
        self.id = identifier
        self.type = __name__.split(".")[-1]
        self.model = model
        self.resolution = resolution
        self.ndims = len(resolution)
        (
            self.min_input_shape,
            self.step_valid_shape,
            self.min_output_shape,
        ) = self.compute_minimal_shapes()

    def forward(self):
        # implement any parsing of input/output buffers here
        # buffers are dictionaries
        self.output_buffer = {self.id: self._model(**self.input_buffer)}

    def add_input(self, **inputs):
        self.input_buffer.update(inputs)

    def clear_buffer(self):
        self.input_buffer = {}
        self.output_buffer = None

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
