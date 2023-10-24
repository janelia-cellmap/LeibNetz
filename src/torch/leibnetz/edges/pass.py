from .edge import Edge


class Pass(Edge):
    def compute_minimal_shapes(self):
        return (
            self.input_node.min_output_shape,
            self.input_node.step_valid_shape,
            self.input_node.min_output_shape,
        )

    def forward(self):
        if self.input_node.output_buffer is None:
            self.input_node.forward()
        self.output_node.add_input(**self.input_node.output_buffer)
