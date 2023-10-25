from edge import Edge


class Skip(Edge):
    # NOTE: Edges can change voxel resolution
    def __init__(self, input_node, output_node, identifier=None) -> None:
        super().__init__(input_node, output_node, model=None, identifier=identifier)

    def forward(self):
        if self.input_node.output_buffer is None:
            self.input_node.forward()
        self.output_node.add_input(**self.input_node.output_buffer)
