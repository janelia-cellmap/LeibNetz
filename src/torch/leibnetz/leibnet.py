import networkx as nx
from torch.nn import Module

import logging

logger = logging.getLogger(__name__)


class LeibNet(Module):
    def __init__(self, nodes, edges):
        super().__init__()
        self.nodes = nodes
        self.edges = edges
        self.graph = nx.DiGraph()
        self.assemble()

    def get_ordered_edges(self):
        # get ordered edges based on graph succession
        ordered_edges = []
        for nodes in nx.topological_generations(self.graph):
            for node in nodes:
                ordered_edges.extend(
                    [
                        edge
                        for edge in self.graph.in_edges(node, data="edge")
                        if edge not in ordered_edges
                    ]
                )
        return ordered_edges

    def assemble(self):
        # verify that all nodes and edges are unique
        node_ids = [node.id for node in self.nodes]
        edge_ids = [edge.id for edge in self.edges]
        if len(node_ids) != len(set(node_ids)):
            msg = "Node identifiers are not unique."
            raise ValueError(msg)
        if len(edge_ids) != len(set(edge_ids)):
            msg = "Edge identifiers are not unique."
            raise ValueError(msg)

        # add nodes to graph
        self.graph.add_nodes_from(self.nodes)

        # add edges to graph
        for edge in self.edges:
            self.graph.add_edge(edge.input_node, edge.output_node, edge=edge)

        # check if graph is acyclic
        if not nx.is_directed_acyclic_graph(self.graph):
            msg = "Graph is not acyclic."
            raise ValueError(msg)

        # collect input nodes for later use
        self.input_nodes = [
            node for node in self.nodes if self.graph.in_degree(node) == 0
        ]
        # collect output nodes for later use
        self.output_nodes = [
            node for node in self.nodes if self.graph.out_degree(node) == 0
        ]
        # define edge call order
        self.ordered_edges = self.get_ordered_edges()

        (
            self.min_input_shape,
            self.step_valid_shape,
            self.min_output_shape,
        ) = self.compute_minimal_shapes()

    def compute_minimal_shapes(self):
        # analyze graph to determine minimal input/output shapes
        ...
        return min_input_shape, step, min_output_shape

    def is_valid_input_shape(self, input_shape):
        return (input_shape >= self.min_input_shape).all() and (
            (input_shape - self.min_input_shape) % self.step_valid_shape == 0
        ).all()

    def forward(self, **inputs):
        # function for forwarding data through the network
        # inputs is a dictionary of tensors, with keys corresponding to input node ids
        # outputs is a dictionary of tensors

        # check if inputs are valid
        if not self.is_valid_input_shape(inputs):
            msg = f"{inputs} is not a valid input shape."
            raise ValueError(msg)

        # initialize buffers
        for node in self.nodes:
            node.clear_buffer()

        # add inputs to appropriate buffers
        for node in self.input_nodes:
            node.add_input(**inputs[node.id])

        # march along edges based on graph succession
        for edge in self.ordered_edges:
            edge.forward()

        # collect outputs
        outputs = {}
        for node in self.output_nodes:
            outputs.update(node.output_buffer)

        return outputs

    def draw(self):
        nx.draw(self.graph)
