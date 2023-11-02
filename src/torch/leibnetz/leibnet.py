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
                        for _, _, edge in self.graph.in_edges(node, data="edge")
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

        # walk along input paths for each node to determine lowest common resolution
        for node in self.nodes:
            if node not in self.input_nodes:
                resolutions = []
                ancestors = nx.ancestors(self.graph, node)
                for ancestor in ancestors:
                    if hasattr(ancestor, "_least_common_resolution"):
                        resolutions.append(ancestor._least_common_resolution)
                        for elder in nx.ancestors(self.graph, ancestor):
                            ancestors.remove(elder)
                    else:
                        resolutions.append(ancestor.resolution)
                node._least_common_resolution = np.lcm.reduce(resolutions)
        # self.compute_minimal_shapes()

    def compute_minimal_shapes(self):
        # analyze graph to determine minimal input/output shapes
        raise NotImplementedError
        for edge in self.ordered_edges:
            if edge.model is None:
                edge.set_crop_factor(crop_factor)
        self.min_input_shape = ...
        self.step_valid_shape = ...
        self.min_output_shape = ...

    def is_valid_input_shape(self, input_shape):
        return (input_shape >= self.min_input_shape).all() and (
            (input_shape - self.min_input_shape) % self.step_valid_shape == 0
        ).all()

    def forward(self, **inputs):
        # function for forwarding data through the network
        # inputs is a dictionary of tensors, with keys corresponding to input node ids
        # outputs is a dictionary of tensors

        # check if inputs are valid
        # if not self.is_valid_input_shape(inputs):
        #     msg = f"{inputs} is not a valid input shape."
        #     raise ValueError(msg)

        # initialize buffers
        for node in self.nodes:
            node.clear_buffer()

        # add inputs to appropriate buffers
        for node in self.input_nodes:
            node.add_input({node.id: inputs[node.id]})

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


if __name__ == "__main__":
    # make a simple affinity prediction UNet
    from .nodes import InputNode, Node
    from .edges import Edge, Skip
    from src.torch.leibnetz.nodes.node_ops import ConvPass
    import numpy as np

    base_resolution = np.array([8, 8, 8])
    # define input node
    input_node = InputNode(resolution=base_resolution, identifier="input")

    # define nodes
    conv0_0 = Node(
        ConvPass(
            1,
            12,
            [
                (3,) * 3,
            ]
            * 2,
        ),
        resolution=base_resolution * 2**0,
        identifier="conv0_0",
    )
    conv1_0 = Node(
        ConvPass(
            12,
            24,
            [
                (3,) * 3,
            ]
            * 2,
        ),
        resolution=base_resolution * 2**1,
        identifier="conv1_0",
    )
    conv2 = Node(
        ConvPass(
            24,
            48,
            [
                (3,) * 3,
            ]
            * 2,
        ),
        resolution=base_resolution * 2**2,
        identifier="conv2",
    )
    conv0_1 = Node(
        ConvPass(
            24,
            12,
            [
                (3,) * 3,
            ]
            * 2,
        ),
        resolution=base_resolution * 2**0,
        identifier="conv0_1",
    )
    conv1_1 = Node(
        ConvPass(
            48,
            24,
            [
                (3,) * 3,
            ]
            * 2,
        ),
        resolution=base_resolution * 2**1,
        identifier="conv1_1",
    )

    # define input edge
    edge0_0 = Edge(input_node, conv0_0, identifier="edge0_0")

    # define downsample edges
    edge0_1 = Edge(conv0_0, conv1_0, identifier="edge0_1")
    edge1_0 = Edge(conv1_0, conv2, identifier="edge1_0")

    # define upsample edges
    edge1_1 = Edge(conv2, conv1_1, identifier="edge1_1")
    edge0_2 = Edge(conv1_1, conv0_1, identifier="edge0_2")

    # define skip connections
    edge0_3 = Edge(conv0_0, conv0_1, identifier="edge0_3")
    edge1_2 = Edge(conv1_0, conv1_1, identifier="edge1_2")

    # create leibnet
    leibnet = LeibNet(
        [input_node, conv0_0, conv1_0, conv2, conv0_1, conv1_1],
        [edge0_0, edge0_1, edge1_0, edge1_1, edge0_2, edge0_3, edge1_2],
    )
