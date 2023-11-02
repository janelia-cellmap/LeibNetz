import networkx as nx
from torch.nn import Module
import matplotlib.pyplot as plt
import numpy as np

import logging

logger = logging.getLogger(__name__)


class LeibNet(Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.graph = nx.DiGraph()
        self.assemble()
        self.retain_buffer = False

    def assemble(self):
        # verify that all nodes and edges are unique
        node_ids = []
        output_to_node_id = {}
        for node in self.nodes:
            if not isinstance(node, Node):
                msg = f"{node} is not a Node."
                raise ValueError(msg)
            node_ids.append(node.id)
            for output in node.output_keys:
                if output in output_to_node_id:
                    msg = f"Output {output} is not unique."
                    raise ValueError(msg)
                output_to_node_id[output] = node.id

        if len(node_ids) != len(set(node_ids)):
            msg = "Node identifiers are not unique."
            raise ValueError(msg)

        # add nodes to graph
        self.graph.add_nodes_from(self.nodes)

        # add edges to graph
        for node in self.nodes:
            for input_key in node.input_keys:
                if input_key in output_to_node_id:
                    self.graph.add_edge(
                        self.graph.nodes[output_to_node_id[input_key]], node
                    )

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

        # collect output_keys
        self.output_keys = []
        for node in self.output_nodes:
            self.output_keys += node.output_keys

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
        
        # determine order of nodes
        self.ordered_nodes = list(nx.topological_sort(self.graph))

        # determine arrays that can be dropped after each node is run
        self.drop_arrays = {}
        for node in self.ordered_nodes:
            self.drop_arrays[node] = []
            for ancestor in nx.ancestors(self.graph, node):
                if hasattr(ancestor, "output_keys"):
                    self.drop_arrays[node] += ancestor.output_keys
                    
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
        # inputs is a dictionary of tensors
        # outputs is a dictionary of tensors

        # check if inputs are valid
        # if not self.is_valid_input_shape(inputs):
        #     msg = f"{inputs} is not a valid input shape."
        #     raise ValueError(msg)

        # initialize buffer
        self.buffer = inputs

        # march along nodes based on graph succession
        for nodes in nx.topological_generations(self.graph): # this isn't the proper order
            for node in nodes:
                self.buffer.update(node.forward(self.buffer[node.input_keys]))
            
            # need to clear unnecessary buffers
            if not self.retain_buffer:


        # collect outputs
        outputs = {}
        for output_key in self.output_keys:
            outputs.update(self.buffer[output_key])


        return outputs

    def draw(self):
        nx.draw(self.graph)
        plt.show()
