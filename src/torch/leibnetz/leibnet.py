import networkx as nx
from torch.nn import Module
import matplotlib.pyplot as plt
import numpy as np
from .nodes import Node

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
        self.output_to_node_id = output_to_node_id

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
                if hasattr(
                    node, "set_resample_params"
                ):  # TODO: decide how much require user to pass to node, and how much to infer
                    if hasattr(node, "input_nc"):
                        node.set_resample_params(
                            input_resolution=self.graph.nodes[
                                output_to_node_id[input_key]
                            ].resolution,
                            input_nc=self.graph.nodes[
                                output_to_node_id[input_key]
                            ].output_nc,
                        )
                    else:
                        node.set_resample_params(
                            input_resolution=self.graph.nodes[
                                output_to_node_id[input_key]
                            ].resolution
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

        # collect input_keys
        self.input_keys = []
        for node in self.input_nodes:
            self.input_keys += node.input_keys

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

        # determine quasi-optimal order of nodes (in reverse order)
        self.ordered_nodes = list(nx.topological_sort(self.graph.reverse()))

        # determine arrays that can be dropped after each node is run
        self.flushable_arrays = [
            [],
        ] * len(self.ordered_nodes)
        flushed_arrays = []
        for i, node in enumerate(self.ordered_nodes):
            for input_key in node.input_keys:
                if (
                    input_key not in flushed_arrays
                    and input_key not in self.output_keys
                ):
                    self.flushable_arrays[i].append(input_key)
                    flushed_arrays.append(input_key)

        # correct order of lists
        self.ordered_nodes = self.ordered_nodes[::-1]
        self.flushable_arrays = self.flushable_arrays[::-1]

        # self.compute_minimal_shapes()

    def compute_minimal_shapes(self):
        # analyze graph to determine minimal input/output shapes
        # first find minimal output shapes (1x1x1 at lowest resolution)
        # NOTE: expects the output_keys to come from nodes that have realworld unit resolutions (i.e. not classifications)
        least_common_resolutions = []
        for output_key in self.output_keys:
            least_common_resolutions.append(
                self.graph.nodes[
                    self.output_to_node_id[output_key]
                ]._least_common_resolution
            )
        self.min_output_shapes = {
            key: np.lcm.reduce(least_common_resolutions) for key in self.output_keys
        }
        self.step_valid_shapes = self.min_output_shapes  # TODO: is this correct?

        # then find minimal input shapes
        array_shapes = {}
        for node in self.ordered_nodes[::-1]:
            for input_key in node.input_keys:
                array_shapes[input_key] = node.get_input_from_output(
                    array_shapes[node.output_keys[0]]
                )
        self.min_input_shape = {}
        for input_key in self.input_keys:
            self.min_input_shapes[input_key] = array_shapes[input_key]

    def is_valid_input_shape(self, input_key, input_shape):
        return (input_shape >= self.min_input_shape[input_key]).all() and (
            (input_shape - self.min_input_shape[input_key])
            % self.step_valid_shapes[input_key]
            == 0
        ).all()

    def check_input_shapes(self, inputs: dict):
        # check if inputs are valid
        shapes_valid = True
        for input_key, val in inputs.items():
            shapes_valid &= self.is_valid_input_shape(input_key, val.shape)
        return shapes_valid

    def forward(self, **inputs):
        # function for forwarding data through the network
        # inputs is a dictionary of tensors
        # outputs is a dictionary of tensors

        # # check if inputs are valid
        # if not self.check_input_shapes(inputs):
        #     msg = f"{inputs} is not a valid input shape."
        #     raise ValueError(msg)

        # initialize buffer
        self.buffer = inputs

        # march along nodes based on graph succession
        for flushable_list, node in zip(self.flushable_arrays, self.ordered_nodes):
            # TODO: determine how to make inputs optional
            try:
                self.buffer.update(node.forward(self.buffer[node.input_keys]))
            except KeyError:
                logger.warning(f"Node {node} is missing inputs.")

            # clear unnecessary arrays from buffer
            if not self.retain_buffer:
                for key in flushable_list:
                    del self.buffer[key]

        # # collect outputs
        # outputs = {}
        # for output_key in self.output_keys:
        #     outputs.update(self.buffer[output_key])

        # return outputs
        return self.buffer

    def draw(self):
        nx.draw(self.graph)
        plt.show()
