from typing import Sequence, Tuple
import networkx as nx
from torch.nn import Module
import matplotlib.pyplot as plt
import numpy as np
from architectures.torch.leibnetz.nodes import Node

import logging

logger = logging.getLogger(__name__)


class LeibNet(Module):
    def __init__(self, nodes, outputs: dict[str, Sequence[Tuple]], retain_buffer=False):
        super().__init__()
        self.nodes = nodes
        self.graph = nx.DiGraph()
        self.assemble(outputs)
        self.retain_buffer = retain_buffer

    def assemble(self, outputs: dict[str, Sequence[Tuple]]):
        """
        Assembles the graph from the nodes,
        sets internal variables,
        and determines the minimal input/output shapes

        Parameters
        ----------
        outputs : dict[str, Sequence[Tuple]]
            Dictionary of output keys and their corresponding shapes and scales

        Returns
        -------
        input_shapes : dict[str, Sequence[Tuple]]
            Dictionary of input keys and their corresponding shapes and scales
        output_shapes : dict[str, Sequence[Tuple]]
            Dictionary of output keys and their corresponding shapes and scales
        """

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

        # set scales for each node (walking backwards from outputs)
        scale_buffer = {key: val[1] for key, val in outputs.items()}
        node_scales_todo = self.nodes.copy()
        self.recurse_scales(self.output_nodes, node_scales_todo, scale_buffer)

        # walk along input paths for each node to determine least common scale
        for node in self.ordered_nodes:
            if node not in self.input_nodes:
                scales = []
                ancestors = nx.ancestors(self.graph, node)
                for ancestor in ancestors:
                    try:
                        scales.append(ancestor.least_common_scale)
                        for elder in nx.ancestors(self.graph, ancestor):
                            ancestors.remove(elder)
                    except RuntimeError:
                        scales.append(ancestor.scale)
                node.set_least_common_scale(np.lcm.reduce(scales))

        # Determine output shapes closest to requested output shapes,
        # and determine corresponding input shapes
        # self.compute_shapes()
        _, _ = self.compute_shapes(outputs)
        _, _ = self.compute_minimal_shapes()

    def recurse_scales(self, nodes, node_scales_todo, scale_buffer):
        if len(node_scales_todo) == 0 or len(nodes) == 0:
            return
        for node in self.nodes:
            key = False
            for key in node.output_keys:
                if key in scale_buffer:
                    break
                else:
                    key = False
            assert key, f"Output {key} not in scale buffer. Please specify all outputs."
            scale = scale_buffer[key]
            if hasattr(node, "set_scale"):
                node.set_scale(scale)
                scale_buffer.update({key: scale for key in node.input_keys})
            else:
                if node._type == "skip":
                    scale_buffer.update({key: scale for key in node.input_keys})
                elif "downsample" in node._type:
                    scale_buffer.update(
                        {key: scale / node.scale_factor for key in node.input_keys}
                    )
                elif "upsample" in node._type:
                    scale_buffer.update(
                        {key: scale * node.scale_factor for key in node.input_keys}
                    )
                else:
                    raise NotImplementedError(f"Scale not set for {node}.")

            node_scales_todo.remove(node)
            next_nodes = list(self.graph.predecessors(node))
            self.recurse_scales(next_nodes, node_scales_todo, scale_buffer)

    def compute_shapes(self, outputs: dict[str, Sequence[Tuple]]):
        # walk backwards through graph to determine input shapes closest to requested output shapes
        shape_buffer = outputs.copy()
        for node in self.ordered_nodes[::-1]:
            shape_buffer.update(
                node.get_input_from_output(shape_buffer[node.output_keys])
            )

        # save input shapes
        self.input_shapes = {key: shape_buffer[key] for key in self.input_keys}

        # walk forwards through graph to determine output shapes based on input shapes
        shape_buffer = self.input_shapes.copy()
        for node in self.ordered_nodes:
            shape_buffer.update(
                node.get_output_from_input(shape_buffer[node.input_keys])
            )

        # save output shapes
        self.output_shapes = {key: shape_buffer[key] for key in self.output_keys}

        # Print input/output shapes
        logger.info("Input shapes:")
        for key in self.input_keys:
            logger.info(f"{key}: {self.input_shapes[key]}")
        logger.info("Output shapes:")
        for key in self.output_keys:
            logger.info(f"{key}: {self.output_shapes[key]}")

        return self.input_shapes, self.output_shapes

    def compute_minimal_shapes(self):
        # analyze graph to determine minimal input/output shapes
        # first find minimal output shapes (1x1x1 at lowest scale)
        # NOTE: expects the output_keys to come from nodes that have realworld unit scales (i.e. not classifications)
        outputs = {}
        for key in self.output_keys:
            node = self.output_to_node_id[key]
            outputs[key] = (np.ones(node.ndims), node.scale)

        return self.compute_shapes(outputs)

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
        nodelist = list(self.graph.nodes)
        labels = {node: node.id for node in nodelist}
        colors = [node.color for node in nodelist]
        nx.draw_networkx(
            self.graph,
            with_labels=True,
            labels=labels,
            node_color=colors,
            font_color="white",
        )
        plt.show()
