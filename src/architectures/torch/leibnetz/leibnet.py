from typing import Optional, Sequence, Tuple, Union
import networkx as nx
from torch import device
import torch
from torch.nn import Module
import matplotlib.pyplot as plt
import numpy as np
from architectures.torch.leibnetz.nodes import Node

import logging

logger = logging.getLogger(__name__)


class LeibNet(Module):
    def __init__(self, nodes, outputs: dict[str, Sequence[Tuple]], retain_buffer=True):
        super().__init__()
        self.nodes = nodes
        self.nodes_dict = {node.id: node for node in nodes}
        self.graph = nx.DiGraph()
        self.assemble(outputs)
        self.retain_buffer = retain_buffer
        self.retain_buffer = True
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

    def assemble(self, outputs: dict[str, Sequence[Tuple]]):
        """
        NOTE: If your scales are non-integer realworld units, you need to treat the scale as integer factors instead.
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
            self.graph.nodes[node]["node_label"] = node.id
            self.graph.nodes[node]["color"] = node.color
            self.graph.nodes[node]["type"] = node._type
            for input_key in node.input_keys:
                if input_key in output_to_node_id:
                    self.graph.add_edge(
                        self.nodes_dict[output_to_node_id[input_key]], node
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
        self.array_keys = []
        flushed_arrays = []
        for i, node in enumerate(self.ordered_nodes):
            for input_key in node.input_keys:
                if input_key not in self.array_keys:
                    self.array_keys.append(input_key)
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
            self.graph.nodes[node]["scale"] = node.scale
            scales = [node.scale]
            if node not in self.input_nodes:
                ancestors = list(nx.ancestors(self.graph, node).copy())
                for ancestor in ancestors:
                    try:
                        scales.append(ancestor.least_common_scale)
                        for elder in nx.ancestors(self.graph, ancestor):
                            if elder in ancestors:
                                ancestors.remove(elder)
                    except RuntimeError:
                        scales.append(ancestor.scale)
            scales = np.ceil(scales).astype(int)
            scales = scales[scales.sum(axis=1) > 0]  # remove NaN scales
            node.set_least_common_scale(np.lcm.reduce(scales))

        # Determine output shapes closest to requested output shapes,
        # and determine corresponding input shapes
        self.input_shapes, self.output_shapes = self.compute_shapes(outputs)
        self.min_input_shapes, self.min_output_shapes = self.compute_minimal_shapes()

    def recurse_scales(self, nodes, node_scales_todo, scale_buffer):
        if len(node_scales_todo) == 0 or len(nodes) == 0:
            return
        for node in nodes:
            if node not in node_scales_todo:
                continue
            key = False
            for key in node.output_keys:
                if key in scale_buffer:
                    break
                else:
                    key = False
            assert (
                key
            ), f'Output "{key}" not in scale buffer. Please specify all outputs.'
            scale = np.array(scale_buffer[key]).astype(int)
            if hasattr(node, "set_scale"):
                node.set_scale(scale)
            if "downsample" in node._type or "upsample" in node._type:
                scale_buffer.update(
                    {
                        key: (scale * node.scale_factor).astype(int)
                        for key in node.input_keys
                    }
                )
            else:
                scale_buffer.update({key: scale for key in node.input_keys})

            node_scales_todo.remove(node)
            next_nodes = list(self.graph.predecessors(node))
            if len(next_nodes) == 0:
                continue
            node_scales_todo, scale_buffer = self.recurse_scales(
                next_nodes, node_scales_todo, scale_buffer
            )

        return node_scales_todo, scale_buffer

    def compute_shapes(self, outputs: dict[str, Sequence[Tuple]], set=True):
        # walk backwards through graph to determine input shapes closest to requested output shapes
        shape_buffer = outputs.copy()
        for node in self.ordered_nodes[::-1]:
            shape_buffer.update(
                node.get_input_from_output(
                    {k: shape_buffer.get(k, None) for k in node.output_keys}
                )
            )

        # save input shapes
        input_shapes = {k: shape_buffer.get(k, None) for k in self.input_keys}

        # walk forwards through graph to determine output shapes based on input shapes
        shape_buffer = input_shapes.copy()
        for node in self.ordered_nodes:
            shape_buffer.update(
                node.get_output_from_input(
                    {k: shape_buffer.get(k, None) for k in node.input_keys}
                )
            )

        # set dimensions
        self.ndims = len(shape_buffer[self.output_keys[0]][0])

        # save output shapes
        output_shapes = {k: shape_buffer.get(k, None) for k in self.output_keys}
        input_shapes = {k: shape_buffer.get(k, None) for k in self.input_keys}

        # Print input/output shapes
        logger.info("Input shapes:")
        for key in self.input_keys:
            logger.info(f"{key}: {input_shapes[key]}")
        logger.info("Output shapes:")
        for key in self.output_keys:
            logger.info(f"{key}: {output_shapes[key]}")

        if set:
            self.input_shapes = input_shapes
            self.output_shapes = output_shapes
            self.array_shapes = shape_buffer

        return input_shapes, output_shapes

    def compute_minimal_shapes(self):
        # analyze graph to determine minimal input/output shapes
        # first find minimal output shapes (1x1x1 at lowest scale)
        # NOTE: expects the output_keys to come from nodes that have realworld unit scales (i.e. not classifications)
        outputs = {}
        for key in self.output_keys:
            node = self.nodes_dict[self.output_to_node_id[key]]
            outputs[key] = (np.ones(node.ndims, dtype=int), node.scale)

        min_input_shapes, min_output_shapes = self.compute_shapes(outputs, set=False)
        return min_input_shapes, min_output_shapes

    def is_valid_input_shape(self, input_key, input_shape):
        raise NotImplementedError("This has not been fully implemented yet.")
        return (input_shape >= self.min_input_shapes[input_key]).all() and (
            (input_shape - self.min_input_shapes[input_key])
            % self.step_valid_shapes[input_key]
            == 0
        ).all()

    def check_input_shapes(self, inputs: dict):
        # check if inputs are valid
        shapes_valid = True
        for input_key, val in inputs.items():
            shapes_valid &= self.is_valid_input_shape(input_key, val.shape)
        return shapes_valid

    def get_example_inputs(self, device: device = None):
        # function for generating example inputs
        if device is None:
            device = self.device
        inputs = {}
        for k, v in self.input_shapes.items():
            inputs[k] = torch.rand(
                (
                    1,
                    1,
                )
                + tuple(v[0].astype(int))
            ).to(device)
        return inputs

    # TODO: Add specification for sending arrays to different devices during forward pass

    def train(self, mode: bool = True):
        # function for setting network to train mode
        for node in self.nodes:
            node.train(mode)

    def eval(self):
        # function for setting network to eval mode
        for node in self.nodes:
            node.eval()

    def to(self, device: device):
        # function for moving network to device
        self.device = device
        for node in self.nodes:
            node.to(device)

    def cuda(self, device: int | device | None = None):
        # function for moving network to cuda
        if device is None:
            device = torch.cuda.current_device()
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.device = device
        self.to(device)

    def cpu(self):
        # function for moving network to cpu
        self.device = torch.device("cpu")
        self.to(self.device)

    def modules(self):
        # function for returning all modules in network
        return [modules for node in self.nodes for modules in node.modules()]

    def parameters(self):
        # function for returning all parameters in network
        return [param for node in self.nodes for param in node.parameters()]

    def forward(self, inputs):
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
                self.buffer.update(
                    node.forward({k: self.buffer.get(k, None) for k in node.input_keys})
                )
            except KeyError:
                logger.warning(f"Node ID {node.id} is missing inputs.")
            except Exception as e:
                logger.error(f"Node ID {node.id} failed with error: {e}")
                raise e

            # clear unnecessary arrays from buffer
            if not self.retain_buffer:
                raise NotImplementedError(
                    "This has not been successfully implemented yet."
                )
                for key in flushable_list:
                    try:
                        del self.buffer[key]
                    except KeyError:
                        pass

        # collect outputs
        return {key: self.buffer[key] for key in self.output_keys}
        # return self.buffer

    def to_mermaid(self, separate_arrays: bool = False, vertical: bool = True):
        # function for converting network to mermaid graph
        # NOTE: mermaid graphs can be rendered at https://mermaid-js.github.io/mermaid-live-editor/
        def seps(_type):
            if "downsample" in _type:
                in_sep = "-.-"
                out_sep = "-.->"
            elif "upsample" in _type:
                in_sep = "==="
                out_sep = "==>"
            else:
                in_sep = "---"
                out_sep = "-->"
            return in_sep, out_sep

        def shapes(_type):
            if "upsample" in _type:
                in_shape = "[/"
                out_shape = "\\]"
            elif "downsample" in _type:
                in_shape = "[\\"
                out_shape = "/]"
            elif "attention" in _type:
                in_shape = "{{"
                out_shape = "}}"
            else:
                in_shape = "(["
                out_shape = "])"
            return in_shape, out_shape

        if vertical:
            outstring = "graph TD\n"
        else:
            outstring = "graph LR\n"
        for node in self.nodes:
            s, e = shapes(node._type)
            outstring += f"\tnode-{node.id}{s}{node.id}{e}\n"
        if separate_arrays:
            for key in self.array_keys:
                outstring += f"\t{key}[{key}]\n"
            for node in self.nodes:
                in_sep, out_sep = seps(node._type)
                for input_key in node.input_keys:
                    scales = [f"{s}nm" for s in self.array_shapes[input_key][1]]
                    scale_str = "x".join(scales)
                    outstring += f"\tsubgraph {scale_str}\n"
                    outstring += f"\t\t{input_key}{in_sep}node-{node.id}\n"
                    outstring += "\tend\n"
                for output_key in node.output_keys:
                    scales = [f"{s}nm" for s in self.array_shapes[output_key][1]]
                    scale_str = "x".join(scales)
                    outstring += f"\tsubgraph {scale_str}\n"
                    outstring += f"\t\tnode-{node.id}{out_sep}{output_key}\n"
                    outstring += "\tend\n"
        else:
            for node in self.nodes:
                for input_key in node.input_keys:
                    try:
                        in_name = f"node-{self.output_to_node_id[input_key]}"
                        in_sep, out_sep = seps(
                            self.nodes_dict[self.output_to_node_id[input_key]]._type
                        )
                        # outstring += f"\tsubgraph {self.nodes_dict[in_name].scale}\n"
                    except KeyError:
                        in_name = input_key
                        out_sep = "-->"
                    scales = [f"{s}nm" for s in self.array_shapes[input_key][1]]
                    scale_str = "x".join(scales)
                    outstring += f"\tsubgraph {scale_str}\n"
                    outstring += f"\t\t{in_name}{out_sep}node-{node.id}\n"
                    outstring += "\tend\n"

        print(outstring)
        return outstring

    def draw(self, type: str = "spiral", node_size: int = 1000, font_size: int = 8):
        DeprecationWarning(
            "This function is deprecated. See LeibNet.to_mermaid()."
        )  # noqa
        labels = {node: "\n".join(node.id.split("_")) for node in self.ordered_nodes}
        colors = [node.color for node in self.ordered_nodes]
        if type == "multipartite":
            pos = nx.multipartite_layout(
                self.graph, subset_key="scale", align="horizontal"
            )
        elif type == "spiral":
            pos = nx.spiral_layout(self.graph, equidistant=True)
        elif type == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        elif type == "planar":
            pos = nx.planar_layout(self.graph)
        else:
            pos = nx.planar_layout(self.graph)
            pos = nx.kamada_kawai_layout(self.graph, pos=pos)

        nx.draw_networkx(
            self.graph,
            pos=pos,
            # with_labels=False,
            labels=labels,
            node_color=colors,
            # font_color="white",
            font_size=font_size,
            node_size=node_size,
            # node_shape="s"
        )
