import logging
from typing import Iterable, Optional, Sequence, Tuple


import networkx as nx
import numpy as np
import numpy as np
import torch
from torch.nn import Module


from leibnetz.nodes import Node

logger = logging.getLogger(__name__)


class LeibNet(Node):
    """
    Main LeibNet class for composing neural network architectures from nodes.

    LeibNet allows composing complex neural networks by connecting multiple
    Node objects in a directed acyclic graph. It handles automatic shape
    propagation, device management, and provides utilities for model export and
    visualization.

    Args:
        nodes: Iterable of Node or Module objects to compose into the network.
        outputs: Dictionary mapping output keys to their shape specifications.
        initialization: Weight initialization method (default: "kaiming").
        name: Name identifier for the network (default: "LeibNet").
    """

    def _determine_network_interface(self, nodes, outputs):
        """Determine input and output keys from node structure."""
        temp_nodes = []
        for node in nodes:
            if isinstance(node, (Node, Module)):
                temp_nodes.append(node)
            else:
                msg = f"{node} is not a Node or LeibNet."
                raise ValueError(msg)

        all_input_keys = []
        all_output_keys = []
        internal_outputs = set()

        for node in temp_nodes:
            if hasattr(node, "input_keys"):
                all_input_keys.extend(node.input_keys)
            if hasattr(node, "output_keys"):
                all_output_keys.extend(node.output_keys)
                internal_outputs.update(node.output_keys)

        input_keys = [key for key in all_input_keys if key not in internal_outputs]
        output_keys = list(outputs.keys())

        return input_keys, output_keys

    def _process_nodes(self, nodes):
        """Process and validate nodes list."""
        full_node_list = []
        for node in nodes:
            if isinstance(node, (Node, LeibNet)):
                full_node_list.append(node)
            else:
                msg = f"{node} is not a Node or LeibNet."
                raise ValueError(msg)
        return full_node_list

    def _apply_weight_initialization(self, initialization):
        """Apply weight initialization to the network."""
        if initialization == "kaiming":
            self.apply(
                lambda m: (
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d))
                    else None
                )
            )
        elif initialization == "xavier":
            self.apply(
                lambda m: (
                    torch.nn.init.xavier_normal_(m.weight)
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d))
                    else None
                )
            )
        elif initialization == "orthogonal":
            self.apply(
                lambda m: (
                    torch.nn.init.orthogonal_(m.weight)
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d))
                    else None
                )
            )
        elif initialization is not None:
            raise ValueError(f"Unknown initialization {initialization}")

    def __init__(
        self,
        nodes: Iterable,
        outputs: dict[str, Sequence[Tuple]],
        initialization="kaiming",
        name="LeibNet",
    ):
        # Determine network interface
        input_keys, output_keys = self._determine_network_interface(nodes, outputs)

        # Initialize parent Node class
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, identifier=name
        )

        # Set LeibNet-specific attributes
        self.color = "#0000FF"
        self._type = "leibnet"

        # Process and store nodes
        self.nodes = self._process_nodes(nodes)
        self.nodes_dict = torch.nn.ModuleDict({node.id: node for node in self.nodes})
        self.graph = nx.DiGraph()

        # Assemble the network
        self.assemble(outputs)

        # Apply weight initialization
        self.initialization = initialization
        self._apply_weight_initialization(initialization)

        self.name = name

    def _validate_nodes_and_build_mapping(self):
        """Validate nodes and build output-to-node mapping."""
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

        return output_to_node_id

    def _build_graph_structure(self, output_to_node_id):
        """Build the graph structure from nodes and edges."""
        self.graph.add_nodes_from(self.nodes)

        for node in self.nodes:
            self.graph.nodes[node]["node_label"] = node.id
            self.graph.nodes[node]["color"] = node.color
            self.graph.nodes[node]["type"] = node._type
            for input_key in node.input_keys:
                if input_key in output_to_node_id:
                    self.graph.add_edge(
                        self.nodes_dict[output_to_node_id[input_key]], node
                    )

        if not nx.is_directed_acyclic_graph(self.graph):
            msg = "Graph is not acyclic."
            raise ValueError(msg)

    def _identify_boundary_nodes(self):
        """Identify input and output nodes in the graph."""
        self.input_nodes = [
            node for node in self.nodes if self.graph.in_degree(node) == 0
        ]
        self.output_nodes = [
            node for node in self.nodes if self.graph.out_degree(node) == 0
        ]

        self._internal_input_nodes = self.input_nodes
        self._internal_output_nodes = self.output_nodes

        self._internal_input_keys = []
        for node in self.input_nodes:
            self._internal_input_keys += node.input_keys

        self._internal_output_keys = []
        for node in self.output_nodes:
            self._internal_output_keys += node.output_keys

    def _determine_execution_order(self):
        """Determine the execution order and flushable arrays."""
        self.ordered_nodes = list(nx.topological_sort(self.graph.reverse()))

        self.flushable_arrays = [[] for _ in range(len(self.ordered_nodes))]
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

        self.ordered_nodes = self.ordered_nodes[::-1]
        self.flushable_arrays = self.flushable_arrays[::-1]

    def _compute_node_scales(self, outputs):
        """Compute scales for all nodes."""
        scale_buffer = {key: val[1] for key, val in outputs.items()}
        node_scales_todo = self.nodes.copy()
        self.recurse_scales(self.output_nodes, node_scales_todo, scale_buffer)

        all_scales = []
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
            scales = scales[scales.sum(axis=1) > 0]
            node.set_least_common_scale(np.lcm.reduce(scales))
            all_scales.extend(scales)
        self.set_least_common_scale(np.lcm.reduce(all_scales))

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
        # Validate nodes and build mapping
        self.output_to_node_id = self._validate_nodes_and_build_mapping()

        # Build graph structure
        self._build_graph_structure(self.output_to_node_id)

        # Identify boundary nodes
        self._identify_boundary_nodes()

        # Determine execution order
        self._determine_execution_order()

        # Compute scales
        self._compute_node_scales(outputs)

        # Determine output and input shapes
        self._input_shapes, self._output_shapes = self.compute_shapes(outputs)
        self.min_input_shapes, self.min_output_shapes = self.compute_minimal_shapes()

    def recurse_scales(self, nodes, node_scales_todo, scale_buffer):
        for node in nodes:
            if node not in node_scales_todo:
                continue
            key = False
            for key in node.output_keys:
                if key in scale_buffer:
                    break
                else:
                    key = False
            assert key, (
                f'No output keys from node "{node.id}" in scale buffer. '
                "Please specify all outputs."
            )
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
            if len(node_scales_todo) == 0 or len(next_nodes) == 0:
                continue
            node_scales_todo, scale_buffer = self.recurse_scales(
                next_nodes, node_scales_todo, scale_buffer
            )

        return node_scales_todo, scale_buffer

    def compute_shapes(self, outputs: dict[str, Sequence[Tuple]], set=True):
        """Compute input shapes required for given output shapes.

        Walks backwards through the network graph to determine input shapes
        needed to produce the requested output shapes.

        Args:
            outputs: Dictionary of desired output shapes.
            set: Whether to store computed shapes in the instance (default: True).

        Returns:
            tuple: (input_shapes, array_shapes) dictionaries.
        """
        # walk backwards through graph to determine input shapes
        # closest to requested output shapes
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
        ndims_value = len(shape_buffer[list(self.output_keys)[0]][0])
        self._ndims = ndims_value

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
            self._input_shapes = input_shapes
            self._output_shapes = output_shapes
            self._array_shapes = shape_buffer

        return input_shapes, output_shapes

    def compute_minimal_shapes(self):
        # analyze graph to determine minimal input/output shapes
        # first find minimal output shapes (1x1x1 at lowest scale)
        # NOTE: expects the output_keys to come from nodes that have
        # realworld unit scales (i.e. not classifications)
        outputs = {}
        for key in self.output_keys:
            node = self.nodes_dict[self.output_to_node_id[key]]
            outputs[key] = (np.ones(node.ndims, dtype=int), node.scale)

        min_input_shapes, min_output_shapes = self.compute_shapes(outputs, set=False)
        return min_input_shapes, min_output_shapes

    def is_valid_input_shape(self, input_key, input_shape):
        # raise NotImplementedError("This has not been fully implemented yet.")
        return (input_shape >= self.min_input_shapes[input_key]).all() and (
            (input_shape - self.min_input_shapes[input_key])
            % self.step_valid_shapes(input_key)
            == 0
        ).all()

    def step_up_size(self, steps: int = 1, step_size: int = 1):
        for n in range(steps):
            target_arrays = {}
            for name, metadata in self.output_shapes.items():
                target_arrays[name] = tuple(
                    (
                        tuple(s + step_size for s in metadata["shape"]),
                        metadata["scale"],
                    )
                )
        self.compute_shapes(target_arrays, set=True)

    def step_valid_shapes(self, input_key):
        input_scale = self._input_shapes[input_key][1]
        step_size = self.least_common_scale / input_scale
        return step_size.astype(int)

    def check_input_shapes(self, inputs: dict):
        # check if inputs are valid
        shapes_valid = True
        for input_key, val in inputs.items():
            shapes_valid &= self.is_valid_input_shape(input_key, val.shape)
        return shapes_valid

    # @torch.jit.export
    def get_example_inputs(self, device: Optional[torch.device] = None):
        # function for generating example inputs
        inputs = {}
        for k, v in self._input_shapes.items():
            inputs[k] = torch.rand((1, 1) + tuple(v[0].astype(int)))
        if device is not None:
            for k in inputs.keys():
                inputs[k] = inputs[k].to(device)
        return inputs

    # TODO: Add specification for sending arrays to different devices during forward pass
    @property
    def devices(self):
        """Get list of devices used by model parameters.

        Returns:
            list: List of devices used by model parameters.
        """
        devices = []
        for parameters in self.parameters():
            devices.append(parameters.device)
        return devices

    def _get_shapes(self, shape_dict: dict):
        return {
            k: {
                "shape": tuple([int(s) for s in s[0]]),
                "scale": tuple([float(s) for s in s[1]]),
            }
            for k, s in shape_dict.items()
        }

    @property
    def input_shapes(self):
        """Get input shapes dictionary with shape and scale information."""
        return self._get_shapes(self._input_shapes)

    @property
    def output_shapes(self):
        """Get output shapes dictionary with shape and scale information."""
        return self._get_shapes(self._output_shapes)

    @property
    def array_shapes(self):
        """Get array shapes dictionary with shape and scale information."""
        return self._get_shapes(self._array_shapes)

    @property
    def param_num(self):
        """Get total number of parameters in the model."""
        param_num = 0
        for key, val in self.named_parameters():
            # print(f"{key}: {val.shape}")
            param_num += val.numel()
        return param_num

    def mps(self):
        """Move model to Apple Silicon MPS device if available."""
        if torch.backends.mps.is_available():
            self.to("mps")
        else:
            logger.error('Unable to move model to Apple Silicon ("mps")')

    def forward(self, inputs: dict[str, dict[str, Sequence[int | float]]]):
        """Forward pass through the network.

        Args:
            inputs: Dictionary of input tensors, where keys match input_keys.

        Returns:
            dict: Dictionary of output tensors, where keys match output_keys.
        """
        # initialize buffer
        if isinstance(inputs, dict):
            return_type = "dict"
            buffer = {key: inputs[key] for key in self.input_keys}
        elif isinstance(inputs, torch.Tensor):
            assert (
                len(self.input_keys) == 1
            ), f"Incorrect number of inputs. Expected 1, got {len(inputs)}."
            return_type = "tensor"
            buffer = {list(self.input_keys)[0]: inputs}
        elif not isinstance(inputs, dict):
            assert len(inputs) == len(
                self.input_keys
            ), f"Incorrect number of inputs. Expected {len(self.input_keys)}, got {len(inputs)}."
            return_type = "list"
            buffer = {key: inputs[i] for i, key in enumerate(self.input_keys)}

        # march along nodes based on graph succession
        for flushable_list, node in zip(self.flushable_arrays, self.ordered_nodes):
            # TODO: determine how to make inputs optional
            try:
                buffer.update(
                    node.forward({k: buffer.get(k, None) for k in node.input_keys})
                )
            except KeyError:
                logger.warning(f"Node ID {node.id} is missing inputs.")
            except Exception as e:
                logger.error(f"Node ID {node.id} failed with error: {e}")
                raise e

        # collect outputs
        if return_type == "tensor":
            return buffer[list(self.output_keys)[0]]
        elif return_type == "list":
            return [buffer[key] for key in self.output_keys]
        else:
            return {key: buffer[key] for key in self.output_keys}

    # @torch.jit.export
    @staticmethod
    def _mermaid_separators(_type):
        """Get mermaid separator strings based on node type."""
        if "downsample" in _type:
            return "-.-", "-.->"
        elif "upsample" in _type:
            return "===", "==>"
        else:
            return "---", "-->"

    @staticmethod
    def _mermaid_shapes(_type):
        """Get mermaid shape markers based on node type."""
        if "upsample" in _type:
            return "[/", "\\]"
        elif "downsample" in _type:
            return "[\\", "/]"
        elif "attention" in _type:
            return "{{", "}}"
        else:
            return "([", "])"

    def _mermaid_add_nodes(self, outstring):
        """Add node definitions to mermaid string."""
        for node in self.nodes:
            s, e = self._mermaid_shapes(node._type)
            outstring += f"\tnode-{node.id}{s}{node.id}{e}\n"
        return outstring

    def _mermaid_separate_arrays(self, outstring):
        """Generate mermaid output with separate array nodes."""
        for key in self.array_keys:
            size_str = "x".join([str(int(s)) for s in self._array_shapes[key][0]])
            outstring += f"\t{key}[{key}: {size_str}]\n"

        for node in self.nodes:
            in_sep, out_sep = self._mermaid_separators(node._type)
            for input_key in node.input_keys:
                scales = [f"{s}nm" for s in self._array_shapes[input_key][1]]
                scale_str = "x".join(scales)
                outstring += f"\tsubgraph {scale_str}\n"
                outstring += f"\t\t{input_key}{in_sep}node-{node.id}\n"
                outstring += "\tend\n"
            for output_key in node.output_keys:
                scales = [f"{s}nm" for s in self._array_shapes[output_key][1]]
                scale_str = "x".join(scales)
                outstring += f"\tsubgraph {scale_str}\n"
                outstring += f"\t\tnode-{node.id}{out_sep}{output_key}\n"
                outstring += "\tend\n"
        return outstring

    def _mermaid_connected_nodes(self, outstring):
        """Generate mermaid output with connected nodes."""
        for node in self.nodes:
            for input_key in node.input_keys:
                try:
                    in_name = f"node-{self.output_to_node_id[input_key]}"
                    _, out_sep = self._mermaid_separators(
                        self.nodes_dict[self.output_to_node_id[input_key]]._type
                    )
                except KeyError:
                    in_name = input_key
                    out_sep = "-->"
                scales = [f"{s}nm" for s in self._array_shapes[input_key][1]]
                scale_str = "x".join(scales)
                outstring += f"\tsubgraph {scale_str}\n"
                outstring += f"\t\t{in_name}{out_sep}node-{node.id}\n"
                outstring += "\tend\n"
        return outstring

    def to_mermaid(self, separate_arrays: bool = False, vertical: bool = False):
        """Convert network to mermaid graph format.

        NOTE: mermaid graphs can be rendered at https://mermaid-js.github.io/mermaid-live-editor/
        """
        outstring = "graph TD\n" if vertical else "graph LR\n"
        outstring = self._mermaid_add_nodes(outstring)

        if separate_arrays:
            outstring = self._mermaid_separate_arrays(outstring)
        else:
            outstring = self._mermaid_connected_nodes(outstring)

        print(outstring)
        return outstring

    @staticmethod
    def load(path: str):
        return torch.load(path)

    def __getitem__(self, key):
        if not hasattr(self, "heads"):
            return self.nodes_dict[key]
        else:
            return torch.nn.Sequential(self, self.heads[key])

    def __setitem__(self, key, value):
        self.add_head(value, key)

    def add_head(self, head, key):
        if not hasattr(self, "heads"):
            self.heads = {}
        self.heads[key] = head

    # Node interface methods - required for LeibNet to be used as a node
    def get_input_from_output_shape(self, output_shape):
        """Calculate required input shapes for a given output shape"""
        # For a LeibNet used as a node, we delegate to the internal compute_shapes method
        # We need to construct a dummy outputs dict using our output_keys
        dummy_outputs = {}
        for key in self.output_keys:
            # Use the provided output_shape for all our outputs
            # This is a simplification - in practice each output might have different shapes
            dummy_outputs[key] = (output_shape, np.ones(len(output_shape)))

        input_shapes, _ = self.compute_shapes(dummy_outputs, set=False)

        # Convert to the format expected by Node interface
        result = {}
        for key, (shape, scale) in input_shapes.items():
            result[key] = (shape, scale)
        return result

    def get_output_from_input_shape(self, input_shape):
        """Calculate output shapes for a given input shape"""
        # For a LeibNet used as a node, we need to determine outputs from inputs
        # This is more complex since we need to run through our internal graph

        # Create dummy inputs using our input_keys
        dummy_inputs = {}
        for key in self.input_keys:
            # Use the provided input_shape for all our inputs
            dummy_inputs[key] = (input_shape, np.ones(len(input_shape)))

        # We can use our internal shape computation, but we need the actual outputs
        # For now, return the stored output shapes (this is a simplification)
        result = {}
        for key in self.output_keys:
            if hasattr(self, "_output_shapes") and key in self._output_shapes:
                shape, scale = self._output_shapes[key]
                result[key] = (shape, scale)
            else:
                # Fallback: assume same shape as input (not accurate but safe)
                result[key] = (input_shape, np.ones(len(input_shape)))
        return result
