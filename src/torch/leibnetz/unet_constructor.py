from .leibnet import LeibNet
from .nodes import Node, ConvPass, InputNode
from .edges import Edge
import numpy as np


def build_unet(
    top_resolution=(8, 8, 8),
    downsample_factors=[(2, 2, 2), (2, 2, 2)],
    kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
    input_nc=1,
    output_nc=1,
    base_nc=12,
    nc_increase_factor=2,
):
    # define input node
    input_node = InputNode(resolution=top_resolution, identifier="input")

    # define downsample nodes
    downsample_nodes = []
    for i, downsample_factor in enumerate([(1, 1, 1)] + downsample_factors):
        downsample_nodes.append(
            Node(
                ConvPass(
                    base_nc * nc_increase_factor ** (i - 1) if i > 0 else input_nc,
                    base_nc * nc_increase_factor**i,
                    kernel_sizes,
                ),
                resolution=tuple(
                    np.array(top_resolution) * np.array(downsample_factor)
                ),
                identifier=f"downsample_node_{i}",
            )
        )

    # define upsample nodes
    upsample_nodes = []
    for i, downsample_factor in enumerate([(1, 1, 1)] + downsample_factors[:-1]):
        upsample_nodes.append(
            Node(
                ConvPass(
                    base_nc * nc_increase_factor ** (i + 1),
                    base_nc * nc_increase_factor**i,
                    kernel_sizes,
                ),
                resolution=tuple(
                    np.array(top_resolution) * np.array(downsample_factor)
                ),
                identifier=f"upsample_node_{i}",
            )
        )

    # define output node
    output_node = Node(
        ConvPass(
            base_nc,
            output_nc,
            kernel_sizes,
        ),
        resolution=top_resolution,
        identifier="output_node",
    )

    # define input edge
    input_edge = Edge(input_node, downsample_nodes[0], identifier="input_edge")

    # define downsample edges
    downsample_edges = []
    for i, downsample_node in enumerate(downsample_nodes[:-1]):
        downsample_edges += [
            Edge(
                downsample_node,
                downsample_nodes[i + 1],
                identifier=f"downsample_edge_{i}",
            )
        ]

    # define upsample edges
    upsample_edges = []
    for i, upsample_node in enumerate(upsample_nodes + [downsample_nodes[-1]]):
        upsample_edges += [
            Edge(
                upsample_node,
                upsample_nodes[i - 1] if i > 0 else output_node,
                identifier=f"upsample_edge_{i}",
            )
        ]

    # define skip connections
    skip_edges = []
    for i, (downsample_node, upsample_node) in enumerate(
        zip(downsample_nodes[:-1], upsample_nodes)
    ):
        skip_edges.append(
            Edge(
                downsample_node,
                upsample_node,
                identifier=f"skip_edge_{i}",
            )
        )

    # define network
    network = LeibNet(
        nodes=[input_node, output_node, *downsample_nodes, *upsample_nodes],
        edges=[
            input_edge,
            *downsample_edges,
            *upsample_edges,
            *skip_edges,
        ],
    )

    return network
