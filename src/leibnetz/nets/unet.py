# %%
from leibnetz import LeibNet
from leibnetz.nodes import ResampleNode, ConvPassNode
import numpy as np


def build_unet(
    top_resolution=(8, 8, 8),
    downsample_factors=None,
    kernel_sizes=None,
    input_nc=1,
    output_nc=1,
    base_nc=12,
    nc_increase_factor=2,
    norm_layer=None,
    residual=False,
    dropout_prob=None,
    activation="ReLU",
    final_activation="Sigmoid",
):
    ndims = len(top_resolution)
    if downsample_factors is None:
        downsample_factors = [(2,) * ndims] * 2
    if kernel_sizes is None:
        kernel_sizes = [(3,) * ndims] * 3
    # define downsample nodes
    downsample_factors = np.array(downsample_factors)
    input_key = "input"
    nodes = []
    c = 0
    for i, downsample_factor in enumerate(downsample_factors):
        output_key = f"in_conv_{c}"
        nodes.append(
            ConvPassNode(
                [input_key],
                [output_key],
                base_nc * nc_increase_factor ** (i - 1) if i > 0 else input_nc,
                base_nc * nc_increase_factor**i,
                kernel_sizes,
                identifier=output_key,
                norm_layer=norm_layer,
                residual=residual,
                dropout_prob=dropout_prob,
                activation=activation,
            ),
        )
        c += 1
        input_key = output_key
        output_key = f"downsample_{i}"
        nodes.append(
            ResampleNode(
                [input_key],
                [output_key],
                1 / downsample_factor,
                identifier=output_key,
            ),
        )
        input_key = output_key

    # define bottleneck node
    output_key = "bottleneck"
    nodes.append(
        ConvPassNode(
            [input_key],
            [output_key],
            base_nc * nc_increase_factor ** (i),
            base_nc * nc_increase_factor ** (i + 1),
            kernel_sizes,
            identifier=output_key,
            norm_layer=norm_layer,
            residual=residual,
            dropout_prob=dropout_prob,
            activation=activation,
        )
    )
    input_key = output_key

    # define upsample nodes
    for i, downsample_factor in reversed(list(enumerate(downsample_factors))):
        output_key = f"upsample_{i}"
        nodes.append(
            ResampleNode(
                [input_key],
                [output_key],
                downsample_factor,
                identifier=output_key,
            )
        )
        input_key = output_key
        c -= 1
        output_key = f"out_conv_{c}"
        nodes.append(
            ConvPassNode(
                [input_key, f"in_conv_{c}"],
                [output_key],
                base_nc * nc_increase_factor**i
                + base_nc * nc_increase_factor ** (i + 1),
                base_nc * nc_increase_factor**i,
                kernel_sizes,
                identifier=output_key,
                norm_layer=norm_layer,
                residual=residual,
                dropout_prob=dropout_prob,
                activation=activation,
            )
        )
        input_key = output_key

    # define output node
    nodes.append(
        ConvPassNode(
            [input_key],
            ["output"],
            base_nc,
            output_nc,
            # kernel_sizes,
            [(1,) * len(top_resolution)],
            identifier="output",
            norm_layer=norm_layer,  # TODO: remove?
            residual=residual,
            dropout_prob=dropout_prob,
            activation=activation,
            final_activation=final_activation,
        )
    )

    # define network
    network = LeibNet(
        nodes,
        outputs={"output": [tuple(np.ones(len(top_resolution))), top_resolution]},
        name="UNet",
    )

    return network


# %%
