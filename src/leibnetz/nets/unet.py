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
    convs_per_level=2,
    num_final_convs=1,
    norm_layer=None,
    residual=False,
    dropout_prob=None,
    activation="ReLU",
    final_activation="Sigmoid",
):
    """Build a U-Net architecture using LeibNetz nodes.

    Creates a U-Net encoder-decoder architecture with skip connections.
    The U-Net is built using ConvPassNode and ResampleNode components.

    Args:
        top_resolution: Target resolution at the lowest level (default: (8, 8, 8)).
        downsample_factors: Downsampling factors for each level.
        kernel_sizes: Convolution kernel sizes.
        input_nc: Number of input channels (default: 1).
        output_nc: Number of output channels (default: 1).
        base_nc: Base number of channels (default: 12).
        nc_increase_factor: Factor to increase channels per level (default: 2).
        convs_per_level: Number of convolution layers per level (default: 2).
        num_final_convs: Number of final convolution layers (default: 1).
        norm_layer: Normalization layer to use.
        residual: Whether to use residual connections (default: False).
        dropout_prob: Dropout probability.
        activation: Activation function (default: "ReLU").
        final_activation: Final activation function (default: "Sigmoid").

    Returns:
        LeibNet: The constructed U-Net model.
    """
    ndims = len(top_resolution)
    if downsample_factors is None:
        downsample_factors = [(2,) * ndims] * 2
    if kernel_sizes is None:
        kernel_sizes = [(3,) * ndims] * convs_per_level
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
            [(1,) * len(top_resolution)] * num_final_convs,
            identifier="output",
            norm_layer=norm_layer,
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
