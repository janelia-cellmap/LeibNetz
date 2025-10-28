# %%
from timeit import Timer
from leibnetz import LeibNet
from leibnetz.nodes import ResampleNode, ConvPassNode
import numpy as np


def build_subnet(
    bottleneck_input_dict=None,
    subnet_id="",
    top_resolution=(8, 8, 8),
    downsample_factors=None,
    kernel_sizes=None,
    input_nc=1,
    output_nc=1,
    base_nc=12,
    nc_increase_factor=2,
    num_final_convs=1,
    norm_layer=None,
    residual=False,
    dropout_prob=None,
    squeeze_excitation=False,
    squeeze_ratio=2,
    activation="ReLU",
    final_activation="Sigmoid",
):
    """Build a subnet for use in multi-scale networks.

    Creates a single subnet that can be used as part of a larger multi-scale architecture.
    Similar to U-Net but designed for integration into multi-scale processing.

    Args:
        bottleneck_input_dict: Optional dictionary for bottleneck input configuration.
        subnet_id: Identifier for this subnet.
        top_resolution: Target resolution at the lowest level (default: (8, 8, 8)).
        downsample_factors: Downsampling factors for each level.
        kernel_sizes: Convolution kernel sizes.
        input_nc: Number of input channels (default: 1).
        output_nc: Number of output channels (default: 1).
        base_nc: Base number of channels (default: 12).
        nc_increase_factor: Factor to increase channels per level (default: 2).
        num_final_convs: Number of final convolution layers (default: 1).
        norm_layer: Normalization layer to use.
        residual: Whether to use residual connections (default: False).
        dropout_prob: Dropout probability.
        squeeze_excitation: Whether to use squeeze-and-excitation blocks (default: False).
        squeeze_ratio: Squeeze ratio for SE blocks (default: 2).
        activation: Activation function (default: "ReLU").
        final_activation: Final activation function (default: "Sigmoid").

    Returns:
        tuple: (nodes_list, outputs_dict, bottleneck_output_dict)
    """
    ndims = len(top_resolution)
    if downsample_factors is None:
        downsample_factors = [(2,) * ndims] * 2
    if kernel_sizes is None:
        kernel_sizes = [(3,) * ndims] * 3
    # define downsample nodes
    downsample_factors = np.array(downsample_factors)
    input_key = f"{subnet_id}_input"
    nodes = []
    c = 0
    for i, downsample_factor in enumerate(downsample_factors):
        output_key = f"{subnet_id}_in_conv_{c}"
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
                squeeze_excitation=squeeze_excitation,
                squeeze_ratio=squeeze_ratio,
                activation=activation,
            ),
        )
        c += 1
        input_key = output_key
        output_key = f"{subnet_id}_downsample_{i}"
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
    output_key = f"{subnet_id}_bottleneck"
    bottleneck_input_fmaps = base_nc * nc_increase_factor ** (i)
    bottleneck_inputs = [input_key]
    if bottleneck_input_dict is not None:
        for key in bottleneck_input_dict.keys():
            bottleneck_input_fmaps += bottleneck_input_dict[key][2]
            bottleneck_inputs.append(key)
    nodes.append(
        ConvPassNode(
            bottleneck_inputs,
            [output_key],
            bottleneck_input_fmaps,
            base_nc * nc_increase_factor ** (i + 1),
            kernel_sizes,
            identifier=output_key,
            norm_layer=norm_layer,
            residual=residual,
            dropout_prob=dropout_prob,
            squeeze_excitation=squeeze_excitation,
            squeeze_ratio=squeeze_ratio,
            activation=activation,
        )
    )
    input_key = output_key

    # define upsample nodes
    for i, downsample_factor in reversed(list(enumerate(downsample_factors))):
        output_key = f"{subnet_id}_upsample_{i}"
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
        output_key = f"{subnet_id}_out_conv_{c}"
        nodes.append(
            ConvPassNode(
                [input_key, f"{subnet_id}_in_conv_{c}"],
                [output_key],
                base_nc * nc_increase_factor**i
                + base_nc * nc_increase_factor ** (i + 1),
                base_nc * nc_increase_factor**i,
                # kernel_sizes,
                [(1,) * len(top_resolution)] * num_final_convs,
                identifier=output_key,
                norm_layer=norm_layer,
                residual=residual,
                dropout_prob=dropout_prob,
                squeeze_excitation=squeeze_excitation,
                squeeze_ratio=squeeze_ratio,
                activation=activation,
            )
        )
        input_key = output_key

    # define output node
    nodes.append(
        ConvPassNode(
            [input_key],
            [f"{subnet_id}_output"],
            base_nc,
            output_nc,
            # kernel_sizes,
            [(1,) * len(top_resolution)],
            identifier=f"{subnet_id}_output",
            norm_layer=norm_layer,
            residual=residual,
            dropout_prob=dropout_prob,
            squeeze_excitation=squeeze_excitation,
            squeeze_ratio=squeeze_ratio,
            activation=activation,
            final_activation=final_activation,
        )
    )
    outputs = {
        input_key: [tuple(np.ones(len(top_resolution))), top_resolution, base_nc],
        f"{subnet_id}_output": [
            tuple(np.ones(len(top_resolution))),
            top_resolution,
            output_nc,
        ],
    }
    return nodes, outputs


# %%
def build_scale_net(
    subnet_dict_list: list[dict] = [
        {"top_resolution": (32, 32, 32)},
        {"top_resolution": (8, 8, 8)},
    ]
):
    """Build a multi-scale network architecture using multiple subnets.

    Creates a ScaleNet that processes inputs at different scales using multiple
    subnet instances. Each subnet handles a different resolution/scale.

    Args:
        subnet_dict_list: List of dictionaries containing subnet configuration.
                         Each dict should contain at least 'top_resolution'.

    Returns:
        LeibNet: The constructed multi-scale network.
    """
    nodes = []
    outputs = {}
    bottleneck_input_dict = None
    for i, subnet_dict in enumerate(subnet_dict_list):
        subnet_id = subnet_dict.get("subnet_id", i)
        subnet_dict["subnet_id"] = subnet_id
        subnet_nodes, subnet_outputs = build_subnet(
            bottleneck_input_dict=bottleneck_input_dict,
            **subnet_dict,
        )
        nodes.extend(subnet_nodes)
        output = subnet_outputs.pop(f"{subnet_id}_output")
        outputs[f"{subnet_id}_output"] = output
        bottleneck_input_dict = subnet_outputs
    network = LeibNet(nodes, outputs=outputs, name="ScaleNet")
    return network


# %%
def testing():
    subnet_dict_list = [
        {"top_resolution": (32, 32, 32)},
        {"top_resolution": (8, 8, 8)},
    ]
    scalenet = build_scale_net(subnet_dict_list)
    param_num = 0
    for key, val in scalenet.named_parameters():
        print(f"{key}: {val.shape}")
        param_num += val.numel()
    scalenet.array_shapes
    # %%
    inputs = scalenet.get_example_inputs()
    outputs = scalenet(inputs)
    # %%
    for key, val in outputs.items():
        print(f"{key}: {val.shape}")

    # %%
    scalenet.to_mermaid()
    # %%
    scalenet.to_mermaid(separate_arrays=True)

    # %%
    timer = Timer(lambda: scalenet(inputs))
    num, time = timer.autorange()
    print(f"Time per run: {time/num} seconds")
    return scalenet


# %%
