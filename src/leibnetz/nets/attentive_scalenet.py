# %%
from timeit import Timer
from leibnetz import LeibNet
from leibnetz.nodes import (
    ResampleNode,
    ConvPassNode,
    AdditiveAttentionGateNode,
)
import numpy as np


def build_subnet(
    bottleneck_input_dict=None,
    subnet_id="",
    top_resolution=(8, 8, 8),
    downsample_factors=[(2, 2, 2), (2, 2, 2)],
    kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
    input_nc=1,
    output_nc=1,
    base_nc=12,
    nc_increase_factor=2,
    norm_layer=None,
    residual=False,
    dropout_prob=None,
):
    # define downsample nodes
    downsample_factors = np.array(downsample_factors)
    input_key = f"{subnet_id}_input"
    nodes = []
    c = 0
    i = 0
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
        output_key = f"{subnet_id}_additiveAttentionGate_{c}"
        nodes.append(
            AdditiveAttentionGateNode(
                output_keys=[output_key],
                gating_key=input_key,
                input_key=f"{subnet_id}_in_conv_{c}",
                output_nc=base_nc * nc_increase_factor**i,
                gating_nc=base_nc * nc_increase_factor ** (i + 1),
                input_nc=base_nc * nc_increase_factor**i,
                identifier=output_key,
            )
        )
        input_key = output_key
        output_key = f"{subnet_id}_out_conv_{c}"
        nodes.append(
            ConvPassNode(
                # [input_key, f"{subnet_id}_in_conv_{c}"],
                [input_key, f"{subnet_id}_upsample_{i}"],
                [output_key],
                base_nc * nc_increase_factor**i
                + base_nc * nc_increase_factor ** (i + 1),
                base_nc * nc_increase_factor**i,
                kernel_sizes,
                identifier=output_key,
                norm_layer=norm_layer,
                residual=residual,
                dropout_prob=dropout_prob,
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
def build_attentive_scale_net(
    subnet_dict_list: list[dict] = [
        {"top_resolution": (32, 32, 32)},
        {"top_resolution": (8, 8, 8)},
    ]
):
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
    network = LeibNet(nodes, outputs=outputs, name="AttentiveScaleNet")
    return network


# %%
def testing():
    subnet_dict_list = [
        {"top_resolution": (32, 32, 32)},
        {"top_resolution": (8, 8, 8)},
    ]
    leibnet = build_attentive_scale_net(subnet_dict_list)
    # leibnet.array_shapes
    # %%
    inputs = leibnet.get_example_inputs()
    for key, val in inputs.items():
        print(f"{key}: {val.shape}")
    outputs = leibnet(inputs)
    # %%
    for key, val in outputs.items():
        print(f"{key}: {val.shape}")

    # %%
    leibnet.to_mermaid()
    # %%
    leibnet.to_mermaid(separate_arrays=True)

    # %%
    timer = Timer(lambda: leibnet(inputs))
    num, time = timer.autorange()
    print(f"Time per run: {time/num} seconds")
    return leibnet
