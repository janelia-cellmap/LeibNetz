# %%
# Unit tests for the LeibNet architecture using the U-Net constructor

from architectures.torch.leibnetz import LeibNet
from architectures.torch.leibnetz.nodes import ResampleNode, ConvPassNode
import numpy as np
import torch


def build_unet(
    top_resolution=(8, 8, 8),
    downsample_factors=[(2, 2, 2), (2, 2, 2)],
    kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
    input_nc=1,
    output_nc=1,
    base_nc=12,
    nc_increase_factor=2,
):
    from importlib import reload
    import architectures.torch.leibnetz.unet_constructor

    reload(architectures.torch.leibnetz.unet_constructor)
    return architectures.torch.leibnetz.unet_constructor.build_unet()
    assert False, "Fix unet constructor in test_leibnet.py"
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
            base_nc * nc_increase_factor**i,
            base_nc * nc_increase_factor**i,
            kernel_sizes,
            identifier=output_key,
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
            kernel_sizes,
            identifier="output",
        )
    )

    # define network
    network = LeibNet(
        nodes, outputs={"output": [tuple(np.ones(len(top_resolution))), top_resolution]}
    )

    return network


# %%
def test_leibnet(device="cpu"):
    unet = build_unet()
    unet.to(device)

    inputs = {}
    for k, v in unet.input_shapes.items():
        inputs[k] = torch.rand(
            (
                1,
                1,
            )
            + tuple(v[0].astype(int))
        ).to(device)

    outputs = unet(inputs)
    for k, v in outputs.items():
        assert np.all([v.shape[-unet.ndims :] == unet.output_shapes[k][0]])


def test_leibnet_cuda():
    test_leibnet("cuda")


def test_leibnet_cpu():
    test_leibnet("cpu")


# %%
