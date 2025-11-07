# %%
# Unit tests for the LeibNet architecture using the U-Net constructor

import numpy as np
import torch

from leibnetz import LeibNet
from leibnetz.nodes import ConvPassNode, ResampleNode


def build_unet(
    top_resolution=(8, 8, 8),
    downsample_factors=[(2, 2, 2), (2, 2, 2)],
    kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
    input_nc=1,
    output_nc=1,
    base_nc=12,
    nc_increase_factor=2,
):
    # from leibnetz.unet_constructor import build_unet

    # return build_unet(
    #     top_resolution,
    #     downsample_factors,
    #     kernel_sizes,
    #     input_nc,
    #     output_nc,
    #     base_nc,
    #     nc_increase_factor,
    # )
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
            base_nc * nc_increase_factor ** (i),
            base_nc * nc_increase_factor ** (i + 1),
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
def test_leibnet(device="cpu", **unet_kwargs):
    unet = build_unet(**unet_kwargs)
    unet.to(device)
    input_nc = unet_kwargs.get("input_nc", 1)

    inputs = {}
    for k, v in unet._input_shapes.items():
        inputs[k] = torch.rand(
            (
                1,
                input_nc,
            )
            + tuple(v[0].astype(int))
        ).to(device)

    # test forward pass
    outputs = unet(inputs)
    for k, v in outputs.items():
        assert np.all([v.shape[-unet.ndims :] == unet._output_shapes[k][0]])

    # test backward pass
    loss = torch.sum(torch.stack([torch.sum(v) for v in outputs.values()]))
    loss.backward()

    # test optimization
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
    optimizer.zero_grad()
    outputs = unet(inputs)
    loss = torch.sum(torch.stack([torch.sum(v) for v in outputs.values()]))
    loss.backward()
    optimizer.step()


def test_leibnet_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        print("GPU not available.")
        return

    test_leibnet(device)
    test_leibnet(
        device,
        downsample_factors=[(3, 3, 3), (2, 2, 2), (2, 2, 2)],
        kernel_sizes=[(5, 5, 5), (3, 3, 3)],
        input_nc=2,
        output_nc=1,
    )


def test_leibnet_cpu():
    test_leibnet("cpu")
    test_leibnet(
        "cpu",
        downsample_factors=[(3, 3, 3), (2, 2, 2), (2, 2, 2)],
        kernel_sizes=[(5, 5, 5), (3, 3, 3)],
        input_nc=2,
        output_nc=1,
    )


# %%
if __name__ == "__main__":
    test_leibnet_cpu()
    test_leibnet_gpu()
