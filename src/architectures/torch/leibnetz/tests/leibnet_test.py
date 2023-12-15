# %%
# Unit tests for the LeibNet architecture using the U-Net constructor

import torch
from architectures.torch.leibnetz import LeibNet
from architectures.torch.leibnetz.nodes import ResampleNode, ConvPassNode
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
    # from architectures.torch.leibnetz.unet_constructor import build_unet

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
    for k, v in unet.input_shapes.items():
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
        assert np.all([v.shape[-unet.ndims :] == unet.output_shapes[k][0]])

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


def test_leibnet_cuda():
    test_leibnet("cuda")
    test_leibnet(
        "cuda",
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
    test_leibnet_cuda()

# %% [markdown]
# ```mermaid
# graph LR
# 	node-in_conv_0([in_conv_0])
# 	node-downsample_0[\downsample_0/]
# 	node-in_conv_1([in_conv_1])
# 	node-downsample_1[\downsample_1/]
# 	node-bottleneck([bottleneck])
# 	node-upsample_1[/upsample_1\]
# 	node-out_conv_1([out_conv_1])
# 	node-upsample_0[/upsample_0\]
# 	node-out_conv_0([out_conv_0])
# 	node-output([output])
# 	out_conv_0[out_conv_0: 20x20x20]
# 	upsample_0[upsample_0: 28x28x28]
# 	in_conv_0[in_conv_0: 76x76x76]
# 	out_conv_1[out_conv_1: 14x14x14]
# 	upsample_1[upsample_1: 20x20x20]
# 	in_conv_1[in_conv_1: 32x32x32]
# 	bottleneck[bottleneck: 10x10x10]
# 	downsample_1[downsample_1: 16x16x16]
# 	downsample_0[downsample_0: 38x38x38]
# 	input[input: 82x82x82]
# 	subgraph 8nmx8nmx8nm
# 		input---node-in_conv_0
# 	end
# 	subgraph 8nmx8nmx8nm
# 		node-in_conv_0-->in_conv_0
# 	end
# 	subgraph 8nmx8nmx8nm
# 		in_conv_0-.-node-downsample_0
# 	end
# 	subgraph 16nmx16nmx16nm
# 		node-downsample_0-.->downsample_0
# 	end
# 	subgraph 16nmx16nmx16nm
# 		downsample_0---node-in_conv_1
# 	end
# 	subgraph 16nmx16nmx16nm
# 		node-in_conv_1-->in_conv_1
# 	end
# 	subgraph 16nmx16nmx16nm
# 		in_conv_1-.-node-downsample_1
# 	end
# 	subgraph 32nmx32nmx32nm
# 		node-downsample_1-.->downsample_1
# 	end
# 	subgraph 32nmx32nmx32nm
# 		downsample_1---node-bottleneck
# 	end
# 	subgraph 32nmx32nmx32nm
# 		node-bottleneck-->bottleneck
# 	end
# 	subgraph 32nmx32nmx32nm
# 		bottleneck===node-upsample_1
# 	end
# 	subgraph 16nmx16nmx16nm
# 		node-upsample_1==>upsample_1
# 	end
# 	subgraph 16nmx16nmx16nm
# 		upsample_1---node-out_conv_1
# 	end
# 	subgraph 16nmx16nmx16nm
# 		in_conv_1---node-out_conv_1
# 	end
# 	subgraph 16nmx16nmx16nm
# 		node-out_conv_1-->out_conv_1
# 	end
# 	subgraph 16nmx16nmx16nm
# 		out_conv_1===node-upsample_0
# 	end
# 	subgraph 8nmx8nmx8nm
# 		node-upsample_0==>upsample_0
# 	end
# 	subgraph 8nmx8nmx8nm
# 		upsample_0---node-out_conv_0
# 	end
# 	subgraph 8nmx8nmx8nm
# 		in_conv_0---node-out_conv_0
# 	end
# 	subgraph 8nmx8nmx8nm
# 		node-out_conv_0-->out_conv_0
# 	end
# 	subgraph 8nmx8nmx8nm
# 		out_conv_0---node-output
# 	end
# 	subgraph 8nmx8nmx8nm
# 		node-output-->output
# 	end
# ```


# %%
