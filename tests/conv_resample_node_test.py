import unittest
import torch
import numpy as np
from leibnetz.nodes.conv_resample_node import ConvResampleNode


class TestConvResampleNode(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures"""
        pass

    def test_init_conv_pass(self):
        """Test ConvResampleNode initialization with unit scale factor (conv pass)"""
        node = ConvResampleNode(
            input_keys=["input"],
            output_keys=["output"],
            scale_factor=(1, 1, 1),
            kernel_sizes=[[3, 3, 3]],
            input_nc=3,
            output_nc=3,
            identifier="test_conv_pass",
        )
        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        self.assertEqual(node._type, "conv_pass")
        self.assertEqual(node.color, "#00FF00")

    def test_init_conv_downsample(self):
        """Test ConvResampleNode initialization with downsampling"""
        node = ConvResampleNode(
            input_keys=["input"],
            output_keys=["output"],
            scale_factor=(0.5, 0.5, 0.5),
            kernel_sizes=[[3, 3, 3]],
            input_nc=3,
            output_nc=3,
            identifier="test_conv_downsample",
        )
        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        self.assertEqual(node._type, "conv_downsample")
        self.assertEqual(node.color, "#00FFFF")

    def test_init_conv_upsample(self):
        """Test ConvResampleNode initialization with upsampling"""
        node = ConvResampleNode(
            input_keys=["input"],
            output_keys=["output"],
            scale_factor=(2, 2, 2),
            kernel_sizes=[[3, 3, 3]],
            input_nc=3,
            output_nc=3,
            identifier="test_conv_upsample",
        )
        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        self.assertEqual(node._type, "conv_upsample")
        self.assertEqual(node.color, "#FFFF00")

    def test_init_mixed_scale_error(self):
        """Test that mixed up/downsampling raises NotImplementedError"""
        with self.assertRaises(NotImplementedError):
            ConvResampleNode(
                input_keys=["input"],
                output_keys=["output"],
                scale_factor=(0.5, 2, 1),  # mixed up and down
                kernel_sizes=[[3, 3, 3]],
                input_nc=3,
                output_nc=3,
                identifier="test_mixed",
            )

    def test_scale_factor_property(self):
        """Test scale_factor property is converted to numpy array"""
        node = ConvResampleNode(
            input_keys=["input"],
            output_keys=["output"],
            scale_factor=[2, 2, 2],  # list input
            kernel_sizes=[[3, 3, 3]],
            input_nc=3,
            output_nc=3,
            identifier="test_property",
        )

        self.assertIsInstance(node.scale_factor, np.ndarray)
        np.testing.assert_array_equal(node.scale_factor, np.array([2, 2, 2]))

    def test_forward_conv_pass(self):
        """Test forward pass with convolution (no resampling)"""
        node = ConvResampleNode(
            input_keys=["input"],
            output_keys=["output"],
            scale_factor=(1, 1, 1),
            kernel_sizes=[[3, 3, 3]],
            input_nc=3,
            output_nc=3,
            identifier="test_forward_conv",
        )
        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_tensor = torch.randn(1, 3, 8, 8, 8)
        inputs = {"input": input_tensor}

        output = node.forward(inputs)
        self.assertIsInstance(output, dict)
        self.assertIn("output", output)
        # With 3x3x3 kernel and 'valid' padding, we lose 2 pixels on each side: 8 - 2 = 6
        expected_shape = torch.Size([1, 3, 6, 6, 6])
        self.assertEqual(output["output"].shape, expected_shape)

    def test_forward_conv_downsample(self):
        """Test forward pass with conv downsampling"""
        node = ConvResampleNode(
            input_keys=["input"],
            output_keys=["output"],
            scale_factor=(0.5, 0.5, 0.5),
            kernel_sizes=[[3, 3, 3]],
            input_nc=3,
            output_nc=3,
            identifier="test_forward_conv_downsample",
        )
        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_tensor = torch.randn(1, 3, 8, 8, 8)
        inputs = {"input": input_tensor}

        output = node.forward(inputs)
        self.assertIsInstance(output, dict)
        self.assertIn("output", output)
        # With downsampling by factor 2 and 3x3x3 kernel, output should be smaller
        self.assertEqual(len(output["output"].shape), 5)  # Should be 5D tensor
        self.assertEqual(output["output"].shape[0], 1)  # Batch size
        self.assertEqual(output["output"].shape[1], 3)  # Channels

    def test_forward_conv_upsample(self):
        """Test forward pass with conv upsampling"""
        node = ConvResampleNode(
            input_keys=["input"],
            output_keys=["output"],
            scale_factor=(2, 2, 2),
            kernel_sizes=[[3, 3, 3]],
            input_nc=3,
            output_nc=3,
            identifier="test_forward_conv_upsample",
        )
        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_tensor = torch.randn(1, 3, 4, 4, 4)
        inputs = {"input": input_tensor}

        output = node.forward(inputs)
        self.assertIsInstance(output, dict)
        self.assertIn("output", output)
        # With upsampling by factor 2, output should be larger
        self.assertEqual(len(output["output"].shape), 5)  # Should be 5D tensor
        self.assertEqual(output["output"].shape[0], 1)  # Batch size
        self.assertEqual(output["output"].shape[1], 3)  # Channels
        # Spatial dimensions should be roughly doubled (exact size depends on implementation)
        self.assertGreater(output["output"].shape[2], 4)  # Should be larger than input

    def test_get_input_from_output_shape(self):
        """Test input shape calculation from output shape"""
        node = ConvResampleNode(
            input_keys=["input"],
            output_keys=["output"],
            scale_factor=(2, 2, 2),
            kernel_sizes=[[3, 3, 3]],
            input_nc=3,
            output_nc=3,
            identifier="test_input_from_output",
        )
        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        output_shape = np.array([8, 8, 8])
        input_shapes = node.get_input_from_output_shape(output_shape)

        self.assertIsInstance(input_shapes, dict)
        self.assertIn("input", input_shapes)
        # For upsampling by 2, input should be roughly half the output size
        self.assertEqual(len(input_shapes["input"]), 2)  # Should have shape and scale

    def test_get_output_from_input_shape(self):
        """Test output shape calculation from input shape"""
        node = ConvResampleNode(
            input_keys=["input"],
            output_keys=["output"],
            scale_factor=(2, 2, 2),
            kernel_sizes=[[3, 3, 3]],
            input_nc=3,
            output_nc=3,
            identifier="test_output_from_input",
        )
        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_shape = np.array([4, 4, 4])
        output_shapes = node.get_output_from_input_shape(input_shape)

        self.assertIsInstance(output_shapes, dict)
        self.assertIn("output", output_shapes)
        # For upsampling by 2, output should be roughly double the input size
        self.assertEqual(len(output_shapes["output"]), 2)  # Should have shape and scale


if __name__ == "__main__":
    unittest.main()
