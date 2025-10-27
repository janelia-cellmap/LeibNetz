import unittest
import torch
import torch.nn as nn
import numpy as np
from leibnetz.nodes.wrapper_node import WrapperNode


class TestWrapperNode(unittest.TestCase):
    """Test cases for WrapperNode class"""

    def test_init_basic(self):
        """Test basic WrapperNode initialization"""
        model = nn.Conv3d(1, 1, 3)
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output"],
            identifier="test_wrapper",
        )

        self.assertEqual(node.input_keys, ["input"])
        self.assertEqual(node.output_keys, ["output"])
        self.assertEqual(node.id, "test_wrapper")  # Node stores identifier as 'id'
        self.assertEqual(node.model, model)
        self.assertEqual(node.color, "#FF0000")

    def test_init_multiple_keys(self):
        """Test WrapperNode initialization with multiple input/output keys"""
        model = nn.Conv3d(2, 3, 3)
        node = WrapperNode(
            model=model,
            input_keys=["input1", "input2"],
            output_keys=["output1", "output2", "output3"],
            identifier="test_multi_wrapper",
        )

        self.assertEqual(node.input_keys, ["input1", "input2"])
        self.assertEqual(node.output_keys, ["output1", "output2", "output3"])
        self.assertEqual(
            node.id, "test_multi_wrapper"
        )  # Node stores identifier as 'id'

    def test_convolution_crop_property(self):
        """Test convolution_crop property calculation"""
        # Create a model with known kernel sizes
        model = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=3),  # loses 2 voxels per dimension
            nn.Conv3d(2, 1, kernel_size=5),  # loses 4 voxels per dimension
        )
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output"],
            identifier="test_crop",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        # Total crop should be (3-1) + (5-1) = 6 per dimension
        expected_crop = np.array([6, 6, 6])
        np.testing.assert_array_equal(node.convolution_crop, expected_crop)

    def test_convolution_crop_same_padding(self):
        """Test convolution_crop with 'same' padding"""
        # Note: This test may not work as intended because PyTorch doesn't use string padding
        # But we test the logic for modules without kernel_size
        model = nn.Identity()
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output"],
            identifier="test_same_padding",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        # Identity layer should have zero crop
        expected_crop = np.array([0, 0, 0])
        np.testing.assert_array_equal(node.convolution_crop, expected_crop)

    def test_forward_single_input_output(self):
        """Test forward pass with single input and output"""
        model = nn.Conv3d(1, 1, 3, padding=1)  # padding=1 to maintain size
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output"],
            identifier="test_forward",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_tensor = torch.randn(1, 1, 8, 8, 8)
        inputs = {"input": input_tensor}

        output = node.forward(inputs)

        self.assertIsInstance(output, dict)
        self.assertIn("output", output)
        self.assertEqual(output["output"].shape[0], 1)  # batch size
        self.assertEqual(output["output"].shape[1], 1)  # channels

    def test_forward_multiple_inputs(self):
        """Test forward pass with multiple inputs"""
        model = nn.Conv3d(2, 1, 3, padding=1)  # 2 input channels, 1 output
        node = WrapperNode(
            model=model,
            input_keys=["input1", "input2"],
            output_keys=["output"],
            identifier="test_multi_input",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input1 = torch.randn(1, 1, 8, 8, 8)
        input2 = torch.randn(1, 1, 8, 8, 8)
        inputs = {"input1": input1, "input2": input2}

        output = node.forward(inputs)

        self.assertIsInstance(output, dict)
        self.assertIn("output", output)
        self.assertEqual(output["output"].shape[1], 1)  # 1 output channel

    def test_forward_multiple_outputs(self):
        """Test forward pass with multiple outputs"""
        model = nn.Conv3d(1, 3, 3, padding=1)  # 1 input, 3 output channels
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output1", "output2", "output3"],
            identifier="test_multi_output",
            output_key_channels=[1, 1, 1],  # 1 channel per output
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_tensor = torch.randn(1, 1, 8, 8, 8)
        inputs = {"input": input_tensor}

        output = node.forward(inputs)

        self.assertIsInstance(output, dict)
        self.assertIn("output1", output)
        self.assertIn("output2", output)
        self.assertIn("output3", output)
        # Each output should have 1 channel (3 channels split according to output_key_channels)
        self.assertEqual(output["output1"].shape[1], 1)
        self.assertEqual(output["output2"].shape[1], 1)
        self.assertEqual(output["output3"].shape[1], 1)

    def test_get_min_crops_equal_sizes(self):
        """Test get_min_crops with equal input sizes"""
        model = nn.Conv3d(2, 1, 3)
        node = WrapperNode(
            model=model,
            input_keys=["input1", "input2"],
            output_keys=["output"],
            identifier="test_min_crops",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input1 = torch.randn(1, 1, 8, 8, 8)
        input2 = torch.randn(1, 1, 8, 8, 8)
        inputs = {"input1": input1, "input2": input2}

        cropped_inputs = node.get_min_crops(inputs)

        # Sizes should remain the same
        self.assertEqual(cropped_inputs["input1"].shape, input1.shape)
        self.assertEqual(cropped_inputs["input2"].shape, input2.shape)

    def test_get_min_crops_different_sizes(self):
        """Test get_min_crops with different input sizes"""
        model = nn.Conv3d(2, 1, 3)
        node = WrapperNode(
            model=model,
            input_keys=["input1", "input2"],
            output_keys=["output"],
            identifier="test_min_crops_diff",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input1 = torch.randn(1, 1, 10, 10, 10)
        input2 = torch.randn(1, 1, 8, 8, 8)
        inputs = {"input1": input1, "input2": input2}

        cropped_inputs = node.get_min_crops(inputs)

        # Both should be cropped to the smaller size (8x8x8)
        self.assertEqual(cropped_inputs["input1"].shape[-3:], (8, 8, 8))
        self.assertEqual(cropped_inputs["input2"].shape[-3:], (8, 8, 8))

    def test_crop_function(self):
        """Test the crop utility function"""
        model = nn.Conv3d(1, 1, 3)
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output"],
            identifier="test_crop_func",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_tensor = torch.randn(1, 1, 10, 10, 10)
        target_shape = np.array([6, 6, 6])

        cropped = node.crop(input_tensor, target_shape)

        self.assertEqual(cropped.shape[-3:], (6, 6, 6))
        self.assertEqual(cropped.shape[:2], (1, 1))  # batch and channel dims unchanged

    def test_get_input_from_output_shape(self):
        """Test get_input_from_output_shape method"""
        model = nn.Conv3d(1, 1, 3)  # kernel size 3, loses 2 voxels per dim
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output"],
            identifier="test_input_from_output",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        output_shape = np.array([8, 8, 8])
        input_shapes = node.get_input_from_output_shape(output_shape)

        self.assertIsInstance(input_shapes, dict)
        self.assertIn("input", input_shapes)
        input_shape, scale = input_shapes["input"]
        # Should add convolution crop (2 per dim for kernel size 3)
        expected_shape = output_shape + np.array([2, 2, 2])
        np.testing.assert_array_equal(input_shape, expected_shape)

    def test_get_output_from_input_shape(self):
        """Test get_output_from_input_shape method"""
        model = nn.Conv3d(1, 1, 3)  # kernel size 3, loses 2 voxels per dim
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output"],
            identifier="test_output_from_input",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_shape = np.array([10, 10, 10])
        output_shapes = node.get_output_from_input_shape(input_shape)

        self.assertIsInstance(output_shapes, dict)
        self.assertIn("output", output_shapes)
        output_shape, scale = output_shapes["output"]
        # Should subtract convolution crop and factor crop
        # For simple case, factor crop should be zero
        expected_shape = input_shape - np.array([2, 2, 2])  # convolution crop only
        np.testing.assert_array_equal(output_shape, expected_shape)

    def test_factor_crop(self):
        """Test factor_crop method"""
        model = nn.Conv3d(1, 1, 3)
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output"],
            identifier="test_factor_crop",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_shape = np.array([10, 10, 10])
        factor_crop = node.factor_crop(input_shape)

        # With unit scale and least_common_scale, factor crop should be minimal
        self.assertIsInstance(factor_crop, np.ndarray)
        self.assertEqual(len(factor_crop), 3)

    def test_crop_to_factor(self):
        """Test crop_to_factor method"""
        model = nn.Conv3d(1, 1, 3, padding=1)  # padding to avoid size issues
        node = WrapperNode(
            model=model,
            input_keys=["input"],
            output_keys=["output"],
            identifier="test_crop_to_factor",
        )

        # Initialize the node
        node.set_scale([1, 1, 1])
        node.set_least_common_scale([1, 1, 1])

        input_tensor = torch.randn(1, 1, 20, 20, 20)
        cropped = node.crop_to_factor(input_tensor)

        # Should return a tensor with appropriate cropping
        self.assertIsInstance(cropped, torch.Tensor)
        self.assertEqual(cropped.shape[0], 1)  # batch size preserved
        self.assertEqual(cropped.shape[1], 1)  # channels preserved


if __name__ == "__main__":
    unittest.main()
