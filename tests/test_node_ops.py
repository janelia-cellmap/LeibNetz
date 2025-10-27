import unittest
import torch
import torch.nn as nn
import numpy as np
from leibnetz.nodes.node_ops import ConvPass, SqueezeExcitation3d


class TestConvPass(unittest.TestCase):
    """Test cases for ConvPass class"""

    def test_init_basic_2d(self):
        """Test basic ConvPass initialization with 2D kernels"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=8,
            kernel_sizes=[(3, 3), (3, 3)],
        )
        
        self.assertEqual(conv_pass.input_nc, 1)
        self.assertEqual(conv_pass.output_nc, 8)
        self.assertEqual(conv_pass.dims, 2)
        self.assertIsInstance(conv_pass.activation, nn.ReLU)
        self.assertIsInstance(conv_pass.final_activation, nn.ReLU)
        self.assertFalse(conv_pass.residual)

    def test_init_basic_3d(self):
        """Test basic ConvPass initialization with 3D kernels"""
        conv_pass = ConvPass(
            input_nc=2,
            output_nc=16,
            kernel_sizes=[(3, 3, 3), (5, 5, 5)],
        )
        
        self.assertEqual(conv_pass.input_nc, 2)
        self.assertEqual(conv_pass.output_nc, 16)
        self.assertEqual(conv_pass.dims, 3)

    def test_init_with_string_activation(self):
        """Test ConvPass initialization with string activation"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            activation="ReLU",
            final_activation="Sigmoid"
        )
        
        self.assertIsInstance(conv_pass.activation, nn.ReLU)
        self.assertIsInstance(conv_pass.final_activation, nn.Sigmoid)

    def test_init_with_callable_activation(self):
        """Test ConvPass initialization with callable activation"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            activation=nn.LeakyReLU,
            final_activation=nn.Tanh
        )
        
        self.assertIsInstance(conv_pass.activation, nn.LeakyReLU)
        self.assertIsInstance(conv_pass.final_activation, nn.Tanh)

    def test_init_with_none_activation(self):
        """Test ConvPass initialization with None activation"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            activation=None,
            final_activation=None
        )
        
        self.assertIsInstance(conv_pass.activation, nn.Identity)
        self.assertIsInstance(conv_pass.final_activation, nn.Identity)

    def test_init_with_batch_norm_2d(self):
        """Test ConvPass initialization with batch normalization for 2D"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3)],
            norm_layer="batch"
        )
        
        # Check that BatchNorm2d is in the sequential
        has_batch_norm = any(isinstance(layer, nn.BatchNorm2d) for layer in conv_pass.conv_pass)
        self.assertTrue(has_batch_norm)

    def test_init_with_batch_norm_3d(self):
        """Test ConvPass initialization with batch normalization for 3D"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            norm_layer="batch"
        )
        
        # Check that BatchNorm3d is in the sequential
        has_batch_norm = any(isinstance(layer, nn.BatchNorm3d) for layer in conv_pass.conv_pass)
        self.assertTrue(has_batch_norm)

    def test_init_with_instance_norm_2d(self):
        """Test ConvPass initialization with instance normalization for 2D"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3)],
            norm_layer="instance"
        )
        
        # Check that InstanceNorm2d is in the sequential
        has_instance_norm = any(isinstance(layer, nn.InstanceNorm2d) for layer in conv_pass.conv_pass)
        self.assertTrue(has_instance_norm)

    def test_init_with_instance_norm_3d(self):
        """Test ConvPass initialization with instance normalization for 3D"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            norm_layer="instance"
        )
        
        # Check that InstanceNorm3d is in the sequential
        has_instance_norm = any(isinstance(layer, nn.InstanceNorm3d) for layer in conv_pass.conv_pass)
        self.assertTrue(has_instance_norm)

    def test_init_with_group_norm_2d_error(self):
        """Test ConvPass initialization with group normalization for 2D raises error (known bug)"""
        with self.assertRaises(TypeError):
            ConvPass(
                input_nc=1,
                output_nc=4,
                kernel_sizes=[(3, 3)],
                norm_layer="group"
            )

    def test_init_with_group_norm_3d_error(self):
        """Test ConvPass initialization with group normalization for 3D raises error (known bug)"""
        with self.assertRaises(TypeError):
            ConvPass(
                input_nc=1,
                output_nc=4,
                kernel_sizes=[(3, 3, 3)],
                norm_layer="group"
            )

    def test_init_with_unknown_norm_layer_error(self):
        """Test ConvPass initialization with unknown normalization layer raises error"""
        with self.assertRaises(ValueError):
            ConvPass(
                input_nc=1,
                output_nc=4,
                kernel_sizes=[(3, 3, 3)],
                norm_layer="unknown"
            )

    def test_init_with_unsupported_dims_norm_error(self):
        """Test ConvPass initialization with unsupported dims for normalization raises error"""
        with self.assertRaises(RuntimeError):
            ConvPass(
                input_nc=1,
                output_nc=4,
                kernel_sizes=[(3,)],  # 1D not supported
                norm_layer="batch"
            )

    def test_init_with_dropout(self):
        """Test ConvPass initialization with dropout"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            dropout_prob=0.2
        )
        
        # Check that Dropout is in the sequential
        has_dropout = any(isinstance(layer, nn.Dropout) for layer in conv_pass.conv_pass)
        self.assertTrue(has_dropout)

    def test_init_with_zero_dropout(self):
        """Test ConvPass initialization with zero dropout (should not add dropout layer)"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            dropout_prob=0.0
        )
        
        # Check that Dropout is NOT in the sequential when prob is 0
        has_dropout = any(isinstance(layer, nn.Dropout) for layer in conv_pass.conv_pass)
        self.assertFalse(has_dropout)

    def test_init_with_unsupported_dims_error(self):
        """Test ConvPass initialization with unsupported dimensions raises error"""
        with self.assertRaises(ValueError):
            ConvPass(
                input_nc=1,
                output_nc=4,
                kernel_sizes=[(3,)]  # 1D not supported
            )

    def test_init_with_squeeze_excitation_2d(self):
        """Test ConvPass initialization with squeeze excitation for 2D"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=8,
            kernel_sizes=[(3, 3), (3, 3)],  # 2 layers
            squeeze_excitation=True,
            squeeze_ratio=2
        )
        
        # Check that SqueezeExcitation is in the sequential
        from torchvision.ops import SqueezeExcitation
        has_se = any(isinstance(layer, SqueezeExcitation) for layer in conv_pass.conv_pass)
        self.assertTrue(has_se)

    def test_init_with_squeeze_excitation_3d(self):
        """Test ConvPass initialization with squeeze excitation for 3D"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=8,
            kernel_sizes=[(3, 3, 3), (3, 3, 3)],  # 2 layers
            squeeze_excitation=True,
            squeeze_ratio=2
        )
        
        # Check that SqueezeExcitation3d is in the sequential
        has_se = any(isinstance(layer, SqueezeExcitation3d) for layer in conv_pass.conv_pass)
        self.assertTrue(has_se)

    def test_init_with_squeeze_excitation_unsupported_dims_error(self):
        """Test ConvPass initialization with squeeze excitation for unsupported dims raises error"""
        with self.assertRaises(ValueError):
            ConvPass(
                input_nc=1,
                output_nc=8,
                kernel_sizes=[(3,), (3,)],  # 1D not supported
                squeeze_excitation=True
            )

    def test_init_with_residual_input_less_than_output(self):
        """Test ConvPass initialization with residual connection when input_nc < output_nc"""
        conv_pass = ConvPass(
            input_nc=2,
            output_nc=8,  # 8 % 2 == 0
            kernel_sizes=[(3, 3, 3)],
            residual=True
        )
        
        self.assertTrue(conv_pass.residual)
        self.assertTrue(hasattr(conv_pass, 'x_init_map'))

    def test_init_with_residual_input_greater_than_output(self):
        """Test ConvPass initialization with residual connection when input_nc > output_nc"""
        conv_pass = ConvPass(
            input_nc=8,
            output_nc=4,  # 8 % 4 == 0
            kernel_sizes=[(3, 3, 3)],
            residual=True
        )
        
        self.assertTrue(conv_pass.residual)
        self.assertTrue(hasattr(conv_pass, 'x_init_map'))

    def test_init_with_residual_no_common_divisor(self):
        """Test ConvPass initialization with residual connection when channels don't divide evenly"""
        conv_pass = ConvPass(
            input_nc=3,
            output_nc=7,  # no common divisor
            kernel_sizes=[(3, 3, 3)],
            residual=True
        )
        
        self.assertTrue(conv_pass.residual)
        self.assertTrue(hasattr(conv_pass, 'x_init_map'))

    def test_forward_basic(self):
        """Test basic forward pass"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            padding='same'
        )
        
        x = torch.randn(1, 1, 8, 8, 8)
        output = conv_pass.forward(x)
        
        self.assertEqual(output.shape[0], 1)  # batch size
        self.assertEqual(output.shape[1], 4)  # output channels

    def test_forward_with_residual_valid_padding(self):
        """Test forward pass with residual connection and valid padding"""
        conv_pass = ConvPass(
            input_nc=2,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            residual=True,
            padding='valid'
        )
        
        x = torch.randn(1, 2, 10, 10, 10)
        output = conv_pass.forward(x)
        
        self.assertEqual(output.shape[0], 1)  # batch size
        self.assertEqual(output.shape[1], 4)  # output channels
        # With valid padding and kernel size 3, output should be 8x8x8
        self.assertEqual(output.shape[2:], (8, 8, 8))

    def test_forward_with_residual_same_padding(self):
        """Test forward pass with residual connection and same padding"""
        conv_pass = ConvPass(
            input_nc=2,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)],
            residual=True,
            padding='same'
        )
        
        x = torch.randn(1, 2, 8, 8, 8)
        output = conv_pass.forward(x)
        
        self.assertEqual(output.shape[0], 1)  # batch size
        self.assertEqual(output.shape[1], 4)  # output channels
        # With same padding, output should maintain input size
        self.assertEqual(output.shape[2:], (8, 8, 8))

    def test_crop_method(self):
        """Test the crop method"""
        conv_pass = ConvPass(
            input_nc=1,
            output_nc=4,
            kernel_sizes=[(3, 3, 3)]
        )
        
        x = torch.randn(1, 1, 10, 10, 10)
        target_shape = (6, 6, 6)
        
        cropped = conv_pass.crop(x, target_shape)
        
        self.assertEqual(cropped.shape[0], 1)  # batch size preserved
        self.assertEqual(cropped.shape[1], 1)  # channels preserved
        self.assertEqual(cropped.shape[2:], target_shape)


class TestSqueezeExcitation3d(unittest.TestCase):
    """Test cases for SqueezeExcitation3d class"""

    def test_init_basic(self):
        """Test basic SqueezeExcitation3d initialization"""
        se = SqueezeExcitation3d(
            input_channels=16,
            squeeze_channels=4
        )
        
        self.assertIsInstance(se.avgpool, nn.AdaptiveAvgPool3d)
        self.assertIsInstance(se.fc1, nn.Conv3d)
        self.assertIsInstance(se.fc2, nn.Conv3d)
        self.assertIsInstance(se.activation, nn.ReLU)
        self.assertIsInstance(se.scale_activation, nn.Sigmoid)

    def test_init_with_custom_activations(self):
        """Test SqueezeExcitation3d initialization with custom activations"""
        se = SqueezeExcitation3d(
            input_channels=16,
            squeeze_channels=4,
            activation=nn.LeakyReLU,
            scale_activation=nn.Tanh
        )
        
        self.assertIsInstance(se.activation, nn.LeakyReLU)
        self.assertIsInstance(se.scale_activation, nn.Tanh)

    def test_scale_method(self):
        """Test the _scale method"""
        se = SqueezeExcitation3d(
            input_channels=16,
            squeeze_channels=4
        )
        
        x = torch.randn(1, 16, 8, 8, 8)
        scale = se._scale(x)
        
        self.assertEqual(scale.shape, (1, 16, 1, 1, 1))

    def test_forward(self):
        """Test forward pass"""
        se = SqueezeExcitation3d(
            input_channels=16,
            squeeze_channels=4
        )
        
        x = torch.randn(1, 16, 8, 8, 8)
        output = se.forward(x)
        
        self.assertEqual(output.shape, x.shape)  # Output should have same shape as input
        
    def test_forward_preserves_gradient(self):
        """Test that forward pass preserves gradients"""
        se = SqueezeExcitation3d(
            input_channels=16,
            squeeze_channels=4
        )
        
        x = torch.randn(1, 16, 8, 8, 8, requires_grad=True)
        output = se.forward(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


if __name__ == "__main__":
    unittest.main()