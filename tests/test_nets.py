import unittest
import torch
import numpy as np
from leibnetz.nets.unet import build_unet
from leibnetz.nets.scalenet import build_subnet as build_scalenet_subnet, build_scalenet
from leibnetz.nets.attentive_scalenet import (
    build_subnet as build_attentive_subnet,
    build_attentive_scale_net,
)
from leibnetz import LeibNet


class TestUNet(unittest.TestCase):
    """Test cases for UNet construction"""

    def test_build_unet_basic_2d(self):
        """Test basic 2D UNet construction"""
        unet = build_unet(
            top_resolution=(16, 16),
            downsample_factors=[(2, 2)],
            kernel_sizes=[(3, 3)],
            input_nc=1,
            output_nc=1,
            base_nc=8,
        )

        self.assertIsInstance(unet, LeibNet)
        self.assertIn("input", unet.input_keys)
        self.assertIn("output", unet.output_keys)

    def test_build_unet_basic_3d(self):
        """Test basic 3D UNet construction"""
        unet = build_unet(
            top_resolution=(8, 8, 8),
            downsample_factors=[(2, 2, 2)],
            kernel_sizes=[(3, 3, 3)],
            input_nc=1,
            output_nc=1,
            base_nc=8,
        )

        self.assertIsInstance(unet, LeibNet)
        self.assertEqual(len(unet.input_keys), 1)
        self.assertEqual(len(unet.output_keys), 1)

    def test_build_unet_with_defaults(self):
        """Test UNet construction with default parameters"""
        unet = build_unet()

        self.assertIsInstance(unet, LeibNet)
        self.assertTrue(len(unet.nodes) > 0)

    def test_build_unet_multiple_downsamples(self):
        """Test UNet construction with multiple downsample factors"""
        unet = build_unet(
            top_resolution=(16, 16, 16),
            downsample_factors=[(2, 2, 2), (2, 2, 2)],
            kernel_sizes=[(3, 3, 3), (3, 3, 3)],
            input_nc=2,
            output_nc=3,
            base_nc=4,
        )

        self.assertIsInstance(unet, LeibNet)
        # Should have more nodes due to multiple downsample layers
        self.assertGreater(len(unet.nodes), 3)

    def test_build_unet_with_norm_layer(self):
        """Test UNet construction with normalization layer"""
        unet = build_unet(top_resolution=(8, 8, 8), norm_layer="batch")

        self.assertIsInstance(unet, LeibNet)

    def test_build_unet_with_residual(self):
        """Test UNet construction with residual connections"""
        unet = build_unet(top_resolution=(8, 8, 8), residual=True)

        self.assertIsInstance(unet, LeibNet)

    def test_build_unet_with_dropout(self):
        """Test UNet construction with dropout"""
        unet = build_unet(top_resolution=(8, 8, 8), dropout_prob=0.2)

        self.assertIsInstance(unet, LeibNet)

    def test_build_unet_custom_activations(self):
        """Test UNet construction with custom activations"""
        unet = build_unet(
            top_resolution=(8, 8, 8), activation="LeakyReLU", final_activation="Tanh"
        )

        self.assertIsInstance(unet, LeibNet)

    def test_build_unet_forward_pass(self):
        """Test that built UNet can perform forward pass"""
        unet = build_unet(
            top_resolution=(8, 8, 8),
            input_nc=1,
            output_nc=1,
            base_nc=4,  # Small for testing
        )

        # Test forward pass
        inputs = unet.get_example_inputs()
        outputs = unet(inputs)

        self.assertIsInstance(outputs, dict)
        self.assertIn("output", outputs)


class TestScaleNet(unittest.TestCase):
    """Test cases for ScaleNet construction"""

    def test_build_scalenet_subnet_basic(self):
        """Test basic ScaleNet subnet construction"""
        nodes, outputs = build_scalenet_subnet(
            subnet_id="test",
            top_resolution=(8, 8, 8),
            downsample_factors=[(2, 2, 2)],
            kernel_sizes=[(3, 3, 3)],
            input_nc=1,
            output_nc=1,
            base_nc=4,
        )

        self.assertIsInstance(nodes, list)
        self.assertIsInstance(outputs, dict)
        self.assertGreater(len(nodes), 0)

    def test_build_scalenet_subnet_with_bottleneck_input(self):
        """Test ScaleNet subnet construction with bottleneck input"""
        # bottleneck_input format: {key: [shape, scale, feature_maps]}
        bottleneck_input = {"test_bottleneck": [(8, 8, 8), (1, 1, 1), 4]}

        nodes, outputs = build_scalenet_subnet(
            bottleneck_input_dict=bottleneck_input,
            subnet_id="test",
            top_resolution=(8, 8, 8),
            base_nc=4,
        )

        self.assertIsInstance(nodes, list)
        self.assertIsInstance(outputs, dict)

    def test_build_scalenet_subnet_with_options(self):
        """Test ScaleNet subnet construction with various options"""
        nodes, outputs = build_scalenet_subnet(
            subnet_id="test",
            top_resolution=(8, 8, 8),
            norm_layer="batch",
            residual=True,
            dropout_prob=0.1,
            squeeze_excitation=True,
            num_final_convs=2,
        )

        self.assertIsInstance(nodes, list)
        self.assertIsInstance(outputs, dict)

    def test_build_scalenet_basic(self):
        """Test basic ScaleNet construction"""
        scale_net = build_scalenet([{"top_resolution": (8, 8, 8), "base_nc": 4}])

        self.assertIsInstance(scale_net, LeibNet)

    def test_build_scalenet_multiple_scales(self):
        """Test ScaleNet construction with multiple scales"""
        subnet_dict_list = [
            {"top_resolution": (8, 8, 8), "base_nc": 4},
            {"top_resolution": (4, 4, 4), "base_nc": 4},
        ]

        scale_net = build_scalenet(subnet_dict_list)

        self.assertIsInstance(scale_net, LeibNet)
        self.assertGreater(len(scale_net.output_keys), 0)

    def test_build_scalenet_forward_pass(self):
        """Test that built ScaleNet can perform forward pass"""
        scale_net = build_scalenet([{"top_resolution": (8, 8, 8), "base_nc": 4}])

        # Test forward pass
        inputs = scale_net.get_example_inputs()
        outputs = scale_net(inputs)

        self.assertIsInstance(outputs, dict)


class TestAttentiveScaleNet(unittest.TestCase):
    """Test cases for Attentive ScaleNet construction"""

    def test_build_attentive_subnet_basic(self):
        """Test basic Attentive ScaleNet subnet construction"""
        nodes, outputs = build_attentive_subnet(
            subnet_id="test",
            top_resolution=(8, 8, 8),
            downsample_factors=[(2, 2, 2)],
            kernel_sizes=[(3, 3, 3)],
            input_nc=1,
            output_nc=1,
            base_nc=4,
        )

        self.assertIsInstance(nodes, list)
        self.assertIsInstance(outputs, dict)
        self.assertGreater(len(nodes), 0)

    def test_build_attentive_subnet_with_bottleneck_input(self):
        """Test Attentive ScaleNet subnet construction with bottleneck input"""
        # bottleneck_input format: {key: [shape, scale, feature_maps]}
        bottleneck_input = {"test_bottleneck": [(8, 8, 8), (1, 1, 1), 4]}

        nodes, outputs = build_attentive_subnet(
            bottleneck_input_dict=bottleneck_input,
            subnet_id="test",
            top_resolution=(8, 8, 8),
            base_nc=4,
        )

        self.assertIsInstance(nodes, list)
        self.assertIsInstance(outputs, dict)

    def test_build_attentive_subnet_with_options(self):
        """Test Attentive ScaleNet subnet construction with various options"""
        nodes, outputs = build_attentive_subnet(
            subnet_id="test",
            top_resolution=(8, 8, 8),
            norm_layer="batch",
            residual=True,
            dropout_prob=0.1,
            squeeze_excitation=True,
            num_final_convs=2,
        )

        self.assertIsInstance(nodes, list)
        self.assertIsInstance(outputs, dict)

    def test_build_attentive_scale_net_basic(self):
        """Test basic Attentive ScaleNet construction"""
        scale_net = build_attentive_scale_net(
            [{"top_resolution": (8, 8, 8), "base_nc": 4}]
        )

        self.assertIsInstance(scale_net, LeibNet)

    def test_build_attentive_scale_net_multiple_scales(self):
        """Test Attentive ScaleNet construction with multiple scales"""
        subnet_dict_list = [
            {"top_resolution": (8, 8, 8), "base_nc": 4},
            {"top_resolution": (4, 4, 4), "base_nc": 4},
        ]

        scale_net = build_attentive_scale_net(subnet_dict_list)

        self.assertIsInstance(scale_net, LeibNet)
        self.assertGreater(len(scale_net.output_keys), 0)

    def test_build_attentive_scale_net_forward_pass(self):
        """Test that built Attentive ScaleNet can perform forward pass"""
        scale_net = build_attentive_scale_net(
            [{"top_resolution": (8, 8, 8), "base_nc": 4}]
        )

        # Test forward pass
        inputs = scale_net.get_example_inputs()
        outputs = scale_net(inputs)

        self.assertIsInstance(outputs, dict)


class TestNetsIntegration(unittest.TestCase):
    """Integration tests for nets functionality"""

    def test_all_net_types_can_be_created(self):
        """Test that all net types can be successfully created"""
        # UNet
        unet = build_unet(top_resolution=(8, 8, 8), base_nc=2)
        self.assertIsInstance(unet, LeibNet)

        # ScaleNet
        scale_net = build_scalenet([{"top_resolution": (8, 8, 8), "base_nc": 2}])
        self.assertIsInstance(scale_net, LeibNet)

        # Attentive ScaleNet
        attentive_scale_net = build_attentive_scale_net(
            [{"top_resolution": (8, 8, 8), "base_nc": 2}]
        )
        self.assertIsInstance(attentive_scale_net, LeibNet)

    def test_different_dimensionalities(self):
        """Test that nets work with different spatial dimensionalities"""
        # 2D
        unet_2d = build_unet(
            top_resolution=(16, 16), downsample_factors=[(2, 2)], base_nc=2
        )
        self.assertIsInstance(unet_2d, LeibNet)

        # 3D
        unet_3d = build_unet(
            top_resolution=(8, 8, 8), downsample_factors=[(2, 2, 2)], base_nc=2
        )
        self.assertIsInstance(unet_3d, LeibNet)

    def test_nets_have_valid_input_output_keys(self):
        """Test that all nets have proper input/output keys"""
        unet = build_unet(base_nc=2)
        self.assertTrue(len(unet.input_keys) > 0)
        self.assertTrue(len(unet.output_keys) > 0)

        scale_net = build_scalenet([{"base_nc": 2}])
        self.assertTrue(len(scale_net.input_keys) > 0)
        self.assertTrue(len(scale_net.output_keys) > 0)


if __name__ == "__main__":
    unittest.main()
