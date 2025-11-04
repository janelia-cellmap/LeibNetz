import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from leibnetz.model_wrapper import ModelWrapper


class TestModelWrapper(unittest.TestCase):
    """Test cases for ModelWrapper class"""

    def setUp(self):
        """Set up test fixtures"""
        # Simple test model (3D convolution)
        self.test_model = nn.Conv3d(1, 2, kernel_size=3, padding=1)

        # Example input and output shapes
        self.input_shapes = {"input": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)}}

        self.output_shapes = {"output": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)}}

    def test_init_basic(self):
        """Test basic ModelWrapper initialization"""
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        self.assertEqual(wrapper.model, self.test_model)
        self.assertEqual(wrapper.name, "ModelWrapper")  # default name
        self.assertEqual(wrapper.input_keys, ["input"])
        self.assertEqual(wrapper.output_keys, ["output"])

    def test_init_with_custom_name(self):
        """Test ModelWrapper initialization with custom name"""
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
            name="CustomWrapper",
        )

        self.assertEqual(wrapper.name, "CustomWrapper")

    def test_init_with_multiple_inputs_outputs(self):
        """Test ModelWrapper initialization with multiple inputs and outputs"""
        multi_input_shapes = {
            "input1": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)},
            "input2": {"shape": (4, 4, 4), "scale": (2.0, 2.0, 2.0)},
        }

        multi_output_shapes = {
            "output1": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)},
            "output2": {"shape": (4, 4, 4), "scale": (2.0, 2.0, 2.0)},
        }

        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=multi_input_shapes,
            output_shapes=multi_output_shapes,
        )

        self.assertEqual(set(wrapper.input_keys), {"input1", "input2"})
        self.assertEqual(set(wrapper.output_keys), {"output1", "output2"})

    def test_init_shapes_conversion(self):
        """Test that shapes are properly converted to tuples of ints"""
        # Input with float values that should be converted to ints
        float_shapes = {
            "input": {
                "shape": [8.0, 8.5, 8.9],  # floats that will be converted to ints
                "scale": [1.5, 2.5, 3.5],  # floats that will be converted to ints
            }
        }

        wrapper = ModelWrapper(
            model=self.test_model, input_shapes=float_shapes, output_shapes=float_shapes
        )

        # Check that shapes are converted to tuples of ints
        self.assertEqual(wrapper.input_shapes["input"]["shape"], (8, 8, 8))
        self.assertEqual(wrapper.input_shapes["input"]["scale"], (1, 2, 3))

    def test_get_example_inputs_cpu(self):
        """Test get_example_inputs method with CPU device"""
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        inputs = wrapper.get_example_inputs(device=torch.device("cpu"))

        self.assertIsInstance(inputs, dict)
        self.assertIn("input", inputs)
        self.assertIsInstance(inputs["input"], torch.Tensor)
        self.assertEqual(inputs["input"].device.type, "cpu")
        self.assertEqual(inputs["input"].shape, (1, 1, 8, 8, 8))

    @patch("torch.cuda.is_available")
    def test_get_example_inputs_auto_device_cuda(self, mock_cuda):
        """Test get_example_inputs method with automatic CUDA device detection"""
        mock_cuda.return_value = True
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        # Mock device creation to avoid actual CUDA requirement
        with patch("torch.device") as mock_device:
            mock_device.return_value = torch.device("cpu")  # Return CPU device instead
            with patch.object(
                torch.Tensor, "to", return_value=torch.rand(1)
            ) as mock_to:
                inputs = wrapper.get_example_inputs()
                # Verify CUDA device was requested
                mock_device.assert_called_with("cuda")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_get_example_inputs_auto_device_mps(self, mock_mps, mock_cuda):
        """Test get_example_inputs method with automatic MPS device detection"""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        # Mock device creation to avoid actual MPS requirement
        with patch("torch.device") as mock_device:
            mock_device.return_value = torch.device("cpu")  # Return CPU device instead
            with patch.object(
                torch.Tensor, "to", return_value=torch.rand(1)
            ) as mock_to:
                inputs = wrapper.get_example_inputs()
                # Verify MPS device was requested
                mock_device.assert_called_with("mps")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_get_example_inputs_auto_device_cpu_fallback(self, mock_mps, mock_cuda):
        """Test get_example_inputs method with CPU fallback"""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        inputs = wrapper.get_example_inputs()

        for key in wrapper.input_keys:
            self.assertEqual(inputs[key].device.type, "cpu")

    def test_get_example_inputs_multiple_shapes(self):
        """Test get_example_inputs with multiple input shapes"""
        multi_input_shapes = {
            "input1": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)},
            "input2": {"shape": (4, 4, 4), "scale": (2.0, 2.0, 2.0)},
        }

        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=multi_input_shapes,
            output_shapes=self.output_shapes,
        )

        inputs = wrapper.get_example_inputs(device=torch.device("cpu"))

        self.assertIn("input1", inputs)
        self.assertIn("input2", inputs)
        self.assertEqual(inputs["input1"].shape, (1, 1, 8, 8, 8))
        self.assertEqual(inputs["input2"].shape, (1, 1, 4, 4, 4))

    def test_to_method(self):
        """Test to method for device transfer"""
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        # Mock the model's to method
        with patch.object(self.test_model, "to") as mock_to:
            result = wrapper.to("cpu")
            mock_to.assert_called_once_with("cpu")
            self.assertEqual(result, wrapper)  # Should return self

    def test_forward_with_dict_input(self):
        """Test forward method with dictionary input"""
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        # Create test input
        test_input = torch.randn(1, 1, 8, 8, 8)
        inputs = {"input": test_input}

        outputs = wrapper.forward(inputs)

        self.assertIsInstance(outputs, dict)
        self.assertIn("output", outputs)
        self.assertEqual(outputs["output"].shape[0], 1)  # batch size
        self.assertEqual(outputs["output"].shape[1], 2)  # output channels from Conv3d

    def test_forward_with_list_input_single(self):
        """Test forward method with single-item list input"""
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        # Create test input as single-item list
        test_input = torch.randn(1, 1, 8, 8, 8)
        inputs = [test_input]

        outputs = wrapper.forward(inputs)

        self.assertIsInstance(outputs, dict)
        self.assertIn("output", outputs)

    def test_forward_with_list_input_multiple(self):
        """Test forward method with multi-item list input"""

        # Create a model that expects multiple inputs (concatenating them)
        class MultiInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(
                    2, 2, kernel_size=3, padding=1
                )  # 2 input channels

            def forward(self, inputs):
                # inputs should be a list of tensors
                return self.conv(torch.cat(inputs, dim=1))

        multi_model = MultiInputModel()
        multi_input_shapes = {
            "input1": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)},
            "input2": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)},
        }

        wrapper = ModelWrapper(
            model=multi_model,
            input_shapes=multi_input_shapes,
            output_shapes=self.output_shapes,
        )

        # Create test inputs as dictionary (will be converted to list)
        test_input1 = torch.randn(1, 1, 8, 8, 8)
        test_input2 = torch.randn(1, 1, 8, 8, 8)
        inputs = {"input1": test_input1, "input2": test_input2}

        outputs = wrapper.forward(inputs)

        self.assertIsInstance(outputs, dict)
        self.assertIn("output", outputs)

    def test_forward_with_tensor_input_directly(self):
        """Test forward method when model returns single tensor"""
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        # Create test input as direct tensor
        test_input = torch.randn(1, 1, 8, 8, 8)

        outputs = wrapper.forward(test_input)

        self.assertIsInstance(outputs, dict)
        self.assertIn("output", outputs)

    def test_forward_with_multiple_outputs(self):
        """Test forward method when model returns multiple outputs"""

        # Create a model that returns multiple outputs
        class MultiOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
                self.conv2 = nn.Conv3d(1, 1, kernel_size=3, padding=1)

            def forward(self, x):
                out1 = self.conv1(x)
                out2 = self.conv2(x)
                return [out1, out2]  # Return as list

        multi_model = MultiOutputModel()
        multi_output_shapes = {
            "output1": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)},
            "output2": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)},
        }

        wrapper = ModelWrapper(
            model=multi_model,
            input_shapes=self.input_shapes,
            output_shapes=multi_output_shapes,
        )

        # Create test input
        test_input = torch.randn(1, 1, 8, 8, 8)

        outputs = wrapper.forward(test_input)

        self.assertIsInstance(outputs, dict)
        self.assertIn("output1", outputs)
        self.assertIn("output2", outputs)

    def test_forward_model_returns_single_tensor_converted_to_list(self):
        """Test that single tensor output from model is converted to list internally"""
        # Use a model that returns a single tensor (not a list)
        wrapper = ModelWrapper(
            model=self.test_model,  # Conv3d returns single tensor
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        test_input = torch.randn(1, 1, 8, 8, 8)

        outputs = wrapper.forward(test_input)

        # The single tensor should be converted to list internally and then to dict
        self.assertIsInstance(outputs, dict)
        self.assertEqual(len(outputs), 1)
        self.assertIn("output", outputs)

    def test_inheritance_from_nn_module(self):
        """Test that ModelWrapper properly inherits from nn.Module"""
        wrapper = ModelWrapper(
            model=self.test_model,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
        )

        self.assertIsInstance(wrapper, nn.Module)

        # Should have the wrapped model as a submodule
        modules = list(wrapper.modules())
        self.assertIn(self.test_model, modules)


if __name__ == "__main__":
    unittest.main()
