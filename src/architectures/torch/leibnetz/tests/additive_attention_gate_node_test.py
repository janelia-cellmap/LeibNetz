import unittest
import torch
import numpy as np


class TestAdditiveAttentionGateNode(unittest.TestCase):
    def setUp(self):
        self.node = AdditiveAttentionGateNode(
            output_keys=["output"],
            gating_key="gating",
            input_key="input",
            input_nc=3,
            gating_nc=3,
            output_nc=3,
            identifier="test",
            ndims=3,
        )
        self.inputs = {
            "input": torch.randn(1, 3, 10, 10, 10),
            "gating": torch.randn(1, 3, 10, 10, 10),
        }

    def test_forward(self):
        output = self.node.forward(self.inputs)
        self.assertIsInstance(output, dict)
        self.assertIn("output", output)
        self.assertEqual(output["output"].shape, torch.Size([1, 10, 10, 10]))

    def test_get_input_from_output_shape(self):
        output_shape = np.array([10, 10, 10])
        input_shape = self.node.get_input_from_output_shape(output_shape)
        self.assertIsInstance(input_shape, dict)
        self.assertIn("input", input_shape)
        self.assertIn("gating", input_shape)

    def test_get_output_from_input_shape(self):
        input_shape = np.array([10, 10, 10])
        output_shape = self.node.get_output_from_input_shape(input_shape)
        self.assertIsInstance(output_shape, dict)
        self.assertIn("output", output_shape)

    def test_factor_crop(self):
        input_shape = np.array([10, 10, 10])
        crop = self.node.factor_crop(input_shape)
        self.assertIsInstance(crop, np.ndarray)

    def test_crop_to_factor(self):
        x = torch.randn(1, 3, 10, 10, 10)
        cropped = self.node.crop_to_factor(x)
        self.assertIsInstance(cropped, torch.Tensor)

    def test_crop(self):
        x = torch.randn(1, 3, 10, 10, 10)
        shape = np.array([5, 5, 5])
        cropped = self.node.crop(x, shape)
        self.assertIsInstance(cropped, torch.Tensor)
        self.assertEqual(cropped.shape, torch.Size([1, 3, 5, 5, 5]))


if __name__ == "__main__":
    unittest.main()
