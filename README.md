# LeibNetz

![PyPI - License](https://img.shields.io/pypi/l/LeibNetz)
[![CI/CD Pipeline](https://github.com/janelia-cellmap/LeibNetz/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/janelia-cellmap/LeibNetz/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/github/janelia-cellmap/LeibNetz/graph/badge.svg?token=PPT4ZNZZCJ)](https://codecov.io/github/janelia-cellmap/LeibNetz)
[![PyPI - Version](https://img.shields.io/pypi/v/leibnetz)](https://pypi.org/project/leibnetz/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/leibnetz)

A lightweight and modular library for rapidly developing and constructing PyTorch models for deep learning, specifically focused on image segmentation and convolutional neural networks.

## Features

- 🧱 **Modular Architecture**: Build networks using composable node-based components
- 🎨 **Visual Network Builder**: Browser-based UI for graphically constructing networks
- 🔧 **Pre-built Networks**: Ready-to-use implementations of U-Net, ScaleNet, and AttentiveScaleNet
- 📐 **Automatic Shape Propagation**: Smart shape calculation and management throughout the network
- 🎯 **Specialized for Segmentation**: Optimized for image segmentation tasks
- 🔬 **Biologically-Inspired Learning**: Local learning rules including Hebbian, Oja's, and Krotov's rules
- ⚡ **PyTorch Integration**: Seamless integration with the PyTorch ecosystem

## Installation

### From PyPI (Recommended)

```bash
pip install leibnetz
```

### From Source

```bash
git clone https://github.com/janelia-cellmap/LeibNetz.git
cd LeibNetz
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/janelia-cellmap/LeibNetz.git
cd LeibNetz
pip install -e ".[dev]"
```

## Quick Start

### Visual Network Builder (New!)

Build networks graphically using the browser-based UI:

```bash
# Launch the visual network builder
leibnetz-ui
```

This opens a browser-based interface where you can:
- Drag and drop nodes onto a canvas
- Connect nodes visually
- Edit node properties in real-time
- Generate Python code automatically
- Export/import network configurations as JSON

See [UI documentation](src/leibnetz/ui/README.md) for more details.

### Building a Simple U-Net

```python
import torch
from leibnetz import build_unet

# Create a U-Net for 4-class segmentation
model = build_unet(
    input_nc=1,        # Single input channel (e.g., grayscale)
    output_nc=4,       # 4 output classes
    base_nc=64,        # Base number of features
    max_nc=512,        # Maximum number of features
    num_levels=4       # Number of resolution levels
)

# Forward pass
x = torch.randn(1, 1, 256, 256)  # Batch, channels, height, width
output = model(x)
print(f"Output shape: {output.shape}")  # [1, 4, 256, 256]
```

### Using the Modular Node System

```python
from leibnetz import LeibNet
from leibnetz.nodes import ConvPassNode, ResampleNode

# Build a custom network using nodes
nodes = [
    ConvPassNode(input_nc=1, output_nc=32, kernel_size=3),
    ResampleNode(scale_factor=0.5, mode="area"),  # Downsample
    ConvPassNode(input_nc=32, output_nc=64, kernel_size=3),
    ResampleNode(scale_factor=2.0, mode="nearest"),  # Upsample
    ConvPassNode(input_nc=64, output_nc=4, kernel_size=1)  # Final classification
]

model = LeibNet(nodes)

# Use the model
x = torch.randn(1, 1, 128, 128)
output = model(x)
```

### ScaleNet for Multi-Scale Processing

```python
from leibnetz import build_scalenet

# Create a ScaleNet with multiple processing scales
model = build_scalenet(
    input_nc=1,
    output_nc=4,
    base_nc=32,
    subnet_dict_list=[
        {"input_shape": (64, 64), "num_levels": 3},
        {"input_shape": (128, 128), "num_levels": 4},
        {"input_shape": (256, 256), "num_levels": 4}
    ]
)

# Process different scales
outputs = model(x)
```

## Core Components

### Networks (`leibnetz.nets`)

- **U-Net**: Classic encoder-decoder architecture for segmentation
- **ScaleNet**: Multi-scale processing network for handling different resolutions
- **AttentiveScaleNet**: ScaleNet enhanced with attention mechanisms

### Nodes (`leibnetz.nodes`)

Building blocks for custom architectures:

- **ConvPassNode**: Convolutional layers with optional normalization and activation
- **ResampleNode**: Upsampling/downsampling operations
- **ConvResampleNode**: Combined convolution and resampling
- **AdditiveAttentionGateNode**: Attention mechanism for feature gating
- **WrapperNode**: Wraps existing PyTorch modules as nodes

### Model Management

- **LeibNet**: Main class for composing nodes into networks
- **ModelWrapper**: Utilities for model management and deployment

### Local Learning Rules

Biologically-inspired learning algorithms:

```python
from leibnetz.local_learning import HebbsRule, OjasRule, KrotovsRule

# Apply Hebbian learning to a model
convert_to_bio(model, rule=HebbsRule())
```

## Advanced Usage

### Custom Node Creation

```python
from leibnetz.nodes import Node

class CustomProcessingNode(Node):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = torch.nn.BatchNorm2d(channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

    def get_output_from_input_shape(self, input_shape):
        # Shape preserved through convolution
        return input_shape

    def get_input_from_output_shape(self, output_shape):
        # Inverse shape calculation
        return output_shape
```

### Network Visualization

```python
import matplotlib.pyplot as plt

# Visualize network structure
model = build_unet(input_nc=1, output_nc=4)
model.visualize_network()
plt.show()
```

## Examples

Complete training examples are available in the [`examples/`](examples/) directory:

- [`train_scalenet.py`](examples/train_scalenet.py): Multi-class segmentation with ScaleNet

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov --cov-report=term-missing

# Run specific test categories
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m "unit"      # Run only unit tests
```

## Development

### Code Quality

The project uses several tools to maintain code quality:

```bash
# Format code
black src/

# Type checking
mypy src/

# Run linting
flake8 src/

# Sort imports
isort src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Format your code (`black src/`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Requirements

- Python 3.10+
- PyTorch 1.9+
- NumPy
- NetworkX

## License

This project is licensed under the BSD-3 License - see the [LICENSE](https://github.com/janelia-cellmap/LeibNetz/blob/main/LICENSE) file for details.

## Citation

If you use LeibNetz in your research, please cite:

```bibtex
@software{leibnetz2024,
  author = {Jeff Rhoades and Larissa Heinrich},
  title = {LeibNetz: A Lightweight and Modular Library for Deep Learning},
  url = {https://github.com/janelia-cellmap/LeibNetz},
  year = {2024}
}
```

## Acknowledgments

- Developed at [Janelia Research Campus](https://www.janelia.org/)
- Part of the CellMap project for large-scale cellular imaging

## Support

- 📖 [Documentation](https://github.com/janelia-cellmap/LeibNetz)
- 🐛 [Issue Tracker](https://github.com/janelia-cellmap/LeibNetz/issues)
- 💬 [Discussions](https://github.com/janelia-cellmap/LeibNetz/discussions)
