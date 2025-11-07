# LeibNetz - GitHub Copilot Instructions

## Project Overview

LeibNetz is a lightweight and modular library for rapidly developing and constructing PyTorch models for deep learning, specifically focused on image segmentation and convolutional neural networks. The library provides building blocks for creating various neural network architectures including U-Net, ScaleNet, and attention-based models.

## Key Architecture

- **Core Components**: Located in `src/leibnetz/`
  - `leibnet.py`: Main LeibNet class for model composition
  - `model_wrapper.py`: ModelWrapper for model management
  - `nets/`: Pre-built network architectures (U-Net, ScaleNet, AttentiveScaleNet)
  - `nodes/`: Modular building blocks for neural networks

- **Modular Node System**: The library uses a node-based architecture where each node represents a specific operation or layer that can be composed together
  - All nodes inherit from base `Node` class in `nodes/node.py`
  - Nodes handle shape calculations, cropping, and forward passes
  - Key node types: ConvPassNode, ResampleNode, ConvResampleNode, AdditiveAttentionGateNode

## Development Guidelines

### Code Style & Standards
- Use **Black** for Python code formatting (configured to format `src/` directory)
- Follow **MyPy** type checking standards
- Maintain **pytest** test coverage for all new functionality
- Follow existing docstring patterns using Google-style docstrings

### Building & Testing
```bash
# Install in development mode
pip install -e .

# Install testing dependencies  
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov --cov-report=term-missing

# Run linting
black src/
mypy src/
```

### Testing Requirements
- All new features must have corresponding tests in `tests/` directory
- Test files should follow the pattern `*_test.py`
- Use the existing test structure as a template (see `wrapper_node_test.py` for comprehensive examples)
- Tests should cover:
  - Forward pass functionality
  - Shape calculation methods (`get_input_from_output_shape`, `get_output_from_input_shape`)
  - Cropping operations
  - Error conditions and edge cases

### Architecture Patterns
- **Node Implementation**: When creating new nodes, inherit from `Node` base class and implement required methods:
  - `forward()`: Define the forward pass
  - `get_output_from_input_shape()`: Calculate output shape from input
  - `get_input_from_output_shape()`: Calculate required input shape from desired output
  - Cropping methods as needed

- **Network Composition**: Use the LeibNet class to compose nodes into complete networks
- **Model Wrapping**: Use ModelWrapper for model management and utilities

### Dependencies
- **Core**: torch, numpy
- **Development**: pytest, pytest-cov, black, mypy
- Python 3.10+ supported

### File Organization
- Source code: `src/leibnetz/`
- Tests: `tests/` (mirrors source structure)
- Examples: `examples/` (training scripts and usage examples)
- Configuration: `setup.cfg`, `pyproject.toml`

## Contribution Workflow

1. **Code Changes**: Make minimal, focused changes
2. **Testing**: Ensure all tests pass with `pytest tests/`
3. **Formatting**: Run `black src/` before committing
4. **Type Checking**: Verify with `mypy src/`
5. **Documentation**: Update docstrings for new public APIs

## Common Patterns

### Creating a New Node
```python
from .node import Node

class MyCustomNode(Node):
    def __init__(self, ...):
        super().__init__()
        # Initialize your layers/operations
    
    def forward(self, x):
        # Implement forward pass
        return output
    
    def get_output_from_input_shape(self, input_shape):
        # Calculate output shape
        return output_shape
    
    def get_input_from_output_shape(self, output_shape):
        # Calculate required input shape  
        return input_shape
```

### Network Composition
```python
from leibnetz import LeibNet
from leibnetz.nodes import ConvPassNode, ResampleNode

# Create network by composing nodes
net = LeibNet([
    ConvPassNode(...),
    ResampleNode(...),
    # ... more nodes
])
```

## Important Notes

- The library emphasizes modularity and reusability
- Shape calculations are critical for proper network composition
- All spatial operations should properly handle cropping
- Maintain backward compatibility when modifying existing APIs
- Performance considerations: avoid unnecessary memory allocations in forward passes

## CI/CD

The repository uses GitHub Actions for:
- **Testing**: Multi-platform testing (Ubuntu, Windows, macOS) with Python 3.10, 3.11
- **Formatting**: Automatic Black formatting with PR creation
- **Type Checking**: MyPy static analysis
- **Coverage**: Codecov integration for test coverage reporting

All PRs must pass these checks before merging.