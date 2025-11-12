# Node Development Instructions

## Overview
This directory contains the core building blocks of the LeibNetz library. Each node represents a specific neural network operation that can be composed together to create complete models.

## Node Development Guidelines

### Base Node Interface
All nodes must inherit from the `Node` base class and implement these critical methods:

1. **`forward(self, *args)`**: Implements the actual computation
2. **`get_output_from_input_shape(self, input_shape)`**: Calculates output tensor shape from input shape
3. **`get_input_from_output_shape(self, output_shape)`**: Calculates required input shape for desired output shape

### Shape Calculation Requirements
- Shape calculations are crucial for network composition and memory planning
- Always handle batch dimensions correctly (typically first dimension)
- Consider cropping effects in spatial dimensions
- Return shapes as tuples or lists of integers

### Cropping Operations
Many nodes support cropping operations for handling spatial dimension mismatches:
- Implement `crop()` method for spatial cropping
- Support `crop_to_factor()` for ensuring spatial dimensions are multiples of a factor
- Handle both symmetric and asymmetric cropping as needed

### Testing Requirements
Every new node must have comprehensive tests covering:
- Forward pass with various input shapes
- Shape calculation methods (`get_input_from_output_shape`, `get_output_from_input_shape`)
- Cropping operations
- Edge cases and error conditions
- GPU compatibility if applicable

### Existing Node Types

#### ConvPassNode (`conv_pass_node.py`)
- Applies convolution operations with optional activation
- Handles same/valid padding modes
- Supports spatial cropping operations

#### ResampleNode (`resample_node.py`)
- Handles upsampling and downsampling operations
- Supports various interpolation modes
- Manages scale factor calculations

#### ConvResampleNode (`conv_resample_node.py`)
- Combines convolution with resampling
- Manages interaction between convolution parameters and resampling

#### AdditiveAttentionGateNode (`additive_attention_gate_node.py`)
- Implements attention mechanisms
- Handles gating operations for feature selection
- Manages multiple input streams

#### WrapperNode (`wrapper_node.py`)
- Wraps existing PyTorch modules as nodes
- Provides shape calculation for arbitrary modules
- Handles complex module compositions

### Common Patterns

#### Initialization
```python
def __init__(self, ...):
    super().__init__()
    # Store parameters
    self.param = param
    # Initialize PyTorch modules
    self.conv = nn.Conv2d(...)
```

#### Forward Pass
```python
def forward(self, x):
    # Apply operations
    output = self.conv(x)
    # Handle cropping if needed
    output = self.crop(output, target_shape)
    return output
```

#### Shape Calculations
```python
def get_output_from_input_shape(self, input_shape):
    # Calculate effects of operations on shape
    output_shape = list(input_shape)
    # Modify spatial dimensions based on operations
    output_shape[2] = input_shape[2] // self.stride
    return tuple(output_shape)
```

### Performance Considerations
- Avoid creating new tensors unnecessarily in forward pass
- Use in-place operations where possible
- Consider memory layout for optimal performance
- Profile GPU operations for performance bottlenecks

### Error Handling
- Validate input shapes in forward pass
- Provide clear error messages for invalid configurations
- Handle edge cases gracefully (e.g., zero-sized tensors)

### Documentation Standards
- Use Google-style docstrings
- Document all parameters with types
- Include usage examples for complex nodes
- Document shape transformation behavior clearly

## Node Composition
Nodes are designed to be composed together using the LeibNet class. When designing new nodes:
- Ensure compatibility with existing nodes
- Consider how your node affects the overall network topology
- Design for reusability across different network architectures
