# Network Architecture Instructions

## Overview
This directory contains pre-built network architectures that demonstrate how to compose nodes from the `nodes/` directory into complete neural networks.

## Architecture Development Guidelines

### Network Composition Patterns
Each network architecture should:
1. Use the LeibNet class as the base for composition
2. Compose nodes in a logical, reusable manner
3. Provide clear configuration options
4. Handle different input/output requirements

### Existing Architectures

#### U-Net (`unet.py`)
- Classic encoder-decoder architecture with skip connections
- Demonstrates hierarchical feature extraction
- Shows proper handling of spatial dimension matching

#### ScaleNet (`scalenet.py`)
- Multi-scale processing architecture
- Demonstrates parallel processing paths
- Shows feature fusion techniques

#### AttentiveScaleNet (`attentive_scalenet.py`)
- Combines multi-scale processing with attention mechanisms
- Demonstrates integration of attention nodes
- Shows advanced feature selection patterns

### Network Development Best Practices

#### Configuration Management
```python
def build_network(input_channels, output_channels, **config):
    # Use configuration dictionaries for flexibility
    conv_config = config.get('conv_config', {})
    attention_config = config.get('attention_config', {})

    # Build network with configurable components
    return LeibNet([...])
```

#### Skip Connections and Feature Fusion
- Use consistent naming for skip connection handling
- Implement proper spatial alignment for feature fusion
- Consider feature channel alignment

#### Multi-Scale Processing
- Design for different input resolutions
- Handle scale-specific feature extraction
- Implement proper feature aggregation

### Testing Network Architectures
Each network should have tests that verify:
- End-to-end forward pass functionality
- Correct output shapes for various input sizes
- Proper gradient flow (if training components are included)
- Memory efficiency for large inputs

### Documentation Requirements
- Document the architectural design rationale
- Provide usage examples with typical configurations
- Document expected input/output formats
- Include performance characteristics and memory requirements

### Common Network Patterns

#### Encoder-Decoder
```python
# Encoder path
encoder_nodes = [
    ConvPassNode(...),
    ResampleNode(scale_factor=0.5),  # Downsample
    # ... more layers
]

# Decoder path with skip connections
decoder_nodes = [
    ResampleNode(scale_factor=2.0),  # Upsample
    # Feature fusion node for skip connections
    ConvPassNode(...),
]
```

#### Multi-Path Processing
```python
# Parallel processing paths
path1 = LeibNet([...])
path2 = LeibNet([...])

# Feature fusion
fusion_node = FusionNode()
```

### Integration Guidelines
- New architectures should integrate well with existing ModelWrapper functionality
- Consider compatibility with training scripts in `examples/`
- Design for extensibility and modification
- Provide sensible default configurations
