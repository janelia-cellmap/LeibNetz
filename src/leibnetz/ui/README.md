# LeibNetz Network Builder UI

A browser-based visual interface for constructing LeibNetz neural networks using drag-and-drop.

## Features

- **Visual Node Palette**: Drag and drop nodes onto the canvas
- **Interactive Canvas**: Pan, zoom, and connect nodes visually
- **Node Properties Editor**: Edit node parameters in real-time
- **Connection Management**: Visual connections between node outputs and inputs
- **Code Generation**: Generate ready-to-use Python code from your visual network
- **Import/Export**: Save and load network configurations as JSON
- **No External Dependencies**: Pure HTML/CSS/JavaScript implementation

## Available Nodes

- **ConvPassNode**: Convolutional layers with activation functions
- **ResampleNode**: Upsampling/downsampling operations
- **ConvResampleNode**: Combined convolution and resampling
- **AdditiveAttentionGateNode**: Attention mechanisms for feature gating
- **WrapperNode**: Wrap existing PyTorch modules as nodes

## Usage

### Starting the UI Server

From command line:

```bash
# Start with default settings (port 8080)
leibnetz-ui

# Specify a custom port
leibnetz-ui --port 9000

# Don't open browser automatically
leibnetz-ui --no-browser
```

From Python:

```python
from leibnetz import serve_ui

# Start with default settings
serve_ui()

# Customize port and browser behavior
serve_ui(port=9000, open_browser=False)
```

### Building Networks

1. **Add Nodes**: Drag node types from the left palette onto the canvas
2. **Connect Nodes**: Click on an output port (green) and drag to an input port (red)
3. **Edit Properties**: Click on a node to view and edit its properties in the right panel
4. **Generate Code**: Click the "Generate Code" button to create Python code
5. **Export/Import**: Save your network as JSON for later use

### Keyboard Shortcuts

- **Delete**: Remove selected node
- **Escape**: Deselect node or cancel connection

## Example Workflow

1. Start the UI server:
   ```bash
   leibnetz-ui
   ```

2. Build your network visually by dragging nodes and connecting them

3. Generate Python code and copy it to your project:
   ```python
   # Generated LeibNetz Network
   from leibnetz import LeibNet
   from leibnetz.nodes import ConvPassNode, ResampleNode
   
   # Define nodes
   node1 = ConvPassNode(
       input_keys=["input"],
       output_keys=["output"],
       input_nc=1,
       output_nc=32,
       kernel_sizes=[3, 3],
       activation="ReLU",
   )
   
   # ... more nodes ...
   
   # Create model
   model = LeibNet(nodes, outputs)
   ```

## Architecture

The UI consists of three main components:

- **network_builder.html**: Main UI structure and styling
- **network_builder.js**: Interactive canvas and network logic
- **server.py**: Simple HTTP server to serve the UI files

## Development

The UI is built with vanilla JavaScript and requires no build step. To modify:

1. Edit the HTML/JavaScript files in `src/leibnetz/ui/`
2. Refresh your browser to see changes
3. No compilation or bundling required

## Limitations

- Network validation is basic (doesn't check shape compatibility)
- Connections are simplified (all connections go through input/output keys)
- Generated code requires manual adjustment of the outputs dictionary
- No undo/redo functionality (yet)

## Future Enhancements

- Real-time shape propagation and validation
- More sophisticated layout algorithms
- Zoom and pan controls
- Node grouping and subgraphs
- Pre-built network templates
- Undo/redo support
