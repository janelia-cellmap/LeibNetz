// LeibNetz Network Builder - JavaScript Application
// This file contains all the logic for the visual network builder

class NetworkBuilder {
    constructor() {
        this.canvas = document.getElementById('network-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.nodes = [];
        this.connections = [];
        this.selectedNode = null;
        this.connectingFrom = null;
        this.draggedNode = null;
        this.panOffset = { x: 0, y: 0 };
        this.isPanning = false;
        this.panStart = { x: 0, y: 0 };
        this.scale = 1;
        this.nodeIdCounter = 1;

        this.nodeDefinitions = {
            ConvPassNode: {
                color: '#3498db',
                inputs: ['input'],
                outputs: ['output'],
                properties: {
                    input_nc: { type: 'number', default: 1, label: 'Input Channels', required: true },
                    output_nc: { type: 'number', default: 32, label: 'Output Channels', required: true },
                    kernel_sizes: { type: 'text', default: '[3, 3]', label: 'Kernel Sizes', help: 'List format: [3, 3]' },
                    activation: { type: 'select', default: 'ReLU', options: ['ReLU', 'LeakyReLU', 'ELU', 'GELU', 'None'], label: 'Activation' },
                    padding: { type: 'select', default: 'valid', options: ['valid', 'same'], label: 'Padding' },
                    residual: { type: 'checkbox', default: false, label: 'Residual Connection' }
                }
            },
            ResampleNode: {
                color: '#e74c3c',
                inputs: ['input'],
                outputs: ['output'],
                properties: {
                    scale_factor: { type: 'text', default: '(1, 1, 1)', label: 'Scale Factor', help: 'Tuple format: (2, 2, 2) for upsample' }
                }
            },
            ConvResampleNode: {
                color: '#9b59b6',
                inputs: ['input'],
                outputs: ['output'],
                properties: {
                    input_nc: { type: 'number', default: 1, label: 'Input Channels', required: true },
                    output_nc: { type: 'number', default: 1, label: 'Output Channels', required: true },
                    scale_factor: { type: 'text', default: '(1, 1, 1)', label: 'Scale Factor' },
                    kernel_sizes: { type: 'text', default: '[3, 3]', label: 'Kernel Sizes', help: 'List format: [3, 3]' }
                }
            },
            AdditiveAttentionGateNode: {
                color: '#f39c12',
                inputs: ['input', 'gating'],
                outputs: ['output'],
                properties: {
                    input_nc: { type: 'number', default: 64, label: 'Input Channels', required: true },
                    gating_nc: { type: 'number', default: 64, label: 'Gating Channels', required: true },
                    output_nc: { type: 'number', default: 64, label: 'Output Channels', required: true },
                    ndims: { type: 'number', default: 3, label: 'Dimensions' }
                }
            },
            WrapperNode: {
                color: '#16a085',
                inputs: ['input'],
                outputs: ['output'],
                properties: {
                    model_description: { type: 'text', default: 'nn.Identity()', label: 'PyTorch Module', help: 'e.g., nn.Identity(), nn.BatchNorm2d(64)' }
                }
            }
        };

        this.init();
    }

    // HTML escape function to prevent XSS
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    init() {
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        this.setupEventListeners();
        this.render();
    }

    resizeCanvas() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.render();
    }

    setupEventListeners() {
        // Drag and drop from palette
        document.querySelectorAll('.node-item').forEach(item => {
            item.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('nodeType', e.target.dataset.nodeType);
            });
        });

        this.canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        this.canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            const nodeType = e.dataTransfer.getData('nodeType');
            if (nodeType) {
                const rect = this.canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left - this.panOffset.x) / this.scale;
                const y = (e.clientY - rect.top - this.panOffset.y) / this.scale;
                this.addNode(nodeType, x, y);
            }
        });

        // Canvas interactions
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('dblclick', (e) => this.handleDoubleClick(e));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Delete' && this.selectedNode) {
                this.deleteNode(this.selectedNode);
            }
            if (e.key === 'Escape') {
                this.selectedNode = null;
                this.connectingFrom = null;
                this.render();
            }
        });
    }

    handleMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.panOffset.x) / this.scale;
        const y = (e.clientY - rect.top - this.panOffset.y) / this.scale;

        // Check if clicking on a node
        const clickedNode = this.getNodeAtPosition(x, y);

        if (clickedNode) {
            // Check if clicking on output port to start connection
            const outputPort = this.getOutputPortAtPosition(clickedNode, x, y);
            if (outputPort !== null) {
                this.connectingFrom = { node: clickedNode, port: outputPort };
                return;
            }

            // Otherwise, start dragging the node
            this.selectedNode = clickedNode;
            this.draggedNode = clickedNode;
            this.dragOffset = {
                x: x - clickedNode.x,
                y: y - clickedNode.y
            };
            this.showNodeProperties(clickedNode);
        } else {
            // Start panning
            this.selectedNode = null;
            this.isPanning = true;
            this.panStart = { x: e.clientX - this.panOffset.x, y: e.clientY - this.panOffset.y };
            this.canvas.classList.add('dragging');
            this.showNodeProperties(null);
        }

        this.render();
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.panOffset.x) / this.scale;
        const y = (e.clientY - rect.top - this.panOffset.y) / this.scale;

        if (this.draggedNode) {
            this.draggedNode.x = x - this.dragOffset.x;
            this.draggedNode.y = y - this.dragOffset.y;
            this.render();
        } else if (this.isPanning) {
            this.panOffset.x = e.clientX - this.panStart.x;
            this.panOffset.y = e.clientY - this.panStart.y;
            this.render();
        } else if (this.connectingFrom) {
            this.tempConnectionEnd = { x, y };
            this.render();
        }
    }

    handleMouseUp(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.panOffset.x) / this.scale;
        const y = (e.clientY - rect.top - this.panOffset.y) / this.scale;

        if (this.connectingFrom) {
            const targetNode = this.getNodeAtPosition(x, y);
            if (targetNode && targetNode !== this.connectingFrom.node) {
                const inputPort = this.getInputPortAtPosition(targetNode, x, y);
                if (inputPort !== null) {
                    this.addConnection(
                        this.connectingFrom.node.id,
                        this.connectingFrom.port,
                        targetNode.id,
                        inputPort
                    );
                }
            }
            this.connectingFrom = null;
            this.tempConnectionEnd = null;
        }

        this.draggedNode = null;
        this.isPanning = false;
        this.canvas.classList.remove('dragging');
        this.render();
    }

    handleDoubleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.panOffset.x) / this.scale;
        const y = (e.clientY - rect.top - this.panOffset.y) / this.scale;

        const clickedNode = this.getNodeAtPosition(x, y);
        if (clickedNode) {
            const newName = prompt('Enter node name:', clickedNode.name);
            if (newName) {
                clickedNode.name = newName;
                this.render();
            }
        }
    }

    addNode(type, x, y) {
        const def = this.nodeDefinitions[type];
        if (!def) return;

        const node = {
            id: this.nodeIdCounter++,
            type: type,
            name: `${type}_${this.nodeIdCounter - 1}`,
            x: x,
            y: y,
            width: 180,
            height: 80 + (def.inputs.length + def.outputs.length) * 15,
            color: def.color,
            properties: {}
        };

        // Initialize properties with defaults
        for (const [key, prop] of Object.entries(def.properties)) {
            node.properties[key] = prop.default;
        }

        this.nodes.push(node);
        this.selectedNode = node;
        this.showNodeProperties(node);
        this.updateStatus();
        this.render();
    }

    deleteNode(node) {
        // Remove connections associated with this node
        this.connections = this.connections.filter(conn =>
            conn.fromNode !== node.id && conn.toNode !== node.id
        );

        // Remove the node
        this.nodes = this.nodes.filter(n => n !== node);

        if (this.selectedNode === node) {
            this.selectedNode = null;
            this.showNodeProperties(null);
        }

        this.updateStatus();
        this.render();
    }

    addConnection(fromNodeId, fromPort, toNodeId, toPort) {
        // Check if connection already exists
        const exists = this.connections.some(conn =>
            conn.fromNode === fromNodeId &&
            conn.fromPort === fromPort &&
            conn.toNode === toNodeId &&
            conn.toPort === toPort
        );

        if (!exists) {
            this.connections.push({
                fromNode: fromNodeId,
                fromPort: fromPort,
                toNode: toNodeId,
                toPort: toPort
            });
            this.updateStatus();
            this.render();
        }
    }

    getNodeAtPosition(x, y) {
        for (let i = this.nodes.length - 1; i >= 0; i--) {
            const node = this.nodes[i];
            if (x >= node.x && x <= node.x + node.width &&
                y >= node.y && y <= node.y + node.height) {
                return node;
            }
        }
        return null;
    }

    getOutputPortAtPosition(node, x, y) {
        const def = this.nodeDefinitions[node.type];
        const portY = node.y + 50;
        const portSpacing = node.width / (def.outputs.length + 1);

        for (let i = 0; i < def.outputs.length; i++) {
            const portX = node.x + portSpacing * (i + 1);
            const dist = Math.sqrt(Math.pow(x - portX, 2) + Math.pow(y - portY, 2));
            if (dist < 8) {
                return i;
            }
        }
        return null;
    }

    getInputPortAtPosition(node, x, y) {
        const def = this.nodeDefinitions[node.type];
        const portY = node.y + node.height - 50;
        const portSpacing = node.width / (def.inputs.length + 1);

        for (let i = 0; i < def.inputs.length; i++) {
            const portX = node.x + portSpacing * (i + 1);
            const dist = Math.sqrt(Math.pow(x - portX, 2) + Math.pow(y - portY, 2));
            if (dist < 8) {
                return i;
            }
        }
        return null;
    }

    showNodeProperties(node) {
        const content = document.getElementById('properties-content');

        if (!node) {
            content.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📋</div>
                    <div class="empty-state-title">No Node Selected</div>
                    <div class="empty-state-text">
                        Select a node from the canvas to view and edit its properties.
                    </div>
                </div>
            `;
            return;
        }

        const def = this.nodeDefinitions[node.type];
        let html = `
            <div style="margin-bottom: 20px;">
                <h3 style="margin-bottom: 10px; color: ${this.escapeHtml(node.color)};">${this.escapeHtml(node.name)}</h3>
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 15px;">${this.escapeHtml(node.type)}</div>
                <button class="btn btn-danger" style="width: 100%;" onclick="app.deleteNode(app.selectedNode)">
                    Delete Node
                </button>
            </div>
        `;

        for (const [key, prop] of Object.entries(def.properties)) {
            html += '<div class="form-group">';
            html += `<label class="form-label">${this.escapeHtml(prop.label)}</label>`;

            if (prop.type === 'number') {
                html += `<input type="number" class="form-input" value="${this.escapeHtml(String(node.properties[key]))}"
                    onchange="app.updateNodeProperty('${this.escapeHtml(key)}', this.value, 'number')">`;
            } else if (prop.type === 'text') {
                html += `<input type="text" class="form-input" value="${this.escapeHtml(String(node.properties[key]))}"
                    onchange="app.updateNodeProperty('${this.escapeHtml(key)}', this.value, 'text')">`;
            } else if (prop.type === 'select') {
                html += `<select class="form-select" onchange="app.updateNodeProperty('${this.escapeHtml(key)}', this.value, 'select')">`;
                for (const option of prop.options) {
                    const selected = node.properties[key] === option ? 'selected' : '';
                    html += `<option value="${this.escapeHtml(option)}" ${selected}>${this.escapeHtml(option)}</option>`;
                }
                html += '</select>';
            } else if (prop.type === 'checkbox') {
                const checked = node.properties[key] ? 'checked' : '';
                html += `<input type="checkbox" ${checked}
                    onchange="app.updateNodeProperty('${this.escapeHtml(key)}', this.checked, 'checkbox')">`;
            }

            if (prop.help) {
                html += `<div class="help-text">${this.escapeHtml(prop.help)}</div>`;
            }
            html += '</div>';
        }

        content.innerHTML = html;
    }

    updateNodeProperty(key, value, type) {
        if (!this.selectedNode) return;

        if (type === 'number') {
            this.selectedNode.properties[key] = parseFloat(value);
        } else if (type === 'checkbox') {
            this.selectedNode.properties[key] = value;
        } else {
            this.selectedNode.properties[key] = value;
        }

        this.render();
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Apply transformations
        this.ctx.save();
        this.ctx.translate(this.panOffset.x, this.panOffset.y);
        this.ctx.scale(this.scale, this.scale);

        // Draw grid
        this.drawGrid();

        // Draw connections
        this.drawConnections();

        // Draw temporary connection
        if (this.connectingFrom && this.tempConnectionEnd) {
            const fromNode = this.nodes.find(n => n.id === this.connectingFrom.node.id);
            if (fromNode) {
                const def = this.nodeDefinitions[fromNode.type];
                const portSpacing = fromNode.width / (def.outputs.length + 1);
                const startX = fromNode.x + portSpacing * (this.connectingFrom.port + 1);
                const startY = fromNode.y + 50;

                this.ctx.strokeStyle = '#95a5a6';
                this.ctx.lineWidth = 2;
                this.ctx.setLineDash([5, 5]);
                this.ctx.beginPath();
                this.ctx.moveTo(startX, startY);
                this.ctx.lineTo(this.tempConnectionEnd.x, this.tempConnectionEnd.y);
                this.ctx.stroke();
                this.ctx.setLineDash([]);
            }
        }

        // Draw nodes
        for (const node of this.nodes) {
            this.drawNode(node);
        }

        this.ctx.restore();
    }

    drawGrid() {
        const gridSize = 20;
        this.ctx.strokeStyle = '#ecf0f1';
        this.ctx.lineWidth = 1;

        for (let x = 0; x < this.canvas.width / this.scale; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height / this.scale);
            this.ctx.stroke();
        }

        for (let y = 0; y < this.canvas.height / this.scale; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width / this.scale, y);
            this.ctx.stroke();
        }
    }

    drawNode(node) {
        const isSelected = this.selectedNode === node;
        const def = this.nodeDefinitions[node.type];

        // Draw node body
        this.ctx.fillStyle = node.color;
        this.ctx.fillRect(node.x, node.y, node.width, node.height);

        // Draw border
        this.ctx.strokeStyle = isSelected ? '#f39c12' : '#2c3e50';
        this.ctx.lineWidth = isSelected ? 3 : 1;
        this.ctx.strokeRect(node.x, node.y, node.width, node.height);

        // Draw title bar
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
        this.ctx.fillRect(node.x, node.y, node.width, 30);

        // Draw node name
        this.ctx.fillStyle = 'white';
        this.ctx.font = 'bold 12px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(node.name, node.x + node.width / 2, node.y + 20);

        // Draw output ports
        const outputPortY = node.y + 50;
        const outputPortSpacing = node.width / (def.outputs.length + 1);
        for (let i = 0; i < def.outputs.length; i++) {
            const portX = node.x + outputPortSpacing * (i + 1);
            this.ctx.fillStyle = '#27ae60';
            this.ctx.beginPath();
            this.ctx.arc(portX, outputPortY, 6, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.strokeStyle = 'white';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            // Draw port label
            this.ctx.fillStyle = 'white';
            this.ctx.font = '10px sans-serif';
            this.ctx.fillText(def.outputs[i], portX, outputPortY + 20);
        }

        // Draw input ports
        const inputPortY = node.y + node.height - 50;
        const inputPortSpacing = node.width / (def.inputs.length + 1);
        for (let i = 0; i < def.inputs.length; i++) {
            const portX = node.x + inputPortSpacing * (i + 1);
            this.ctx.fillStyle = '#e74c3c';
            this.ctx.beginPath();
            this.ctx.arc(portX, inputPortY, 6, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.strokeStyle = 'white';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            // Draw port label
            this.ctx.fillStyle = 'white';
            this.ctx.font = '10px sans-serif';
            this.ctx.fillText(def.inputs[i], portX, inputPortY - 10);
        }
    }

    drawConnections() {
        for (const conn of this.connections) {
            const fromNode = this.nodes.find(n => n.id === conn.fromNode);
            const toNode = this.nodes.find(n => n.id === conn.toNode);

            if (!fromNode || !toNode) continue;

            const fromDef = this.nodeDefinitions[fromNode.type];
            const toDef = this.nodeDefinitions[toNode.type];

            const fromPortSpacing = fromNode.width / (fromDef.outputs.length + 1);
            const toPortSpacing = toNode.width / (toDef.inputs.length + 1);

            const startX = fromNode.x + fromPortSpacing * (conn.fromPort + 1);
            const startY = fromNode.y + 50;
            const endX = toNode.x + toPortSpacing * (conn.toPort + 1);
            const endY = toNode.y + toNode.height - 50;

            // Draw bezier curve
            this.ctx.strokeStyle = '#34495e';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(startX, startY);

            const controlPoint1X = startX;
            const controlPoint1Y = startY + (endY - startY) / 2;
            const controlPoint2X = endX;
            const controlPoint2Y = endY - (endY - startY) / 2;

            this.ctx.bezierCurveTo(
                controlPoint1X, controlPoint1Y,
                controlPoint2X, controlPoint2Y,
                endX, endY
            );
            this.ctx.stroke();
        }
    }

    updateStatus() {
        document.getElementById('node-count').textContent = this.nodes.length;
        document.getElementById('connection-count').textContent = this.connections.length;
    }

    clearCanvas() {
        if (this.nodes.length === 0 || confirm('Are you sure you want to clear all nodes?')) {
            this.nodes = [];
            this.connections = [];
            this.selectedNode = null;
            this.showNodeProperties(null);
            this.updateStatus();
            this.render();
        }
    }

    newNetwork() {
        this.clearCanvas();
    }

    exportNetwork() {
        const data = {
            nodes: this.nodes.map(node => ({
                id: node.id,
                type: node.type,
                name: node.name,
                x: node.x,
                y: node.y,
                properties: node.properties
            })),
            connections: this.connections
        };

        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'leibnetz_network.json';
        a.click();
        URL.revokeObjectURL(url);
    }

    openImportModal() {
        document.getElementById('import-modal').classList.add('active');
    }

    importNetwork() {
        const json = document.getElementById('import-json').value;
        try {
            const data = JSON.parse(json);

            // Validate data structure
            if (!data.nodes || !Array.isArray(data.nodes)) {
                throw new Error('Invalid network structure');
            }

            this.nodes = data.nodes.map(node => ({
                ...node,
                width: 180,
                height: 80 + (this.nodeDefinitions[node.type].inputs.length +
                    this.nodeDefinitions[node.type].outputs.length) * 15,
                color: this.nodeDefinitions[node.type].color
            }));
            this.connections = data.connections || [];
            this.nodeIdCounter = Math.max(...this.nodes.map(n => n.id), 0) + 1;

            this.selectedNode = null;
            this.showNodeProperties(null);
            this.updateStatus();
            this.render();
            this.closeModal('import-modal');
            document.getElementById('import-json').value = '';
        } catch (error) {
            alert('Error importing network: ' + error.message);
        }
    }

    generateCode() {
        if (this.nodes.length === 0) {
            alert('Please add some nodes first!');
            return;
        }

        let code = '# Generated LeibNetz Network\n';
        code += 'from leibnetz import LeibNet\n';
        code += 'from leibnetz.nodes import (\n';

        const nodeTypes = [...new Set(this.nodes.map(n => n.type))];
        code += nodeTypes.map(t => `    ${t}`).join(',\n');
        code += '\n)\nimport torch.nn as nn\n\n';

        // Generate node definitions
        code += '# Define nodes\n';
        for (const node of this.nodes) {
            code += this.generateNodeCode(node);
        }

        code += '\n# Compose network\n';
        code += 'nodes = [\n';
        code += this.nodes.map(n => `    ${n.name}`).join(',\n');
        code += '\n]\n\n';

        code += '# Create LeibNet model\n';
        code += '# Note: You need to define appropriate outputs dictionary\n';
        code += '# outputs = {"output_key": (shape, scale)}\n';
        code += 'outputs = {"final_output": ((256, 256), (1, 1))}\n\n';
        code += 'model = LeibNet(nodes, outputs)\n';

        document.getElementById('generated-code').textContent = code;
        document.getElementById('code-modal').classList.add('active');
    }

    generateNodeCode(node) {
        const def = this.nodeDefinitions[node.type];
        let code = `${node.name} = ${node.type}(\n`;

        // Add input/output keys
        code += `    input_keys=[${def.inputs.map(i => `"${i}"`).join(', ')}],\n`;
        code += `    output_keys=[${def.outputs.map(o => `"${o}"`).join(', ')}],\n`;

        // Add properties
        for (const [key, value] of Object.entries(node.properties)) {
            const propDef = def.properties[key];
            let formattedValue = value;

            if (propDef.type === 'text') {
                // Try to parse as Python literal
                if (value.startsWith('[') || value.startsWith('(')) {
                    formattedValue = value;
                } else {
                    formattedValue = `"${value}"`;
                }
            } else if (propDef.type === 'checkbox') {
                formattedValue = value ? 'True' : 'False';
            } else if (propDef.type === 'select') {
                if (value === 'None') {
                    formattedValue = 'None';
                } else {
                    formattedValue = `"${value}"`;
                }
            }

            code += `    ${key}=${formattedValue},\n`;
        }

        code += ')\n\n';
        return code;
    }

    copyCode() {
        const code = document.getElementById('generated-code').textContent;
        navigator.clipboard.writeText(code).then(() => {
            alert('Code copied to clipboard!');
        }).catch(err => {
            alert('Failed to copy code: ' + err);
        });
    }

    closeModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }
}

// Initialize the application
const app = new NetworkBuilder();
