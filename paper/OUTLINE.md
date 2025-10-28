## 1. Abstract
ScaleNet introduces a chained multi-FOV convolutional architecture enabling high-resolution predictions with large effective receptive fields. Lower-resolution UNet outputs feed into higher-resolution UNet bottlenecks to integrate contextual information efficiently. Validated on FIBSEM data from the CellMap Segmentation Challenge, ScaleNet improves segmentation accuracy compared to single UNet baselines. Additionally, we present *LeibNetz*, a composable, Node-based graph framework for transparent and modular CNN construction in PyTorch. Both contributions aim to enhance reproducibility and scalability in bioimaging and beyond.

## 2. Introduction
- Motivation: large 3D microscopy volumes require both high spatial resolution and wide context, challenging for standard UNets due to FOV–memory tradeoffs.
- Problem: existing multi-scale or patch-based methods introduce artifacts or obscure input–output dimensional consistency.
- Contributions:
  1. **ScaleNet**: chained UNet architecture capturing multi-scale context efficiently.
  2. **LeibNetz**: composable framework ensuring explicit shape propagation and modular design.
  3. Open-source PyTorch implementation validated on FIBSEM segmentation tasks.

## 3. Related Work
### 3.1 Multi-Scale and Multi-Resolution Segmentation
- Review of feature pyramid and multi-resolution architectures (FPN, HRNet, Laplacian UNets).  
- Limitations in maintaining valid output sizes and preventing tiling artifacts.

### 3.2 Contextual Learning and Field-of-View Extensions
- Overview of receptive field scaling, cascaded architectures, and their computational tradeoffs.  
- ScaleNet positioned as a memory-efficient chaining strategy.

### 3.3 Frameworks for Composable Network Design
- Review of graph-based APIs (Keras, PyTorch Lightning).  
- LeibNetz solves practical issues of valid convolution tracking and shape propagation.

## 4. Methods
### 4.1 Overview of ScaleNet
- Hierarchical chain of UNets operating at decreasing FOVs and increasing resolutions.  
- Lower-res UNet output fed into bottleneck of higher-res UNet; optional skip/gate mechanisms.

### 4.2 Theoretical Justification
- Effective receptive field increase vs. memory complexity.  
- Chaining mitigates boundary truncation from valid convolutions.

### 4.3 Handling Valid Convolutions
- Explanation of artifact-free inference via valid convolutions.  
- Formulas for size propagation and integration into Node abstraction.

### 4.4 The LeibNetz Framework
- Core concept: *Nodes* define input/output sizes and resolution scales; models composed as DAGs.  
- Features: automatic shape inference, modular combination of heterogeneous nodes, and transparency.  
- Example: composing two UNet Nodes into a chained ScaleNet configuration.

## 5. Experiments
### 5.1 Dataset and Evaluation
- CellMap Segmentation Challenge (FIBSEM multi-class segmentation).  
- Details: voxel size, normalization, augmentation, and train/val/test splits.

### 5.2 Experimental Setup
- Baselines: single-UNet and multi-scale UNet.  
- Metrics: Dice, mean IoU, class-wise recall and precision.

### 5.3 Results
- Quantitative comparison and qualitative examples (context-aware segmentations).  
- Ablations: chaining depth, memory scaling, inference tradeoffs.

### 5.4 Discussion
- Benefits: larger FOV without over-smoothing or memory explosion.  
- Limitations: computational overhead, FOV tuning sensitivity.

## 6. The LeibNetz Framework in Practice
- Reproducibility: deterministic input/output shape propagation.  
- Extensibility: plug-in architecture for CNNs, transformers, or VAEs.  
- Open-source release on GitHub with community contribution guidelines.

## 7. Discussion & Future Work
- Broader applications: geospatial imagery, histopathology, cryo-EM, self-supervised extensions.  
- Future directions: learned scale fusion, distributed 3D training, hybrid CNN–Transformer designs.

## 8. Conclusion
ScaleNet enables efficient multi-FOV context aggregation without excessive memory cost. LeibNetz provides a reusable foundation for transparent, composable model design. Together, they advance reproducible, interpretable architectures for bioimaging and other high-resolution domains.

## Appendices
- **A:** Receptive field and valid-convolution derivations.  
- **B:** Training configurations and scripts.  
- **C:** CellMap preprocessing details.  
- **D:** Pseudocode for Node composition and graph validation.

## References
Key references: UNet, V-Net, FPN, HRNet, DeepLab cascades, composable ML frameworks, CellMap, EMPIAR, ISBI datasets.