# GSoC 2026 - ML4SCI DeepLense: Specific Test IV - Neural Operators

## Overview

This repository contains my solution for the **Specific Test IV: Neural Operators** task for the Google Summer of Code 2026 application with ML4SCI's DeepLense project.

**Task:** Build a model for classifying gravitational lensing images into three classes using a neural operator architecture (FNO) as the backbone, replacing or augmenting standard convolutional feature extractors with spectral convolution layers that operate in function space.

**Project:** [Neural Operators for Learning Lensing Maps](https://ml4sci.org/gsoc/2026/proposal_DEEPLENSE5.html)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architectures](#model-architectures)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Comparison with Common Test I](#comparison-with-common-test-i)
- [Discussion: Neural Operators vs Standard CNNs](#discussion-neural-operators-vs-standard-cnns)
- [Future Directions](#future-directions)
- [Installation & Usage](#installation--usage)
- [File Structure](#file-structure)
- [References](#references)

---

## Problem Statement

Classify strong gravitational lensing images into three categories:
1. **No Substructure** (`no`) -- Strong lensing images without substructure
2. **Spherical Substructure** (`sphere`) -- Images with CDM subhalo substructure
3. **Vortex Substructure** (`vort`) -- Images with vortex/axion substructure

The classifier must use a **neural operator architecture** (e.g., Fourier Neural Operator) as the backbone, and performance is compared against the Common Test I baseline.

**Evaluation Metrics:** ROC Curve and AUC Score

---

## Dataset

- **Source:** [Google Drive Dataset](https://drive.google.com/file/d/1QUVUpknFKMKLKvzWz-BWBOnL1Mf8b5tv/view)
- **Format:** NumPy arrays (`.npy` files)
- **Image Size:** 150x150 pixels, single-channel (grayscale)
- **Classes:** 3 (no, sphere, vort) -- balanced
- **Preprocessing:** Min-max normalized
- **Split:** Train (30,000) / Validation (6,375) / Test (1,125)

---

## Approach

### Why Neural Operators for Gravitational Lensing?

Neural operators learn mappings between **function spaces** rather than finite-dimensional vectors. This is fundamentally aligned with gravitational lensing physics:

1. **Physics Connection:** Gravitational lensing maps a continuous mass distribution to a continuous lensed image via the lens equation (a PDE). Neural operators were designed precisely to learn such PDE solution operators. The spectral convolution in FNO is conceptually aligned with how lensing distortions manifest in Fourier space.

2. **Global Receptive Field:** FNO processes the entire image in the frequency domain from layer 1. Unlike CNNs that build receptive fields gradually through stacking, neural operators capture global structures (lensing arcs, Einstein rings) immediately.

3. **Resolution Invariance:** FNO can generalize across input resolutions by spectral zero-padding. A model trained at 150x150 can infer at higher resolution -- critical for real survey data from different telescopes (HSC-SSP, Euclid, Rubin).

4. **Spectral Sparsity:** Lensing images have structured frequency content. The spectral convolution naturally exploits this by operating only on the most informative frequency modes.

5. **Novel for DeepLense:** This is the first exploration of neural operators within the ML4SCI DeepLense ecosystem.

### Strategy

Two neural operator architectures are implemented and compared:

| Model | Approach | Neural Operator Component |
|-------|----------|--------------------------|
| **FNO Classifier** | Direct use of `neuraloperator` library | `SpectralConv` layers (FFT-based) |
| **FNO-Enhanced ResNet** | Pretrained ResNet-18 + spectral branch | Custom `SpectralConvBlock` injection |

---

## Model Architectures

### Model 1: FNO Classifier

Direct adaptation of the Fourier Neural Operator for classification. Uses `SpectralConv` from the official `neuraloperator` library (v2.0.0).

```
Input (1x150x150)
       |
  Lifting (Conv 1x1: 1 -> 64 channels, GELU)
       |
  4x FNO Layers (with residual connections):
    ┌──────────────────────────────────────────┐
    │  SpectralConv(64, 64, modes=32x32)       │
    │  + Conv1x1(64, 64)  [local skip]         │
    │  -> GELU -> BatchNorm -> + residual       │
    └──────────────────────────────────────────┘
       |
  Projection (Conv 1x1: 64 -> 128, GELU)
       |
  Global Average Pooling
       |
  Dropout(0.3) -> Linear(128 -> 3)
```

Each `SpectralConv` layer:
1. Applies `rfft2` to go to Fourier domain
2. Multiplies by learned complex weight tensor (keeping lowest 32x32 modes)
3. Applies `irfft2` to return to spatial domain
4. Adds a local Conv1x1 skip connection + residual from input

**Parameters:** ~8.9M

### Model 2: FNO-Enhanced ResNet (Hybrid)

Pretrained ResNet-18 augmented with a parallel spectral convolution branch after layer1. This directly addresses the task requirement of *"replacing or augmenting the standard convolutional feature extractor with a neural operator layer."*

```
Input (1x150x150)
       |
  ResNet Stem (conv1 + bn + relu + maxpool)
       |
  ResNet layer1 (64 channels)
       |
  ┌────────────────┴────────────────┐
  |                                 |
  | Spectral Branch:                | Spatial Branch:
  | 2x [SpectralConvBlock(64ch,     | ResNet layer2 (128ch)
  |      modes=16x16)               | ResNet layer3 (256ch)
  |     + BatchNorm + GELU]         | ResNet layer4 (512ch)
  | Projection (64->512 via         |
  |  3x strided Conv2d)             |
  |                                 |
  └────────────────┬────────────────┘
                   | (addition + adaptive pooling)
             Global Average Pooling
                   |
             Dropout(0.3) -> Linear(512 -> 3)
```

Each `SpectralConvBlock`:
1. Applies `rfft2` (ortho-normalized)
2. Multiplies by learned complex weight tensor in frequency domain
3. Applies `irfft2` back to spatial domain

The spectral branch provides **global frequency-domain features** that complement the **local spatial features** from the ResNet backbone. The model benefits from ImageNet pretraining on the spatial branch.

**Parameters:** ~13.9M

---

## Training Strategy

### Hyperparameters

| Parameter | FNO Classifier | FNO-Enhanced ResNet |
|-----------|---------------|---------------------|
| Epochs | 150 | 150 |
| Batch Size | 64 | 64 |
| Learning Rate | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Weight Decay | 0.01 | 0.05 |
| Warmup Epochs | 10 | 10 |
| Scheduler | Warmup + CosineAnnealing | Warmup + CosineAnnealing |
| Early Stopping | Patience 30 (ROC-AUC) | Patience 30 (ROC-AUC) |
| Label Smoothing | 0.1 | 0.1 |
| Gradient Clipping | max_norm=1.0 | max_norm=1.0 |

### Data Augmentation

| Augmentation | Parameters |
|-------------|------------|
| Random Horizontal Flip | p=0.5 |
| Random Vertical Flip | p=0.5 |
| Random 90-degree Rotations | {0, 90, 180, 270} |
| Gaussian Noise | sigma in [0, 0.05], p=0.5 |
| Random Brightness | [0.8, 1.2], p=0.5 |
| Random Contrast | [0.8, 1.2], p=0.5 |
| Random Center Crop | [80%, 95%], p=0.3 |

---

## Results

### Performance Comparison

| Model | ROC-AUC (macro) | ROC-AUC (micro) | Accuracy | Parameters |
|-------|----------------|----------------|----------|------------|
| **FNO Classifier** | 0.9717 | 0.9738 | 88.31% | ~8.9M |
| **FNO-Enhanced ResNet** | **0.9888** | **0.9899** | **93.54%** | ~13.9M |
| Common Test I (ESCNN+ResNet) | 0.9882 | 0.9885 | 93.71% | ~11.5M |

### Test Set Evaluation (Best Model: FNO-Enhanced ResNet)

| Metric | Score |
|--------|-------|
| **Test ROC-AUC (macro)** | **0.9890** |
| **Test Accuracy** | **93.60%** |

### Key Findings

- **FNO-Enhanced ResNet surpasses the Common Test I baseline** (AUC 0.9888 vs 0.9882), demonstrating that augmenting CNNs with spectral convolution layers improves classification performance on lensing images.
- **Pure FNO Classifier** achieves AUC 0.9717 -- respectable for a from-scratch model without pretraining, but the lack of ImageNet initialization limits its performance compared to the hybrid approach.
- The **spectral branch** in the hybrid model provides complementary global frequency-domain features that the spatial ResNet layers alone cannot capture, particularly for the ring/arc structures characteristic of gravitational lensing.

### ROC Curves

![ROC Comparison](checkpoints/comparison_roc.png)

---

## Comparison with Common Test I

| Aspect | Common Test I | This Work (Neural Operators) |
|--------|--------------|------------------------------|
| **Architecture** | ESCNN Canonicalization + ResNet-18 | FNO / FNO-Enhanced ResNet |
| **Best AUC** | 0.9882 | **0.9888** |
| **Feature Domain** | Spatial (pixel convolutions) | Spatial + Spectral (Fourier coefficients) |
| **Receptive Field** | Local -> global (stacked layers) | Global from layer 1 (full FFT) |
| **Resolution** | Fixed 150x150 | Resolution-invariant (spectral zero-padding) |
| **Physics Prior** | Rotation equivariance (E(2) symmetry) | Spectral sparsity + global structure |
| **Pretraining** | ImageNet (ResNet backbone) | ImageNet (hybrid) / From scratch (FNO) |

The FNO-Enhanced ResNet slightly outperforms the Common Test I baseline, showing that **spectral convolution layers provide meaningful complementary features** for lensing classification. The pure FNO model, while not matching the pretrained baseline, demonstrates that neural operators can learn useful representations from scratch on this dataset.

---

## Discussion: Neural Operators vs Standard CNNs

### How Neural Operators Differ

| Property | Standard CNN | FNO (Neural Operator) |
|----------|-------------|----------------------|
| **Domain** | Spatial (pixel convolutions) | Spectral (Fourier coefficients) |
| **Receptive field** | Local -> global (stacked layers) | Global from layer 1 (full FFT) |
| **Resolution** | Fixed input size | Resolution-invariant (spectral zero-padding) |
| **Complexity** | O(N) per layer | O(N log N) per layer (FFT) |
| **Inductive bias** | Translation equivariance | Spectral sparsity + global structure |

### Why the Hybrid Approach Works Best

The FNO-Enhanced ResNet outperforms both the pure FNO and the pure CNN baselines because:

1. **Complementary features:** ResNet captures local texture and edge features; the spectral branch captures global frequency patterns (ring structures, arc symmetries).
2. **Pretrained initialization:** The ResNet backbone starts from ImageNet weights, providing a strong feature extraction foundation that the spectral branch enhances.
3. **Parallel fusion:** The additive fusion after layer1 allows the model to learn an optimal blend of spatial and spectral representations.

### Relevance to the GSoC Project

The GSoC project *"Neural Operators for Learning Lensing Maps"* proposes learning the functional mapping: **mass distribution -> lensed image**. While this test focuses on classification (a simpler downstream task), the same spectral convolution layers demonstrated here would form the core of a full neural operator surrogate simulator. The results validate that neural operators can effectively process lensing images, supporting the feasibility of the more ambitious operator learning task.

---

## Future Directions

### 1. Equivariant Neural Operators

Gravitational lensing images are rotationally symmetric -- rotating a lens image does not change its substructure class. Standard FNO is translation-equivariant (via FFT) but **not** rotation-equivariant. Combining equivariant architectures with neural operators is a promising direction:

- **ESCNN Canonicalization + FNO Backbone:** Use the E(2)-equivariant canonicalizer from Common Test I to normalize image orientation, then feed into an FNO classifier. This separates geometric symmetry handling from spectral feature extraction.
- **E(2)-Equivariant Lifting + FNO + Invariant Pooling:** Use `escnn` steerable convolutions for the lifting/projection layers (handling group structure) with FNO layers in between for spectral processing. Group pooling at the end ensures rotation invariance.
- **Steerable Spectral Convolutions:** Replace standard `SpectralConv` with group-equivariant spectral convolutions that respect rotation symmetry directly in Fourier space. This is theoretically elegant but requires custom implementation beyond what `neuraloperator` currently supports.

### 2. Backbone Exploration

The modular FNO-Enhanced design allows systematic backbone swapping:

| Backbone | Expected Benefit |
|----------|-----------------|
| **EfficientNet-B0/B2** | Better accuracy-efficiency tradeoff |
| **ConvNeXt-Tiny** | Modern CNN with larger effective receptive field |
| **Swin Transformer** | Hierarchical attention + spectral branch |
| **ResNet-50** | Deeper spatial features for spectral fusion |
| **MobileNetV3** | Lightweight deployment with spectral augmentation |

### 3. Full Operator Learning

Extend from classification to learning the lens equation mapping (**mass distribution -> lensed image**) using:
- **FNO** for the forward mapping (mass -> image)
- **DeepONet** for the inverse mapping (image -> mass parameters)
- **Physics-Informed Neural Operators (PINO):** Incorporate the lens equation as a soft constraint during training

### 4. Resolution Transfer

Train at 150x150, evaluate at higher resolutions (300x300, 600x600) to demonstrate the resolution invariance property of FNO -- critical for real survey data from different telescopes with varying pixel scales.

### 5. Uncertainty Quantification

Use neural operator ensembles or probabilistic variants (e.g., Bayesian FNO) for uncertainty estimation in lens classification, enabling more reliable automated lens detection in large survey pipelines.

---

## Installation & Usage

### Requirements

```bash
pip install torch torchvision timm neuraloperator scikit-learn matplotlib numpy tqdm
```

### Running the Notebook

1. Download the dataset from [Google Drive](https://drive.google.com/file/d/1QUVUpknFKMKLKvzWz-BWBOnL1Mf8b5tv/view)
2. Extract to `~/work/dataset/` directory
3. Run `neural_operator_classifier.ipynb`

---

## File Structure

```
Specific_Test_IV_Neural_Operators/
  README.md                            # This file
  neural_operator_classifier.ipynb     # Main notebook with full implementation
  checkpoints/
    best_FNO_Classifier.pth            # Trained FNO model weights
    best_FNO_Enhanced_ResNet.pth       # Trained hybrid model weights
    roc_FNO_Classifier.png             # FNO ROC curve
    roc_FNO_Enhanced_ResNet.png        # Hybrid ROC curve
    roc_test_best.png                  # Test set ROC curve (best model)
    comparison_roc.png                 # Side-by-side comparison
```

---

## References

1. **FNO:** Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential Equations." ICLR 2021.
2. **AFNO:** Guibas, J. et al. "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers." ICLR 2022.
3. **neuraloperator:** Kossaifi, J. et al. "A Library for Learning Neural Operators." arXiv:2412.10354, 2025.
4. **FNO for Classification:** Kabri, S. et al. "Resolution-Invariant Image Classification based on Fourier Neural Operators." DAGM GCPR 2022.
5. **ESCNN:** Weiler, M. & Cesa, G. "General E(2)-Equivariant Steerable CNNs." NeurIPS 2019.
6. **DeepLense:** [ML4SCI DeepLense Project](https://github.com/ML4SCI/DeepLense)

---

## Author

**Susmanth Reddy**
GSoC 2026 Applicant -- ML4SCI DeepLense

---
