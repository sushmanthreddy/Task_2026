# GSoC 2026 - ML4SCI DeepLense: Specific Test IV - Neural Operators

## Overview

This repository contains my solution for the **Specific Test IV: Neural Operators** task for the Google Summer of Code 2026 application with ML4SCI's DeepLense project.

**Task:** Build a model for classifying gravitational lensing images into three classes using a neural operator architecture as the backbone, replacing or augmenting standard convolutional feature extractors with spectral convolution layers that operate in function space.

**Project:** [Neural Operators for Fast Simulation of Strong Gravitational Lensing](https://ml4sci.org/gsoc/2026/proposal_DEEPLENSE3.html)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architectures](#model-architectures)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Comparison with Common Test I](#comparison-with-common-test-i)
- [Discussion](#discussion)
- [Installation & Usage](#installation--usage)
- [File Structure](#file-structure)
- [References](#references)

---

## Problem Statement

Classify strong gravitational lensing images into three categories:
1. **No Substructure** (`no`) -- Strong lensing images without substructure
2. **Spherical Substructure** (`sphere`) -- Images with CDM subhalo substructure
3. **Vortex Substructure** (`vort`) -- Images with vortex/axion substructure

The classifier must use a **neural operator architecture** as the backbone, and performance is compared against the Common Test I baseline.

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

1. **Physics Connection:** Gravitational lensing maps a continuous mass distribution to a continuous lensed image via the lens equation (a PDE). Neural operators were designed precisely to learn such PDE solution operators. The spectral convolution is conceptually aligned with how lensing distortions manifest in Fourier space.

2. **Global Receptive Field:** Spectral layers process the entire image in the frequency domain from layer 1. Unlike CNNs that build receptive fields gradually through stacking, neural operators capture global structures (lensing arcs, Einstein rings) immediately.

3. **Resolution Invariance:** Neural operators can generalize across input resolutions by spectral zero-padding. A model trained at 150x150 can infer at higher resolution -- critical for real survey data from different telescopes (HSC-SSP, Euclid, Rubin).

4. **Spectral Sparsity:** Lensing images have structured frequency content. The spectral convolution naturally exploits this by operating only on the most informative frequency modes.

### Strategy

Four neural operator architectures are implemented and compared, progressively increasing the use of spectral convolution:

| Model | Approach | Type |
|-------|----------|------|
| **FNO Classifier** | Direct spectral convolution classifier | Pure neural operator |
| **FNO-Enhanced ResNet** | ResNet-18 + parallel spectral branch | Hybrid CNN + spectral |
| **UFNO-Enhanced ResNet** | ResNet-18 + U-shaped spectral branch with skip connections | Hybrid CNN + U-shaped spectral |
| **UFNO Classifier** | Pure U-shaped spectral convolution classifier | Pure neural operator |

---

## Model Architectures

### Model 1: FNO Classifier

Direct adaptation of spectral convolution layers for classification. Uses `SpectralConv` from the official `neuraloperator` library (v2.0.0).

- Lifting layer (1 -> 64 channels) followed by 4 spectral convolution layers (modes=32x32) with residual connections, projection to 128 channels, global average pooling, and a linear classifier.
- Each spectral layer applies FFT, multiplies by learned complex weights in the frequency domain, applies inverse FFT, and adds a 1x1 convolution skip connection.
- **Parameters:** ~8.9M

### Model 2: FNO-Enhanced ResNet (Hybrid)

Pretrained ResNet-18 augmented with a parallel spectral convolution branch after layer1.

- The spectral branch consists of 2 spectral convolution blocks (modes=16x16) with BatchNorm and GELU, projected to 512 channels via strided convolutions.
- The spatial branch continues through ResNet layers 2-4 (512 channels).
- Both branches are fused via addition before global average pooling and classification.
- **Parameters:** ~13.9M

### Model 3: UFNO-Enhanced ResNet (Hybrid)

ResNet-18 augmented with a U-shaped spectral branch that combines spectral convolution with a U-Net encoder-decoder for multi-scale local feature extraction.

- After ResNet layer1, features are lifted and processed by 4 spectral layers:
  - Layers 0-1: spectral convolution + 1x1 convolution (global frequency features)
  - Layers 2-3: spectral convolution + 1x1 convolution + U-Net branch (global + local multi-scale features)
- The U-Net branch provides a 3-level encoder-decoder with skip connections, capturing local spatial structures at multiple scales that complement the global spectral features.
- Projected to 512 channels and fused with ResNet spatial branch via addition.
- **Parameters:** ~15.2M

### Model 4: UFNO Classifier (Pure)

Pure U-shaped spectral convolution classifier without any CNN backbone.

- Lifting stem (strided convolutions to downsample 150x150 -> 38x38) followed by 6 spectral layers:
  - Layers 0-1: spectral convolution + 1x1 convolution
  - Layers 2-5: spectral convolution + 1x1 convolution + U-Net branch
- Global average pooling followed by a 2-layer MLP classifier (width -> 128 -> 3).
- Achieves the best performance among all models despite having no pretrained backbone.
- **Parameters:** ~27.6M

---

## Training Strategy

### Hyperparameters

| Parameter | All Models |
|-----------|-----------|
| Epochs | 150 |
| Batch Size | 64 |
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Weight Decay | 0.05 |
| Warmup Epochs | 10 |
| Scheduler | Warmup + CosineAnnealing |
| Early Stopping | Patience 30 (ROC-AUC) |
| Gradient Clipping | max_norm=1.0 |

### Loss Function

**FocalCrossEntropyLoss** -- a blend of smoothed cross-entropy and focal loss:
- 70% CrossEntropyLoss (label_smoothing=0.1) for stable training
- 30% Focal Loss (gamma=2.0) to focus on hard class-boundary samples where substructure differences are subtle

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

| Model | ROC-AUC (macro) | Accuracy | Parameters |
|-------|----------------|----------|------------|
| FNO Classifier | 0.9717 | 88.31% | ~8.9M |
| FNO-Enhanced ResNet | 0.9888 | 93.54% | ~13.9M |
| UFNO-Enhanced ResNet | 0.9942 | 96.44% | ~15.2M |
| **UFNO Classifier** | **0.9977** | **98.31%** | ~27.6M |
| Common Test I (ESCNN+ResNet) | 0.9882 | 93.71% | ~11.5M |

### Key Findings

- **UFNO Classifier achieves the best performance** (AUC 0.9977, accuracy 98.31%), surpassing all other models including the hybrid approaches and the Common Test I baseline by a significant margin.

- **Progressive improvement with U-shaped architecture:** Adding the U-Net encoder-decoder branch to spectral layers consistently improves performance. The U-Net captures multi-scale local spatial features (edges, arcs, rings at different scales) that complement the global frequency features from spectral convolution.

- **Pure spectral operator outperforms hybrid:** The UFNO Classifier without any pretrained CNN backbone achieves better results than all hybrid models, demonstrating that spectral convolution with multi-scale local processing is sufficient -- and superior -- for lensing image classification.

- **All UFNO models surpass the baseline:** Both UFNO variants significantly outperform the Common Test I baseline (ESCNN+ResNet, AUC 0.9882).

### ROC Curves

**FNO Classifier:**

![FNO Classifier ROC](checkpoints/roc_FNO_Classifier.png)

**FNO-Enhanced ResNet:**

![FNO-Enhanced ResNet ROC](checkpoints/roc_FNO_Enhanced_ResNet.png)

**UFNO-Enhanced ResNet:**

![UFNO-Enhanced ResNet ROC](checkpoints/roc_UFNO_Enhanced_ResNet.png)

**UFNO Classifier (best):**

![UFNO Classifier ROC](checkpoints/roc_UFNO_Classifier.png)

### Combined ROC Comparison

![Combined ROC comparison across all four neural operator models](checkpoints/comparison_roc.png)

### Training History

**FNO Classifier:**

![FNO Classifier training history](checkpoints/history_FNO_Classifier.png)

**FNO-Enhanced ResNet:**

![FNO-Enhanced ResNet training history](checkpoints/history_FNO-Enhanced_ResNet.png)

**UFNO-Enhanced ResNet:**

![UFNO-Enhanced ResNet training history](checkpoints/history_UFNO-Enhanced_ResNet-18_Training.png)

**UFNO Classifier (best):**

![UFNO Classifier training history](checkpoints/history_UFNO_Classifier_Training.png)

---

## Notebook Output Gallery

### FNO Classifier and FNO-Enhanced ResNet (`neural_operator_classifier.ipynb`)

| FNO Training | FNO-Enhanced ResNet Training | FNO Classifier ROC | FNO-Enhanced ResNet ROC |
|---|---|---|---|
| ![](notebook_outputs/fno_classifier_and_enhanced_resnet/01_fno_training_history.png) | ![](notebook_outputs/fno_classifier_and_enhanced_resnet/03_fno_enhanced_resnet_training_history.png) | ![](notebook_outputs/fno_classifier_and_enhanced_resnet/05_fno_classifier_roc.png) | ![](notebook_outputs/fno_classifier_and_enhanced_resnet/06_fno_enhanced_resnet_roc.png) |

### U-Shaped FNO-Enhanced ResNet (`u-shaped-fno_enhanced_resnet.ipynb`)

| Training History | Loss Curve | ROC Curve |
|---|---|---|
| ![](notebook_outputs/ufno_enhanced_resnet/01_training_history.png) | ![](notebook_outputs/ufno_enhanced_resnet/02_loss_curve.png) | ![](notebook_outputs/ufno_enhanced_resnet/03_roc_curve.png) |

### U-Shaped FNO Classifier (`u-shaped_fno_classifier.ipynb`) -- best, AUC 0.9977

| Training History | Loss Curve | ROC Curve |
|---|---|---|
| ![](notebook_outputs/ufno_classifier/01_training_history.png) | ![](notebook_outputs/ufno_classifier/02_loss_curve.png) | ![](notebook_outputs/ufno_classifier/03_roc_curve_best.png) |

---

## Comparison with Common Test I

| Aspect | Common Test I | This Work (Best: UFNO Classifier) |
|--------|--------------|-----------------------------------|
| **Architecture** | ESCNN Canonicalization + ResNet-18 | U-shaped spectral convolution classifier |
| **Best AUC** | 0.9882 | **0.9977** |
| **Best Accuracy** | 93.71% | **98.31%** |
| **Feature Domain** | Spatial (pixel convolutions) | Spectral + multi-scale spatial |
| **Receptive Field** | Local -> global (stacked layers) | Global from layer 1 (full FFT) + multi-scale local (U-Net) |
| **Resolution** | Fixed 150x150 | Resolution-invariant (spectral zero-padding) |
| **Physics Prior** | Rotation equivariance (E(2) symmetry) | Spectral sparsity + multi-scale structure |
| **Pretraining** | ImageNet (ResNet backbone) | From scratch |

The UFNO Classifier **significantly outperforms the Common Test I baseline** (+0.95% AUC, +4.6% accuracy), demonstrating that U-shaped spectral convolution with multi-scale local processing provides superior representations for lensing classification without requiring pretrained weights or explicit symmetry encoding.

---

## Discussion

### Why the U-shaped Spectral Architecture Works Best

1. **Global + Local features:** Spectral convolution captures global frequency patterns (ring structures, arc symmetries) while the U-Net branch captures local multi-scale spatial features (edges, textures, fine substructure details).

2. **Skip connections preserve detail:** The U-Net encoder-decoder with skip connections preserves fine-grained spatial information that would be lost in pure frequency-domain processing.

3. **Progressive complexity:** The architecture processes features through pure spectral layers first (learning global patterns) before introducing the U-Net branch (adding local detail), allowing the model to build representations hierarchically.

4. **No pretrained bias:** Training from scratch allows the model to learn features specifically optimized for lensing images, rather than adapting ImageNet features that may not transfer well to astronomical data.

### Relevance to the GSoC Project

The GSoC project *["Neural Operators for Fast Simulation of Strong Gravitational Lensing"](https://ml4sci.org/gsoc/2026/proposal_DEEPLENSE3.html)* proposes learning the functional mapping: **mass distribution → lensed image**, replacing expensive Lenstronomy-based ray-tracing with a fast neural operator surrogate. The U-Shaped FNO architecture demonstrated here is directly suitable for this task:

**Architecture Adaptability:** The U-Shaped FNO backbone requires minimal modification for image generation. The spectral convolution layers and U-Net encoder-decoder branches remain identical -- only the output head changes from global average pooling + classifier to a spatial projection layer that outputs a full 2D lensed image. The skip connections in the U-Net branch are particularly critical for generation, as they preserve the fine-scale spatial details (sharp arcs, thin Einstein rings) that pure spectral methods tend to blur due to Fourier mode truncation.

**Why This Architecture Fits the Physics:**

| Property | Why it matters for lensing simulation |
|----------|--------------------------------------|
| **Spectral convolution (FFT)** | The lens equation is a PDE -- spectral methods are the natural basis for approximating PDE solution operators. Lensing distortions manifest as structured patterns in Fourier space. |
| **U-Net multi-scale branches** | Lensing involves structure at multiple scales simultaneously: large-scale Einstein ring geometry, intermediate-scale arc morphology, and fine-scale substructure perturbations. |
| **Skip connections** | Preserve high-frequency image details that spectral layers would otherwise lose, which is essential for generating sharp, physically accurate lensed images. |
| **Resolution invariance** | Spectral zero-padding allows a model trained at one resolution to infer at another -- critical for generalizing across different telescope pixel scales (HSC, Euclid, Rubin). |
| **No pretrained backbone** | The model learns features directly from lensing physics rather than adapting generic ImageNet representations, which is necessary when the input domain (convergence maps) has no natural photographic analogue. |

**Comparison with DeepONet:** The project also calls for comparison with DeepONet-style architectures. While DeepONet offers flexibility for irregular grids and point-queryable outputs, the U-Shaped FNO's grid-to-grid mapping is more natural and efficient for the fixed-resolution simulation setting. Both would be implemented and compared on the same training data generated from Lenstronomy.

**Path from Classification to Generation:**

```
Classification (this work):
   Lensed image → [Lift → Spectral+UNet blocks → AvgPool → MLP] → 3 class logits

Generation (GSoC project):
   Convergence map κ(x,y) → [Lift → Spectral+UNet blocks → Project] → Lensed image I(x,y)

Same backbone. Different head. Different loss (MSE + SSIM + spectral loss).
```

The 0.9977 AUC classification result validates that this architecture can effectively capture the relevant physics of gravitational lensing -- the same spectral and multi-scale spatial representations that distinguish substructure classes are exactly what's needed to accurately predict lensed image morphology from mass distributions.

---

## Installation & Usage

### Requirements

pip install torch torchvision timm scikit-learn matplotlib numpy tqdm
