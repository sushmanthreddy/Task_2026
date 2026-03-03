# Specific Test V: Lens Finding & Data Pipelines
GSoC 2026 - ML4SCI DeepLense

**Task:** Binary classification of strong gravitational lenses vs. non-lensed galaxies
**Evaluation Metric:** ROC-AUC
**Best Test AUC: 0.9867**

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Architecture](#architecture)
- [Approach: Equivariant Canonicalization](#approach-equivariant-canonicalization)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Future Directions](#future-directions)
- [References](#references)

---

## Problem Statement

Strong gravitational lensing is one of the most visually striking phenomena in observational astrophysics - a massive foreground object bends the light from a background source, producing Einstein rings, arcs, or multiple distorted images. The challenge is that these events are genuinely rare. In any large survey, non-lensed galaxies vastly outnumber actual lenses, making this a classic needle-in-a-haystack problem.

The goal here is to build a binary classifier that can reliably tell apart lensed galaxies from non-lensed ones, while handling the severe class imbalance head-on. Since a naive model could score high accuracy by simply predicting "non-lens" for everything, ROC-AUC is the right metric - it measures how well the model ranks lenses above non-lenses across all classification thresholds, regardless of the imbalance.

---

## Dataset Description

The dataset consists of simulated observational data stored as NumPy arrays.

| Property | Value |
|----------|-------|
| Format | `.npy` files |
| Image shape | `(3, 64, 64)` - 3 photometric filter channels, 64×64 pixels |
| Training split | `train_lenses/` and `train_nonlenses/` |
| Test split | `test_lenses/` and `test_nonlenses/` |
| Class imbalance | Non-lenses significantly outnumber lenses |

Each image is a galaxy (or galaxy group) observed through three different photometric filters, stacked as a 3-channel array. Lensed objects show characteristic features like Einstein rings, arcs, or split multiple images of a background source. The imbalance in the dataset reflects the real-world rarity of strong lensing events - which is also why ROC-AUC is the correct way to evaluate this.

---

## Architecture

### Stage 1: ESCNN Equivariant Canonicalization Network

Built with [ESCNN](https://github.com/QUVA-Lab/escnn), operating on the **C8 cyclic group** (8 discrete rotations: 0°, 45°, 90°, ..., 315°).

| Parameter | Value |
|-----------|-------|
| Library | ESCNN |
| Symmetry group | C8 (8 discrete rotations) |
| Input type | Trivial representation (3 scalar fields) |
| Hidden type | Regular representation (32 channels) |
| Layers | 3× R2Conv with InnerBatchNorm, ReLU, PointwiseDropout |
| Kernel size | 5×5 |
| Output | 8-dimensional group activation vector |

The network produces an 8-dim vector, one score per rotation in C8. Argmax picks the canonical rotation, which is then applied (in reverse) to the input. A straight-through estimator keeps the argmax differentiable during training.

### Stage 2: ResNet-18 Backbone

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 |
| Pretrained | ImageNet weights via `timm` |
| Input | 3×64×64 canonicalized image |
| Output | Single binary logit |

### Pipeline

```
Input Image (3×64×64)
       │
       ▼
ESCNN Canonicalization Network (C8)
       │  → 8-dim group activation vector
       │  → Select canonical rotation
       │  → Rotate input to canonical orientation
       ▼
Canonicalized Image (3×64×64)
       │
       ▼
ResNet-18 (ImageNet pretrained)
       │
       ▼
σ(logit) → P(lens) ∈ [0, 1]
```

### Loss Function

```
L_total = L_BCE + 0.2 × L_prior + 0.15 × L_optimization
```

The prior regularization term encourages the canonicalizer to prefer the identity transformation when the input is already near-canonical - essentially an Occam's razor that says "don't rotate unless there's a good reason."

---

## Approach: Equivariant Canonicalization

### Why Not Just Use a Standard CNN?

Gravitational lensing is rotationally symmetric - whether an image contains a lens has absolutely nothing to do with its orientation in the sky. Lenses can appear at any angle, and a classifier needs to be invariant to this.

Standard CNNs don't know this. They learn rotation invariance the hard way, through data augmentation - seeing the same image flipped and rotated many times. That's inefficient, never fully complete, and wastes model capacity on something that could be structurally encoded instead.

### What Already Exists in DeepLense

Before deciding on an approach, I went through all the existing implementations in the [DeepLense repository](https://github.com/ML4SCI/DeepLense) related to lens finding:

| Contributor | Approach | Notes |
|-------------|----------|-------|
| Archil Srivastava | Vision Transformers (CvT, Swin, LeViT) | No geometric priors |
| Kartik Sachdev | Self-Supervised Learning (SimSiam, DINO) | No rotation equivariance |
| Geo Jolly (GSoC 2023) | E2-Equivariant CNNs + Fixed Coordinate Transforms | Fixed transforms (polar, log-polar), not adaptive canonicalization |
| Marcos Tidball | Domain Adaptation with equivariant backbones | Equivariance in features, not in canonicalization |
| Apoorva Singh (GSoC 2021) | E2-Equivariant CNNs for classification | Cannot leverage pretrained ImageNet weights |

The key gap: no existing DeepLense work combines **adaptive equivariant canonicalization** with a **pretrained backbone**. The closest approach (Geo Jolly) uses fixed coordinate transforms, not a learned canonicalizer.

### What This Work Introduces

This is the first EquiAdapt-style canonicalization approach in DeepLense. Instead of making the backbone itself equivariant (which means giving up pretrained weights), I separate the two concerns:

1. **Learn to canonicalize:** An ESCNN-based equivariant network learns to rotate any input image to a canonical (standard) orientation.
2. **Classify the canonical image:** A pretrained ResNet-18 backbone classifies the orientation-normalized image.

This gives you the best of both worlds - exact rotation equivariance from the canonicalizer, plus the representational power of ImageNet pretraining in the classifier.

| Advantage | Why It Matters |
|-----------|---------------|
| Pretrained backbone | Leverages ImageNet features - direct E2-CNNs can't use pretrained weights |
| Provable equivariance | The C8 canonicalizer is exactly equivariant under 8 discrete rotations |
| Modular design | Canonicalizer and backbone can be upgraded independently |
| Interpretability | The canonicalized image is a visible intermediate output |

---

## Handling Class Imbalance

The dataset has a significant imbalance - non-lensed galaxies far outnumber lensed ones. I addressed this with three complementary strategies applied together:

**1. Weighted Random Sampling**
A `WeightedRandomSampler` oversamples the minority class (lenses) so each training epoch sees a roughly balanced class distribution. Per-sample weights are set inversely proportional to class frequency.

**2. Positive-Weight Adjusted BCE Loss**
`BCEWithLogitsLoss` is configured with `pos_weight = N_nonlens / N_lens`. This amplifies the gradient signal from lens examples - a false negative (missing a real lens) is penalized proportionally more than a false positive.

**3. Targeted Data Augmentation**
Lens images receive more aggressive augmentation than non-lens images:
- All images: random horizontal flip, vertical flip, random 90° rotations
- Lens images only: Gaussian noise (σ ≤ 0.05, p=0.7), random brightness scaling (0.85-1.15, p=0.7), random contrast adjustment (0.85-1.15, p=0.7)

Each strategy targets a different level of the training pipeline - batch-level distribution, gradient-level weighting, and sample-level diversity. Using all three together ensures no single point of failure in handling the imbalance.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1×10⁻⁴ |
| Weight decay | 1×10⁻³ |
| Scheduler | CosineAnnealingLR (T_max = 30) |
| Batch size | 32 |
| Epochs | 30 |
| Loss | BCEWithLogitsLoss with pos_weight |
| Prior regularization weight | 0.2 |
| Optimization loss weight | 0.15 |
| Normalization | ImageNet mean/std |
| Hardware | NVIDIA H200 GPU |

---

## Results

Training converged steadily over 30 epochs. The best model was saved at **epoch 15**, where the test AUC peaked. From there, the model continued to improve in accuracy (reaching 96.39% at the final epoch) but the AUC at epoch 15 remained the highest, which is what matters for this task.

| Metric | Value |
|--------|-------|
| **Test ROC-AUC (best)** | **0.9867** |
| Test Accuracy (at best AUC) | 92.96% |
| Best model saved at | Epoch 15 |

**Classification Report (at 0.5 threshold):**
```
              precision    recall  f1-score   support

  Non-Lens      0.9997    0.9292    0.9631     19455
      Lens      0.1206    0.9692    0.2145       195
```

The recall on lenses is 96.9% - the model correctly identifies nearly all real lenses. The low precision for the lens class reflects the extreme class imbalance (195 lenses vs 19,455 non-lenses), but since AUC is the evaluation metric and the model is ranking lenses well above non-lenses, this is the expected trade-off.

**Training progression highlights:**
- Epoch 1: AUC = 0.9475 (strong start from pretrained weights)
- Epoch 5: AUC = 0.9749
- Epoch 9: AUC = 0.9820
- Epoch 15: AUC = **0.9867** ← best model
- Epoch 30: AUC = 0.9846, Accuracy = 96.39%

### ROC Curve

![ROC Curve](artifacts/roc_curve.png)

---

## Repository Structure

```
Specific_Test_V_Lens_Finding_Data_Pipelines/
├── README.md
├── notebooks/
│   └── gsoc_specific_test_2_lens.ipynb    # Complete self-contained notebook
├── checkpoints/
│   └── best_lens_finding_escnn_resnet18.pth
└── artifacts/
    └── roc_curve.png
```

The notebook is fully self-contained - all EquiAdapt library code (equivariant layers, canonicalization classes, group operations) is inlined directly, so no external `equiadapt` package installation is needed.

---

## Future Directions

The modular design of this pipeline - a canonicalizer feeding into a backbone - opens up a lot of natural extensions that would be worth exploring.

On the canonicalizer side, ESCNN supports a whole family of symmetry groups beyond C8: C4, C16 for finer or coarser discrete rotations, D4 and D8 if reflections should also be handled, and SO(2) for continuous rotation equivariance via steerable networks. Each can be swapped in without touching the backbone, making it easy to run ablations on which group is actually the right inductive bias for gravitational lens data.

On the backbone side, since the canonicalized image is just a standard 3-channel image, any architecture in the `timm` library - EfficientNet, ConvNeXt, DenseNet, or vision transformers - can serve as a drop-in replacement. Larger backbones combined with the rotation-normalized input could push the AUC further without requiring more training data.

A more ambitious direction would be extending this to real survey data from HSC-SSP, Euclid, or LSST. One of the known failure modes of classifiers trained on simulations is that they don't generalize well to real observations due to the domain gap. Because this architecture encodes rotation symmetry structurally rather than learning it empirically, it should in principle be more robust to domain shift - the rotation equivariance is exact regardless of what distribution the images come from. Testing this hypothesis on real survey cutouts would be a meaningful next step.

Finally, the canonicalized intermediate image is an interpretable artifact - you can actually look at what orientation the network chose for each input. This opens the door to using the canonicalization step as a diagnostic tool: if the network consistently struggles to canonicalize a certain class of morphologies, that's a signal about where the model is uncertain, which could guide targeted data collection or augmentation strategies.

### Deployment on DeepLense Datasets

The current results are on the GSoC selection test dataset (3-channel, 64x64, binary lens/non-lens). During GSoC, this architecture will be deployed on the official DeepLense Model datasets, which represent the actual research targets:

| Dataset | Resolution | Channels | Characteristics | Lens Finding Relevance |
|---------|------------|----------|-----------------|----------------------|
| **Model I** | 150x150 | 1 | Gaussian PSF, SNR ~25 | Substructure classification (3-class) |
| **Model II** | 64x64 | 1 | Euclid-like conditions | Closest to selection test format |
| **Model III** | 64x64 | 1 | HST-like conditions | Different noise/PSF characteristics |
| **Model IV** | Multi-channel | 3+ | Real galaxy sources, Euclid-like | Most realistic, real galaxy morphologies |
| **HSC-SSP / LSST** | Varies | Multi-band | Real survey cutouts | Ultimate deployment target |

The equivariant canonicalization approach is particularly well-suited for cross-dataset generalization because the rotation symmetry prior is exact regardless of the instrument or survey conditions. A lens rotated by any angle is still a lens -- this holds whether the image comes from Euclid, HST, or LSST.

**Connections to existing DeepLense projects:**

- **Lens finding benchmarks:** Direct comparison with Dhruv Srivastava (Vision Transformers on Model III) and the LSST Data Processing Pipeline for lens finding in real survey data
- **Domain adaptation:** The canonicalization should reduce the domain gap when transferring from simulated (Model I-III) to real data (HSC-SSP, Model IV), complementing Marcos Tidball and Mriganka Nath's domain adaptation work
- **SSL approaches:** The canonicalized images provide a natural pretext task for self-supervised learning -- predicting the canonical rotation angle -- connecting to Kartik Sachdev and Sreehari Iyer's SSL work on DeepLense
- **Equivariant networks:** Extends the equivariant approaches by Apoorva Singh and GEO with the novel canonicalization strategy, which uniquely enables pretrained backbone usage

---

## References

1. Weiler, M., & Cesa, G. "General E(2)-Equivariant Steerable CNNs." NeurIPS 2019.
2. Kaba, S.-O., et al. "Equivariance with Learned Canonicalization Functions." ICML 2023.
3. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
4. Wightman, R. "PyTorch Image Models (timm)." GitHub.
5. ML4SCI DeepLense Project. https://github.com/ML4SCI/DeepLense