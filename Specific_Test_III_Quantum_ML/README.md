# Specific Test III: Quantum ML for Dark Matter Substructure Classification

GSoC 2026 - ML4SCI DeepLense

| Model | Test Acc | ROC-AUC |
|-------|----------|---------|
| ResNet-18 + Quantum Circuit | 92.40% | 0.9837 |
| E2-Equivariant CNN + Quantum Circuit | 94.93% | 0.9812 |
| E2-Equivariant CNN + p4m Equivariant QCNN | **96.93%** | **0.9966** |

---

## What this is

Three-class classification of strong gravitational lensing images - no substructure, spherical substructure (CDM subhalos), and vortex substructure (axion dark matter). We built three hybrid quantum-classical models that pair different classical backbones with parameterized quantum circuits.

**Dataset:** 30,000 training images, 150×150 single-channel `.npy` files, three balanced classes (`no`, `sphere`, `vort`).

---

## Why TorchQuantum

We went with TorchQuantum instead of PennyLane because backpropagation through quantum circuits on GPU just works out of the box - it sits on top of PyTorch's autograd, so the whole model trains end-to-end with `loss.backward()` like any other neural network. PennyLane's default simulator doesn't do GPU-accelerated backprop the same way, and the parameter-shift rule it relies on for gradients needs two circuit evaluations per parameter per step, which made training noticeably slower in our experiments. TorchQuantum also batches the entire mini-batch through the quantum circuit in one vectorized call, which matters when you're running 30k images for 80 epochs.

---

## The three models

Models A and B follow the same pattern: classical backbone extracts features → MLP compresses them down to 8 values → quantum circuit processes them → MLP maps to 3 class logits. The quantum circuit is identical in both - only the backbone changes. Model C takes a different approach: it replaces the variational quantum circuit with a p4m equivariant QCNN that preserves symmetry at the quantum level too.

### Model A: ResNet-18 + Quantum Circuit

`torchquantum_v10.ipynb`

Pretrained ResNet-18 extracts 512-dim features. An MLP gradually compresses them down to 8 values, maps through `tanh × π/2` to get rotation angles, feeds into the quantum circuit, and a small post-net produces class logits. Fine-tuned with differential learning rates and cosine annealing over 80 epochs.

```
Input (1×150×150) → Grayscale→RGB → Resize 224×224
       │
  ResNet-18 (ImageNet pretrained, all layers fine-tuned)
       │  → 512-dim
       ▼
  Pre-net: 512 → 256 → 128 → 64 → 32 → 8  (BN + ReLU + Dropout each)
       │  → tanh(x) × π/2
       ▼
  ┌─ Quantum Circuit (8 qubits, 6 layers)    ─┐
  │  H gates → RY encoding → 6× {             │
  │    forward CNOTs → RY → reverse CNOTs →   │
  │    RZ → skip CNOTs → RX → circular CNOT   │
  │  } → PauliZ measurement                   │
  └───────────────────────────────────────────┘
       │  → 8 values
       ▼
  Post-net: 8 → 32 → 16 → 3
       │
  softmax → P(no), P(sphere), P(vort)
```

### Model B: E2-Equivariant CNN + Quantum Circuit

`hybrid_ecnn_qnn_torchquantum.ipynb`

Replaces ResNet with a C4 rotation-equivariant steerable CNN built with `e2cnn`. Gravitational lensing is rotationally symmetric - an Einstein ring looks the same at any angle - so encoding this symmetry directly into the network makes sense. The equivariant CNN produces rotation-invariant features without needing data augmentation to learn it.

Six equivariant convolutional blocks with group pooling produce a 40,000-dim feature vector. Same MLP compression and quantum circuit as Model A. Trains from scratch, converges in 42 epochs with early stopping.

```
Input (1×128×128)
       │
  C4 Steerable CNN (e2cnn, 6 equivariant blocks)
  ├─ R2Conv(1→24, k=7) + BN + ReLU
  ├─ R2Conv(24→48, k=5) + BN + ReLU + AvgPool
  ├─ R2Conv(48→48, k=5) + BN + ReLU
  ├─ R2Conv(48→96, k=5) + BN + ReLU + AvgPool
  ├─ R2Conv(96→96, k=5) + BN + ReLU
  ├─ R2Conv(96→64, k=5) + BN + ReLU + AvgPool
  └─ GroupPooling → rotation-invariant
       │  → 40,000-dim
       ▼
  Pre-net: 40000 → 256 → 128 → 64 → 32 → 8  (BN + ReLU + Dropout each)
       │  → tanh(x) × π/2
       ▼
  ┌─ Quantum Circuit (8 qubits, 6 layers)    ─┐
  │  same circuit as Model A                  │
  └───────────────────────────────────────────┘
       │  → 8 values
       ▼
  Post-net: 8 → 32 → 16 → 3
       │
  softmax → P(no), P(sphere), P(vort)
```

### Model C: E2-Equivariant CNN + p4m Equivariant QCNN

`equiv_qnn.ipynb`

Takes the equivariant idea to its logical conclusion: equivariance at *both* the classical and quantum levels. The classical backbone is the same C4 steerable CNN from Model B, but the quantum circuit is replaced with a p4m equivariant QCNN. Instead of a generic variational circuit, this uses equivariant U2 gates (RX + IsingZZ + RX + IsingYY) and equivariant pooling (RX + RY + RZ + CRX) that respect p4m symmetry. The QCNN structure progressively reduces 8 qubits to 1 through three conv-pool stages, mirroring classical CNN downsampling. Only 33 trainable quantum parameters.

```
Input (1×150×150)
       │
  C4 Steerable CNN (e2cnn, 6 equivariant blocks)
  ├─ R2Conv(1→24, k=7) + BN + ReLU
  ├─ R2Conv(24→48, k=5) + BN + ReLU + AvgPool
  ├─ R2Conv(48→48, k=5) + BN + ReLU
  ├─ R2Conv(48→96, k=5) + BN + ReLU + AvgPool
  ├─ R2Conv(96→96, k=5) + BN + ReLU
  ├─ R2Conv(96→64, k=5) + BN + ReLU + AvgPool
  └─ GroupPooling → rotation-invariant
       │  → 61,504-dim
       ▼
  Bridge: 61504 → 128 → 64 → 8  (BN + ReLU + Dropout each)
       │  → tanh(x) × π/2
       ▼
  ┌─ p4m Equivariant QCNN (8 qubits, 33 params)   ─┐
  │  RY angle encoding                              │
  │  Conv1: 8 equivariant U2 gates (weight-shared)  │
  │  Pool1: 8→4 qubits (equivariant pooling)        │
  │  Conv2: 2 equivariant U2 gates                  │
  │  Pool2: 4→2 qubits                              │
  │  Conv3: 1 equivariant U2 gate                   │
  │  Pool3: 2→1 qubit → H gate                      │
  │  PauliZ measurement on all 8 wires              │
  └─────────────────────────────────────────────────┘
       │  → 8 values
       ▼
  Post-net: 8 → 32 → 3
       │
  softmax → P(no), P(sphere), P(vort)
```

---

## Quantum circuits

### Models A & B: Variational circuit

8 qubits, 6 variational layers. Hadamard gates create superposition, then RY gates encode the 8 input features as rotation angles.

Each variational layer does:
1. Forward CNOT chain (nearest-neighbor entanglement)
2. Trainable RY rotations
3. Reverse CNOT chain (asymmetric entanglement)
4. Trainable RZ rotations
5. Skip CNOTs (long-range: q0→q2, q2→q4, q4→q6)
6. Trainable RX rotations
7. Circular CNOT (q7→q0, closes the ring)

All three rotation axes (RY+RZ+RX) give full Bloch sphere access. The mixed entanglement pattern - forward, reverse, skip, circular - creates rich correlations without excessive depth. PauliZ measurement on all qubits gives 8 expectation values that go to the post-net.

8 qubits and 6 layers keep the circuit NISQ-compatible - this would run on current IBM/Google hardware.

### Model C: p4m Equivariant QCNN

A fundamentally different quantum circuit design. Instead of generic variational layers, it uses gates that respect the p4m symmetry group - the same dihedral symmetry that governs the gravitational lensing images. The circuit has a hierarchical conv-pool structure that mirrors classical CNNs:

Each equivariant convolution layer applies a U2 gate: RX rotations on both qubits, an IsingZZ entangling gate, more RX rotations, and an IsingYY entangling gate. These 6 parameters per gate are weight-shared across all qubit pairs in a layer, just like a convolutional kernel. Equivariant pooling uses RX + RY + RZ rotations plus a controlled-RX to merge information from one qubit into another (5 params per pool).

Three conv-pool stages reduce 8 qubits → 4 → 2 → 1, with a final Hadamard on the output qubit. Total: 33 trainable quantum parameters (18 conv + 15 pool), compared to 144 in the variational circuit. Despite having far fewer quantum parameters, the equivariant structure is more parameter-efficient because every parameter respects the problem's symmetry.

---

## Results

### Model A - ResNet-18 + QNN

| Class | Precision | Recall | F1 | AUC |
|-------|-----------|--------|-----|-----|
| no | 0.902 | 0.947 | 0.924 | 0.986 |
| sphere | 0.915 | 0.886 | 0.900 | 0.975 |
| vort | 0.956 | 0.941 | 0.948 | 0.990 |

**Test accuracy: 92.40% · ROC-AUC: 0.9837 · 80 epochs, ~170 min**

### Model B - E2-Equivariant CNN + QNN

| Class | Precision | Recall | F1 | AUC |
|-------|-----------|--------|-----|-----|
| no | 0.920 | 0.988 | 0.952 | 0.995 |
| sphere | 0.978 | 0.890 | 0.932 | 0.955 |
| vort | 0.954 | 0.972 | 0.963 | 0.994 |

**Test accuracy: 94.93% · ROC-AUC: 0.9812 · 42 epochs, ~107 min**

### Model C - E2-Equivariant CNN + p4m Equivariant QCNN

| Class | Precision | Recall | F1 | AUC |
|-------|-----------|--------|-----|-----|
| no | 0.942 | 1.000 | 0.970 | 0.996 |
| sphere | 0.987 | 0.929 | 0.957 | 0.995 |
| vort | 0.980 | 0.980 | 0.980 | 0.999 |

**Test accuracy: 96.93% · ROC-AUC: 0.9966 · 50 epochs (early stopping at best), ~149 min**

Model C is the clear winner across the board. Making the quantum circuit itself equivariant - not just the classical backbone - pushes test accuracy from 94.93% to 96.93% and ROC-AUC from 0.9812 to 0.9966. The p4m QCNN uses only 33 trainable quantum parameters versus 144 in the variational circuit, yet achieves better results because every quantum parameter respects the problem's symmetry. Sphere remains the hardest class, but Model C's sphere precision (0.987) is substantially higher than Models A and B, meaning fewer false sphere predictions. The "no" class hits perfect recall (1.000).

---

## Why this methodology

The dressed quantum circuit approach - classical backbone doing feature extraction, quantum circuit doing classification - is the most practical way to apply quantum computing to real image data right now. Raw images are way too high-dimensional for current quantum hardware (150×150 = 22,500 pixels vs 8 qubits), so you need a classical front-end to compress the information first. The question is what that front-end should be.

We explored three answers. Model A uses transfer learning (ResNet-18 pretrained on ImageNet) - this is the standard approach in hybrid QML and gives you strong features immediately. Model B takes a different angle: since gravitational lensing is rotationally symmetric, we use an equivariant CNN that has this symmetry baked into its architecture. The network doesn't need to waste capacity learning that a rotated lens is still a lens - it knows that structurally. This turned out to work better (94.93% vs 92.40%) and train faster (42 vs 80 epochs), even without pretrained weights.

Model C pushes the equivariance principle further: instead of feeding equivariant features into a generic variational quantum circuit, it uses a p4m equivariant QCNN where the quantum gates themselves respect the symmetry group. This is end-to-end equivariance - classical backbone, quantum circuit, and all. The result is the best performance across all metrics (96.93% accuracy, 0.9966 ROC-AUC) with only 33 trainable quantum parameters, showing that encoding the right inductive bias into the quantum circuit matters more than circuit depth or parameter count.

All three models use angle encoding via RY gates and train end-to-end on GPU through TorchQuantum's autograd - no separate quantum optimization loop, no parameter-shift overhead.

## Future directions

The backbone is swappable by design, so the full project would systematically explore which classical-quantum pairings work best for this physics domain.

**More equivariant backbones + QNN:**

The equivariant approach clearly has legs - Model B already shows that encoding rotation symmetry into the backbone improves both accuracy and convergence speed. The reason equivariant networks are a natural fit here is that gravitational lensing physics doesn't care about orientation. A dark matter subhalo distorts light the same way whether the image is rotated 0° or 137°. Standard CNNs have to learn this fact from seeing augmented examples - they burn model capacity on something that's already known from the physics. Equivariant networks encode it as a structural constraint, so every parameter goes toward learning features that actually matter.

Our C4 steerable CNN handles 4 discrete rotations (0°, 90°, 180°, 270°). But there are richer equivariant architectures worth exploring with the quantum head:

- **C8 Steerable CNN + QNN** - the most direct upgrade from C4. C8 handles 8 discrete rotations (every 45°) using the same `e2cnn` library, so it's nearly a drop-in replacement. More rotation samples means the network captures finer angular structure in lensing arcs and subhalo distortions that C4 might miss. The cost is a larger group representation (8 group channels per feature instead of 4), which increases intermediate feature dimensions and compute, but the architecture and training pipeline stay the same. This is the lowest-effort way to test whether finer rotational resolution helps the quantum head.

- **Equivariant Wide ResNet + QNN** - our current equivariant backbone is relatively shallow (6 blocks, no skip connections). An equivariant wide ResNet adds residual connections and wider layers, which helps gradient flow in deeper networks and lets the model learn more complex rotation-invariant features. The richer feature representation going into the quantum circuit could improve classification, especially for the harder sphere class.

- **Harmonic Networks (H-Nets) + QNN** - these use circular harmonic filters instead of standard convolutional kernels. The key difference is that they achieve *continuous* rotation equivariance, not just discrete (C4 or C8). Lensing images can be rotated by any arbitrary angle, so a network that's equivariant to all rotations - not just multiples of 90° - is a better match for the physics. The features coming out of an H-Net would be invariant to any rotation, giving the quantum circuit cleaner input.

- **Equivariant Vision Transformers + QNN** - transformers capture long-range dependencies through self-attention, which CNNs (even equivariant ones) struggle with due to their local receptive fields. Lensing arcs and rings are global structures that span the entire image - an attention mechanism that's also equivariant to rotations could capture these patterns more effectively. The combination of global attention, rotation equivariance, and quantum processing is unexplored territory for this task.

**More pretrained backbones + QNN:**

The transfer learning path is also worth expanding. Deeper ResNets (34, 50) have more layers to extract hierarchical features. EfficientNet uses compound scaling to balance depth, width, and resolution efficiently. ConvNeXt modernizes the pure convolutional design with techniques borrowed from transformers. Vision Transformers (ViT) process the image as a sequence of patches with self-attention. Each of these would feed different feature representations into the quantum circuit, and benchmarking them all on the same dataset would show which classical features the quantum circuit benefits from most.

**Quantum circuit exploration:**
The current circuit is one design point. Scaling to 12 or 16 qubits with correspondingly wider pre-nets, trying data re-uploading where classical features are re-encoded at every layer (shown to make even single-qubit circuits universal approximators by Pérez-Salinas et al.), and quantum kernel methods which estimate kernel functions via quantum circuits instead of using variational optimization - these are all directions that could change the picture.

**Noise robustness:**
Everything so far is ideal simulation. For NISQ feasibility, we'd add depolarizing noise, bit-flip errors, and amplitude/phase damping to the quantum circuit and measure how the accuracy holds up as noise increases. TorchQuantum supports all of these noise models. This would give a concrete answer to whether the circuit design is practical on real hardware.

### Deployment on DeepLense Datasets

The current results are on the GSoC selection test dataset (3-class, 150x150, Google Drive). During GSoC, these hybrid quantum-classical models will be benchmarked on the official DeepLense Model datasets:

| Dataset | Resolution | Characteristics | Quantum ML Relevance |
|---------|------------|-----------------|---------------------|
| **Model I** | 150x150 | Gaussian PSF, SNR ~25, 3 classes | Direct match to current setup |
| **Model II** | 64x64 | Euclid-like, 3 classes | Smaller images = faster quantum training |
| **Model III** | 64x64 | HST-like, 3 classes | Different noise characteristics |
| **Model IV** | Multi-channel | Real galaxy sources, 3 classes | Most challenging -- real morphologies |

The hybrid quantum-classical approach is interesting for cross-dataset evaluation because the quantum circuit operates on a compressed 8-dimensional representation. If the classical backbone extracts physics-relevant features (which the equivariant backbone does by construction), the quantum circuit should generalize across datasets since the underlying physics is the same.

**Connections to existing DeepLense projects:**

- **Classification benchmarks:** Direct comparison with all classification projects (Archil Srivastava, Kartik Sachdev, Saranga Mahanta, Equivariant Networks) on Models I-III, with the added dimension of quantum processing
- **Equivariant approaches:** Model B's E2-equivariant backbone connects to the equivariant work by Apoorva Singh and GEO -- the quantum circuit adds a novel processing layer on top of equivariant features
- **Regression extension:** The quantum circuit can be repurposed for axion mass regression (Yurii Halychanskyi, Zhongchao Guan) by changing the post-net output dimension from 3 classes to 1 continuous value
- **Physics-informed integration:** Combining the quantum circuit with the physics-informed features from my [Specific Test V (Physics-Guided ML)](../Specific_Test_V_Physics_Guided_ML/) work -- feeding convergence/shear features through the quantum circuit instead of raw backbone features

---

## Files

```
Specific_Test_III_Quantum_ML/
├── README.md
├── torchquantum_v10.ipynb                 # Model A: ResNet-18 + Variational QC
├── hybrid_ecnn_qnn_torchquantum.ipynb     # Model B: E2-ECNN + Variational QC
└── equiv_qnn.ipynb                        # Model C: E2-ECNN + p4m Equivariant QCNN
```

---

## References

1. Mari et al. "Transfer learning in hybrid classical-quantum neural networks." Quantum 4, 340, 2020.
2. Pérez-Salinas et al. "Data re-uploading for a universal quantum classifier." Quantum 4, 226, 2020.
3. Preskill. "Quantum Computing in the NISQ era and beyond." Quantum 2, 79, 2018.
4. Weiler & Cesa. "General E(2)-Equivariant Steerable CNNs." NeurIPS 2019.
5. Wang et al. "TorchQuantum." DAC 2022.
6. ML4SCI DeepLense. https://github.com/ML4SCI/DeepLense
