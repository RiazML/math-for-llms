[← Back to Math for Specific Models](../README.md) | [Next: Attention Mechanisms →](../09-Attention-Mechanisms/notes.md)

---

# CNNs and Convolution Mathematics

> _"Convolution is not magic — it is a carefully engineered inductive bias: the assumption that the same pattern detector should fire regardless of where in the image it appears. This single constraint reduces the parameter count of image recognition from billions to thousands."_

## Overview

Convolutional Neural Networks (CNNs) dominated computer vision from 2012 (AlexNet) until the rise of Vision Transformers (ViT) in 2020. Even today, CNNs remain competitive with Transformers on many vision tasks and are fundamental to understanding how structured data (images, audio, time series) can be processed efficiently.

The core mathematical operation — **discrete 2D convolution** — is far richer than its simple formula suggests. It connects to Fourier analysis (the convolution theorem), linear algebra (Toeplitz/circulant matrices), signal processing (filters, frequency response), and differential geometry (equivariance under symmetry groups). Understanding convolution deeply illuminates not only CNNs but also the design of Transformers (attention is a form of learned, content-dependent convolution), sequence models (causal convolution in WaveNet), and graph neural networks.

This section develops the full mathematical theory: from the continuous convolution integral to the discrete cross-correlation computed on GPUs, the linear-algebraic structure that makes backpropagation efficient, the normalization techniques that stabilise training, and the architectural innovations (residual connections, depthwise separable filters) that made modern deep CNNs possible.

## Prerequisites

- Linear algebra: matrix multiplication, eigenvalues, Toeplitz matrices (Chapters 02-03)
- Calculus: partial derivatives, chain rule (Chapters 04-05)
- Probability: random variables, expectation, variance (Chapter 06)
- Neural networks: forward pass, backpropagation (Section 14-02)

## Companion Notebooks

| Notebook                           | Description                                                                                                                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive demos: convolution as sliding dot-product, Fourier interpretation, gradient flow, feature map visualisation, BatchNorm dynamics, receptive field growth                                         |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems with solutions: 2D conv from scratch, output size formula, backprop through conv, BatchNorm, depthwise separable, dilated receptive field, transposed conv, ResNet skip, patch embedding |

## Learning Objectives

After completing this section, you will:

- State the continuous and discrete convolution definitions and explain how cross-correlation differs from true convolution
- Derive the output size formula for any padding/stride/dilation combination
- Express 2D convolution as a matrix multiplication (Toeplitz structure) and use this to derive backpropagation equations
- Compute the parameter count and receptive field of multi-layer CNN architectures
- Implement BatchNorm forward and backward passes, explain why it stabilises training
- Explain depthwise separable convolution and derive its parameter reduction factor
- Draw the receptive field of a dilated convolution stack and explain why exponential dilation is optimal
- Describe how patch embedding in Vision Transformers is equivalent to a strided convolution

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
- [2. Convolution as a Mathematical Operation](#2-convolution-as-a-mathematical-operation)
- [3. Discrete 2D Convolution for Images](#3-discrete-2d-convolution-for-images)
- [4. Linear Algebra View of Convolution](#4-linear-algebra-view-of-convolution)
- [5. Multi-Channel Convolution](#5-multi-channel-convolution)
- [6. Pooling Operations](#6-pooling-operations)
- [7. Normalisation in CNNs](#7-normalisation-in-cnns)
- [8. Backpropagation Through Convolution](#8-backpropagation-through-convolution)
- [9. Depthwise Separable Convolution](#9-depthwise-separable-convolution)
- [10. Dilated (Atrous) Convolution](#10-dilated-atrous-convolution)
- [11. Transposed Convolution](#11-transposed-convolution)
- [12. 1D Convolution for Sequences](#12-1d-convolution-for-sequences)
- [13. CNN Architectures](#13-cnn-architectures)
- [14. CNNs and Vision Transformers](#14-cnns-and-vision-transformers)
- [15. Common Mistakes](#15-common-mistakes)
- [16. Exercises](#16-exercises)
- [17. Why This Matters for AI (2026)](#17-why-this-matters-for-ai-2026)
- [18. Conceptual Bridge](#18-conceptual-bridge)

---

## 1. Intuition and Motivation

### 1.1 Against Fully-Connected Networks

Consider classifying a $224 \times 224$ RGB image. A fully-connected layer from the raw pixels to 4096 hidden units requires $224 \times 224 \times 3 \times 4096 = 616{,}562{,}688$ parameters — over 600 million weights — for a **single layer**. This is problematic for three reasons:

1. **Overfitting:** With 600M parameters and only millions of training images, the network memorises rather than generalises.
2. **Computational cost:** Training and inference are prohibitively slow.
3. **No structure exploitation:** The flat architecture treats pixels at $(0, 0)$ and $(223, 223)$ as completely independent features, ignoring the spatial structure of images.

A CNN achieves dramatically better results with a $3 \times 3$ filter having only **27 parameters** (plus bias), applied to every spatial position. The parameter count drops by a factor of $\sim 10^7$.

### 1.2 Three Key Inductive Biases

CNNs encode three strong assumptions about visual data:

**1. Translation equivariance.** If the input is shifted, the output is shifted by the same amount:

$$f(\text{shift}(\mathbf{x})) = \text{shift}(f(\mathbf{x}))$$

A detector trained to find an edge in one part of the image automatically detects it everywhere. Formally, convolution **commutes with translation**: $(f * g)(x - x_0) = (f * g)(x)$ when $g$ is shifted by $x_0$.

This is the **weight-sharing** mechanism: the same kernel weights are applied at every spatial position. A $3 \times 3$ kernel scans the entire image, detecting the same pattern everywhere, rather than learning a separate detector for each location.

**2. Locality.** Each output unit depends only on a small spatial neighbourhood (the kernel footprint). Features at position $(i, j)$ are determined by input values near $(i, j)$, not the entire image.

This exploits the fact that local image statistics are highly informative: edges, textures, and patterns are local phenomena. A pixel's semantic content (is it part of an eye?) depends primarily on nearby pixels, not pixels on the other side of the image.

**3. Hierarchical composition.** Multiple convolution layers composed together build increasingly abstract representations: pixels → edges → textures → parts → objects. A 5-layer network with $3 \times 3$ kernels has an effective receptive field of $11 \times 11$ pixels at the top layer, yet each layer's computation is local.

**What CNNs do NOT assume:** Translation **invariance** (output unchanged when input shifts) is different from equivariance. CNNs are equivariant — the feature map shifts with the input. Pooling layers introduce approximate invariance by summarising spatial regions.

### 1.3 Historical Timeline

```text
CNN ARCHITECTURE TIMELINE
════════════════════════════════════════════════════════════════════════

  1989  LeCun             LeNet-1: CNNs for handwritten digit recognition
  1998  LeCun             LeNet-5: 5-layer CNN, used in production by banks
  2012  Krizhevsky et al. AlexNet: deep CNN wins ImageNet, era of deep learning
  2014  Simonyan et al.   VGGNet: uniform 3x3 convolutions, very deep networks
  2014  Szegedy et al.    GoogLeNet (Inception): multi-scale parallel convolutions
  2015  He et al.         ResNet: residual connections, 152+ layers possible
  2016  Huang et al.      DenseNet: dense skip connections
  2017  Howard et al.     MobileNet: depthwise separable, mobile deployment
  2017  Chollet           Xception: extreme depthwise separable convolutions
  2018  Tan & Le          EfficientNet: compound scaling law
  2020  Dosovitskiy et al. ViT: pure Transformer for images (no convolutions)
  2022  Liu et al.        ConvNeXt: CNN modernised to match ViT performance
  2023  Liu et al.        ConvNeXt V2: self-supervised pre-training for CNNs

════════════════════════════════════════════════════════════════════════
```

---

## 2. Convolution as a Mathematical Operation

### 2.1 Continuous Convolution

The **continuous convolution** of two functions $f, g: \mathbb{R} \to \mathbb{R}$ is:

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \, g(t - \tau) \, d\tau$$

**Interpretation:** The output at time $t$ is a weighted average of all past and future values of $f$, with weights given by $g$ flipped and shifted to position $t$. The function $g$ is the **filter** or **kernel**; $f$ is the **signal** or **input**.

**Properties of convolution:**

| Property                    | Formula                           |
| --------------------------- | --------------------------------- |
| Commutativity               | $f * g = g * f$                   |
| Associativity               | $(f * g) * h = f * (g * h)$       |
| Distributivity              | $f * (g + h) = (f * g) + (f * h)$ |
| Linearity in both arguments | $(af) * g = a(f * g)$             |
| Derivative rule             | $(f * g)' = f' * g = f * g'$      |

The **derivative rule** is particularly important: it shows why convolving with the derivative of a Gaussian detects edges.

**2D continuous convolution** for images $f, g: \mathbb{R}^2 \to \mathbb{R}$:

$$(f * g)(x, y) = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f(u, v) \, g(x - u, y - v) \, du \, dv$$

### 2.2 Cross-Correlation vs True Convolution

In signal processing, **true convolution** flips the kernel before sliding:

$$(f * g)(i, j) = \sum_{m, n} f(i - m, j - n) \, g(m, n)$$

In deep learning, what is called "convolution" is actually **cross-correlation** — the kernel is NOT flipped:

$$(f \star g)(i, j) = \sum_{m, n} f(i + m, j + n) \, g(m, n)$$

**Why does this matter?** For learned kernels, it makes no difference — if the optimal filter is $g$, the network will learn $g_{\text{flipped}}$ instead, achieving the same result. The distinction matters when:

1. Using handcrafted filters from signal processing (e.g., Sobel edge detector) — these assume true convolution.
2. Computing convolution via FFT — the formula assumes true convolution.
3. Analysing mathematical properties — commutativity holds for true convolution but not cross-correlation.

**In PyTorch and TensorFlow:** `torch.nn.Conv2d` and `tf.nn.conv2d` both implement cross-correlation, despite being called "convolution".

### 2.3 The Convolution Theorem

The most powerful property of convolution:

$$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

**Convolution in the spatial domain equals pointwise multiplication in the frequency domain.** This has profound implications:

1. **Fast computation:** A large convolution (say, $1024 \times 1024$ input with $128 \times 128$ kernel) can be computed as two FFTs, pointwise multiplication, and one inverse FFT. Cost: $O(n^2 \log n)$ vs $O(n^2 k^2)$ for direct computation.

2. **Filter design:** The Fourier transform of the kernel reveals its frequency response. A Gaussian kernel is a low-pass filter (blurs). The derivative of a Gaussian is a band-pass filter (detects edges). The $\operatorname{rect}$ function is a band-pass filter in frequency.

3. **Theoretical analysis:** The eigenvalues of a circulant matrix are the DFT coefficients of its first row — this is why convolution operators are so well-understood mathematically (Section 4.3).

**In practice:** For small kernels ($3 \times 3$, $5 \times 5$), direct computation on GPU is faster than FFT-based computation due to GPU parallelism and cache effects. FFT convolution is used for large kernels (audio processing, astrophysics).

### 2.4 Separable Convolutions

A 2D convolution kernel $\mathbf{K} \in \mathbb{R}^{k \times k}$ is **separable** if it can be written as an outer product of two 1D vectors:

$$\mathbf{K} = \mathbf{u} \otimes \mathbf{v} = \mathbf{u} \mathbf{v}^\top$$

**Computational advantage.** Direct 2D convolution with a $k \times k$ kernel costs $O(k^2 \cdot H \cdot W)$ operations. Separable convolution first applies $\mathbf{v}$ along rows (cost $O(k \cdot H \cdot W)$), then $\mathbf{u}$ along columns (cost $O(k \cdot H \cdot W)$). Total: $O(2k \cdot H \cdot W)$ — a factor-of-$k/2$ reduction.

**Examples of separable kernels:**

- **Gaussian blur:** $G_\sigma(x, y) = G_\sigma(x) \cdot G_\sigma(y)$ — a 2D Gaussian is the product of two 1D Gaussians.
- **Sobel edge detector:** $\mathbf{S}_x = \begin{pmatrix}1 \\ 2 \\ 1\end{pmatrix} \begin{pmatrix}-1 & 0 & 1\end{pmatrix}$

**Separability test.** A matrix $\mathbf{K}$ is separable if and only if it has rank 1 — i.e., $\operatorname{rank}(\mathbf{K}) = 1$. Check via SVD: $\mathbf{K}$ is separable iff only one singular value is nonzero.

**Learned separable filters.** This motivates the architecture choice in Section 9: depthwise separable convolutions explicitly factorise 3D spatial+channel convolutions into a spatial (depthwise) step and a channel-mixing (pointwise) step.

---

## 3. Discrete 2D Convolution for Images

### 3.1 Kernel and Feature Map

Let $\mathbf{X} \in \mathbb{R}^{H \times W}$ be a single-channel input image and $\mathbf{K} \in \mathbb{R}^{k_h \times k_w}$ be a convolutional kernel (filter). The **cross-correlation** output $\mathbf{Y} \in \mathbb{R}^{H' \times W'}$ is:

$$Y[i, j] = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} X[i \cdot s + m, \; j \cdot s + n] \cdot K[m, n] + b$$

where $s$ is the stride and $b$ is the bias. Each output value is a dot product between the kernel and a $k_h \times k_w$ patch of the input centred at position $(i \cdot s, j \cdot s)$.

**Geometric intuition:** The kernel is a sliding window that sweeps across the input, computing a local dot product at every position. Different kernels detect different patterns:

| Kernel                                                                       | Detects                      |
| ---------------------------------------------------------------------------- | ---------------------------- |
| $\begin{pmatrix}-1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{pmatrix}$         | Vertical edges (Sobel $x$)   |
| $\begin{pmatrix}-1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1\end{pmatrix}$         | Horizontal edges (Sobel $y$) |
| $\frac{1}{9}\begin{pmatrix}1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1\end{pmatrix}$ | Blur (average)               |
| $\begin{pmatrix}0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0\end{pmatrix}$        | Sharpening (Laplacian)       |

In CNNs, the kernel values are **learned** from data rather than handcrafted.

### 3.2 Padding Strategies

**Valid padding** (no padding): the kernel must fit entirely within the input. Output is smaller than input.

**Same padding**: zero-pad the input so that the output has the same spatial dimensions as the input (for stride 1). Padding size: $p = \lfloor k/2 \rfloor$ on each side for a $k \times k$ kernel.

**Full padding**: pad enough so that every input pixel appears in exactly one position as the kernel center. Output is larger than input.

```text
PADDING STRATEGIES (1D illustration, input size 5, kernel size 3)
════════════════════════════════════════════════════════════════════════

  Valid:   [x x x x x]                  output: 3 elements
            [k k k]
              [k k k]
                [k k k]

  Same:  [0 x x x x x 0]               output: 5 elements
           [k k k]
             [k k k]
               [k k k]
                 [k k k]
                   [k k k]

  Full:  [0 0 x x x x x 0 0]           output: 7 elements
           [k k k]
             [k k k]
               ...

════════════════════════════════════════════════════════════════════════
```

**Choice in practice:** Same padding is the default in most architectures (preserves spatial dimensions at each layer). Valid padding is used when the output is intentionally downsampled (e.g., replacing a pooling layer with a strided convolution).

### 3.3 Stride

The stride $s$ controls how many pixels the kernel moves between applications. Stride-1 convolution moves one pixel at a time; stride-2 downsamples by a factor of 2.

- **Stride 1:** Output size $\approx$ input size (with same padding).
- **Stride 2:** Output size $\approx$ input size / 2. Common replacement for max pooling — learns what information to preserve rather than using a fixed summary statistic.
- **Stride-2 convolution vs max pooling:** Stride-2 conv has learnable parameters; max pooling does not. Modern architectures (ResNet-50-D, EfficientNet) use stride-2 convolutions rather than pooling.

### 3.4 Dilation

**Dilated convolution** (atrous convolution) introduces gaps between kernel elements, controlled by dilation rate $d$:

$$Y[i, j] = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} X[i + m \cdot d, \; j + n \cdot d] \cdot K[m, n]$$

A $3 \times 3$ kernel with dilation rate $d = 2$ has the same number of parameters as a standard $3 \times 3$ kernel, but covers a $5 \times 5$ spatial region (with holes). With dilation $d = 4$, it covers $9 \times 9$.

**Why dilation?** It exponentially expands the receptive field without increasing parameters or losing resolution — crucial for dense prediction tasks (semantic segmentation, depth estimation) where you need both large receptive fields and high-resolution outputs.

### 3.5 Output Size Formula

For input size $H$ (or $W$), kernel size $k$, padding $p$, stride $s$, dilation $d$:

$$H_{\text{out}} = \left\lfloor \frac{H + 2p - d(k-1) - 1}{s} \right\rfloor + 1$$

**Special cases:**

| Setting              | Formula                       | Example ($H=32, k=3$)               |
| -------------------- | ----------------------------- | ----------------------------------- |
| Valid, stride 1      | $H - k + 1$                   | $32 - 3 + 1 = 30$                   |
| Same, stride 1       | $H$                           | $32$                                |
| Valid, stride 2      | $\lfloor(H - k)/2\rfloor + 1$ | $\lfloor(32 - 3)/2\rfloor + 1 = 15$ |
| Same, stride 2       | $\lceil H/2 \rceil$           | $16$                                |
| Dilated $d=2$, valid | $H - d(k-1)$                  | $32 - 2(2) = 28$                    |

**Verification trick:** For a $k \times k$ kernel, valid padding, stride 1: the output shrinks by $k - 1$ in each dimension. Stacking $L$ such layers shrinks by $L(k-1)$.

---

## 4. Linear Algebra View of Convolution

### 4.1 Convolution as Matrix Multiplication (Toeplitz)

For a 1D input $\mathbf{x} \in \mathbb{R}^n$ and kernel $\mathbf{k} \in \mathbb{R}^m$, convolution can be written as matrix-vector multiplication $\mathbf{y} = \mathbf{C} \mathbf{x}$ where $\mathbf{C}$ is a **Toeplitz matrix**:

$$\mathbf{C} = \begin{pmatrix} k_1 & 0 & \cdots & 0 \\ k_2 & k_1 & \cdots & 0 \\ k_3 & k_2 & k_1 & \vdots \\ 0 & k_3 & k_2 & k_1 \\ \vdots & & \ddots & k_2 \\ 0 & \cdots & 0 & k_3 \end{pmatrix}$$

Each row is a shifted copy of the kernel. The Toeplitz structure enforces **weight sharing** — the same kernel values appear in every row, just shifted.

**For 2D convolution:** The matrix is a **doubly block Toeplitz** matrix (a Toeplitz matrix of Toeplitz blocks). Each block corresponds to one row of the convolution, and the blocks themselves have Toeplitz structure.

**Why this matters:**

1. **Memory:** The full Toeplitz matrix for a $224 \times 224$ image convolution would require $O(n^4)$ entries — storing it would be intractable. The Toeplitz structure means we only need to store $k^2$ parameters.
2. **Backpropagation:** The gradient $\partial \mathcal{L}/\partial \mathbf{x} = \mathbf{C}^\top \partial \mathcal{L}/\partial \mathbf{y}$ — the backward pass through convolution is a convolution with the transposed (flipped) kernel.
3. **Eigenanalysis:** Toeplitz matrices are approximately circulant, and circulant matrices are diagonalised by the DFT.

### 4.2 Doubly Block Circulant Structure

A **circulant matrix** is a special Toeplitz matrix where each row is a circular shift of the previous:

$$\mathbf{C}_{\text{circ}} = \operatorname{circ}(c_0, c_1, \ldots, c_{n-1}) = \begin{pmatrix} c_0 & c_{n-1} & \cdots & c_1 \\ c_1 & c_0 & \cdots & c_2 \\ \vdots & & \ddots & \vdots \\ c_{n-1} & c_{n-2} & \cdots & c_0 \end{pmatrix}$$

With **circular (wrap-around) padding**, a 2D convolution produces a doubly block circulant matrix. This is the mathematically cleanest case because:

1. Circulant matrices commute: $\mathbf{C}_1 \mathbf{C}_2 = \mathbf{C}_2 \mathbf{C}_1$.
2. The DFT diagonalises all circulant matrices simultaneously.

In practice, images use zero (not circular) padding, breaking strict circularity. But for large images relative to kernel size, the approximation is excellent.

### 4.3 Eigenvalues via DFT

**Theorem.** Every circulant matrix $\mathbf{C} = \operatorname{circ}(c_0, \ldots, c_{n-1})$ is diagonalised by the DFT matrix $\mathbf{F}$:

$$\mathbf{C} = \mathbf{F}^{-1} \operatorname{diag}(\hat{\mathbf{c}}) \mathbf{F}$$

where $\hat{\mathbf{c}} = \mathbf{F} \mathbf{c}$ is the DFT of the first column $(c_0, c_1, \ldots, c_{n-1})$.

**Consequence:** The eigenvalues of a convolution operator are the Fourier coefficients of the kernel. A low-pass filter (Gaussian) has eigenvalues that decay for high frequencies — it suppresses high-frequency content. An edge detector has large eigenvalues at mid-frequencies.

**Spectral norm of a convolution layer:** $\lVert \mathbf{C} \rVert_2 = \max_f |\hat{k}_f|$ — the $\ell^\infty$ norm of the kernel's Fourier transform. This is used in spectral normalisation for GANs and theoretical analyses of gradient flow.

### 4.4 Gradient Computation Structure

For the convolution $\mathbf{y} = \mathbf{C}_{\mathbf{k}} \mathbf{x}$ (linear in both $\mathbf{x}$ and $\mathbf{k}$):

**Gradient w.r.t. input:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{C}_{\mathbf{k}}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

Since $\mathbf{C}_{\mathbf{k}}^\top$ corresponds to convolution with the kernel **flipped** (rotated 180°), the backward pass is also a convolution — of the output gradient with the flipped kernel.

**Gradient w.r.t. kernel:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{k}} = \mathbf{C}_{\mathbf{x}}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

The gradient w.r.t. the kernel is a convolution of the input with the output gradient. This is the **cross-correlation** of the input and the upstream gradient — a key insight for implementing conv backprop efficiently.

---

## 5. Multi-Channel Convolution

### 5.1 RGB and Feature Maps

A colour image has $C_{\text{in}} = 3$ input channels. The input tensor is $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$. A single kernel $\mathbf{K} \in \mathbb{R}^{C_{\text{in}} \times k_h \times k_w}$ has one slice per input channel. The output is a single feature map — the sum of convolutions across all input channels:

$$Y[i, j] = b + \sum_{c=0}^{C_{\text{in}}-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X[c, \, i \cdot s + m, \, j \cdot s + n] \cdot K[c, m, n]$$

This is a **3D dot product** between the kernel volume and the input patch, producing a single scalar per spatial position.

### 5.2 Multiple Filters

To produce $C_{\text{out}}$ output channels, we apply $C_{\text{out}}$ different kernels, each of shape $C_{\text{in}} \times k_h \times k_w$. The full weight tensor is $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w}$.

Output tensor $\mathbf{Y} \in \mathbb{R}^{C_{\text{out}} \times H' \times W'}$:

$$Y[f, i, j] = b_f + \sum_{c, m, n} X[c, \, i \cdot s + m, \, j \cdot s + n] \cdot W[f, c, m, n]$$

Each output channel $f$ is a separate feature map, representing the response of filter $f$ at every spatial position.

```text
MULTI-CHANNEL CONVOLUTION
════════════════════════════════════════════════════════════════════════

  Input: [C_in × H × W]

  Filter 1: [C_in × k × k] → Feature Map 1: [H' × W']
  Filter 2: [C_in × k × k] → Feature Map 2: [H' × W']
  ...
  Filter F: [C_in × k × k] → Feature Map F: [H' × W']

  Output: [F × H' × W']    (F = C_out)

  Each filter is a 3D probe asking: "does this pattern exist here?"
  The output is a stack of F such answers, at every spatial location.

════════════════════════════════════════════════════════════════════════
```

### 5.3 Parameter Count

For a convolutional layer with:

- Input channels: $C_{\text{in}}$
- Output channels: $C_{\text{out}}$
- Kernel size: $k \times k$
- Bias: yes (one per output channel)

$$\text{Parameters} = C_{\text{out}} \times C_{\text{in}} \times k \times k + C_{\text{out}} = C_{\text{out}}(C_{\text{in}} k^2 + 1)$$

**Comparison with FC layer** (mapping $C_{\text{in}} \times H \times W$ features to $C_{\text{out}} \times H' \times W'$ features):

$$\text{FC parameters} = (C_{\text{in}} \times H \times W) \times (C_{\text{out}} \times H' \times W')$$

For typical values ($C_{\text{in}} = 128$, $C_{\text{out}} = 256$, $H = W = 28$, $k = 3$):

- Conv: $256 \times (128 \times 9 + 1) = 295{,}168$ parameters
- FC: $(128 \times 28 \times 28) \times (256 \times 28 \times 28) = 3{,}623{,}878{,}656$ parameters

**The ratio is $\sim 12{,}000\times$ — convolutions are vastly more parameter-efficient.**

### 5.4 Receptive Field Analysis

The **receptive field** of a neuron is the region of the input that influences its output value. For a single convolutional layer with kernel size $k$ and stride $s$: receptive field = $k \times k$.

For $L$ stacked convolutional layers with kernel size $k$ and stride 1:

$$\text{Receptive field} = k + (k-1)(L-1) = L(k-1) + 1$$

For $L$ layers with stride $s$ and kernel $k$:

$$\text{Effective receptive field (ERF)} = \sum_{l=1}^{L} k_l \cdot \prod_{l'=1}^{l-1} s_{l'}$$

**Example — VGG block (2 layers, $k=3$, $s=1$):**

After layer 1: RF = 3. After layer 2: RF = $3 + 2 = 5$. Two $3 \times 3$ layers see the same area as one $5 \times 5$ layer, with fewer parameters ($2 \times 9 = 18 < 25$) and more non-linearities.

**ResNet-50 receptive field:** At the final feature map (before global average pooling), the theoretical receptive field is much larger than $224 \times 224$ — the entire image. In practice, the **effective receptive field** (weighted by gradient magnitude) is significantly smaller, as shown by Luo et al. (2017).

| Architecture    | Theoretical RF | Depth    | Top-1 ImageNet |
| --------------- | -------------- | -------- | -------------- |
| VGG-16          | 228            | 13 conv  | 74.4%          |
| ResNet-50       | 483            | 49 conv  | 76.9%          |
| EfficientNet-B7 | 851            | ~66 conv | 84.3%          |

---

## 6. Pooling Operations

### 6.1 Max Pooling

**Max pooling** takes the maximum value in a $k \times k$ spatial window:

$$Y[f, i, j] = \max_{0 \le m < k, \; 0 \le n < k} X[f, \, i \cdot s + m, \, j \cdot s + n]$$

**Properties:**

- **Translation invariance:** Small shifts in the input do not change the max-pooled output (approximately — the shift must be smaller than the pool window).
- **Selectivity:** Keeps the strongest activation in each region, discarding weaker responses.
- **Non-parametric:** No learned parameters — a fixed operation.

**Gradient:** Max pooling is not differentiable at the boundary between which element achieves the max. In practice, the gradient is routed only to the element that was the maximum (all other gradients are zero):

$$\frac{\partial \mathcal{L}}{\partial X[f, i^*, j^*]} = \frac{\partial \mathcal{L}}{\partial Y[f, i, j]}$$

where $(i^*, j^*)$ is the argmax position. This is implemented via **switch variables** that store which element was the max during the forward pass.

### 6.2 Average and Global Average Pooling

**Average pooling** takes the mean over the pool window:

$$Y[f, i, j] = \frac{1}{k^2} \sum_{m, n} X[f, \, i \cdot s + m, \, j \cdot s + n]$$

Average pooling has smoother gradients than max pooling (all elements receive equal gradient) but is less selective.

**Global average pooling (GAP)** reduces each feature map to a single scalar by averaging over the entire spatial extent:

$$Y[f] = \frac{1}{H' \times W'} \sum_{i,j} X[f, i, j]$$

GAP is used at the end of most modern CNNs (ResNet, EfficientNet) instead of fully-connected layers. Advantages:

1. **Fewer parameters:** A $7 \times 7 \times 512$ feature map → $512$ values via GAP, then one FC layer to the number of classes. Compare with FC: $7 \times 7 \times 512 \times 1000 = 25.1$M parameters vs GAP: $512 \times 1000 = 512$K.
2. **Regularisation:** Acts as a structural regulariser — forces the network to produce a spatially meaningful feature map where the feature value at each location indicates the presence of the concept.
3. **Any input size:** No fixed spatial dimension required — the model accepts variable-size inputs.

### 6.3 Adaptive Pooling

PyTorch's `AdaptiveAvgPool2d(output_size)` computes the pooling parameters automatically to produce the desired output size, regardless of input size:

$$s = \lfloor H_{\text{in}} / H_{\text{out}} \rfloor, \quad k = H_{\text{in}} - (H_{\text{out}} - 1) \cdot s$$

This enables models trained at one resolution to be applied at a different resolution at inference time.

### 6.4 Spatial Pyramid Pooling

**SPP** (He et al., 2015) applies multiple pooling operations at different scales and concatenates the results, producing a fixed-size representation from variable-size inputs:

$$\text{SPP}(\mathbf{X}) = [\operatorname{GAP}(\mathbf{X}), \; \operatorname{AvgPool}_{2\times2}(\mathbf{X}), \; \operatorname{AvgPool}_{4\times4}(\mathbf{X})]$$

The output has size $(1 + 4 + 16) \times C = 21C$ regardless of input size. SPP is the precursor to multi-scale attention in Transformers.

---

## 7. Normalisation in CNNs

### 7.1 Why Normalisation Matters

Without normalisation, the distribution of inputs to each layer changes during training as the parameters of the previous layers are updated — a phenomenon called **internal covariate shift** (Ioffe & Szegedy, 2015). This causes:

1. **Vanishing/exploding gradients:** If activations are very large or very small, the gradients through sigmoid/tanh saturate.
2. **Slow training:** The learning rate must be set very conservatively to prevent instability.
3. **Bad initialisation sensitivity:** The network is very sensitive to weight initialisation.

Normalisation layers address this by explicitly controlling the distribution of activations.

### 7.2 Batch Normalisation: Forward Pass

Given a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$ of activations for a single feature:

**Step 1: Batch mean:**
$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**Step 2: Batch variance:**
$$\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

**Step 3: Normalise:**
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

**Step 4: Scale and shift:**
$$y_i = \gamma \hat{x}_i + \beta$$

The learnable parameters $\gamma$ (scale) and $\beta$ (shift) allow the network to **undo** the normalisation if needed — without them, BatchNorm would force every feature to have mean 0 and variance 1, losing representational power.

**For convolutional layers:** BatchNorm is applied per-channel, computing statistics over $(N, H, W)$ for each channel $c$:

$$\mu_c = \frac{1}{N H W} \sum_{n, h, w} X[n, c, h, w]$$

**At test time:** The training statistics $\mu_\mathcal{B}$, $\sigma_\mathcal{B}^2$ are not available (batch size 1 at deployment). Instead, running exponential moving averages are maintained:

$$\mu_{\text{running}} \leftarrow \alpha \, \mu_{\text{running}} + (1 - \alpha) \, \mu_\mathcal{B}$$

### 7.3 Batch Normalisation: Backward Pass

The backward pass through BatchNorm requires applying the chain rule through all four steps. Define $\delta = \partial \mathcal{L} / \partial \mathbf{y}$ as the upstream gradient.

**Gradient w.r.t. $\gamma$ and $\beta$:**
$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} \delta_i \hat{x}_i, \qquad \frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{m} \delta_i$$

**Gradient w.r.t. $\hat{x}_i$:**
$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \delta_i \cdot \gamma$$

**Gradient w.r.t. $\sigma_\mathcal{B}^2$:**
$$\frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} = -\frac{1}{2} \sum_i \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot (x_i - \mu_\mathcal{B}) \cdot (\sigma_\mathcal{B}^2 + \epsilon)^{-3/2}$$

**Gradient w.r.t. input $x_i$:**
$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{m\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} \left[ m \cdot \frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \sum_{j} \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \hat{x}_i \sum_{j} \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j \right]$$

The key property: the gradient from a single sample $x_i$ flows to all other samples in the batch through the shared $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ statistics. **This is why batch size matters for BatchNorm** — with small batches, the statistics are noisy.

### 7.4 Layer, Instance, and Group Normalisation

Different normalisation strategies differ in which axes they compute statistics over:

```text
NORMALISATION STRATEGIES (N=batch, C=channel, H=height, W=width)
════════════════════════════════════════════════════════════════════════

  Tensor: [N, C, H, W]

  Batch Norm:    normalise over (N, H, W) for each C
                 → works well for large batches, fails for batch size 1

  Layer Norm:    normalise over (C, H, W) for each N
                 → batch-independent, used in Transformers

  Instance Norm: normalise over (H, W) for each (N, C)
                 → style transfer, removes style while preserving content

  Group Norm:    normalise over (C/G, H, W) for each (N, G)
                 → G groups of channels, works at batch size 1

════════════════════════════════════════════════════════════════════════
```

| Norm type     | Best for                            | Fails when                            |
| ------------- | ----------------------------------- | ------------------------------------- |
| Batch Norm    | Large-batch supervised CNN training | Small batch (detection, segmentation) |
| Layer Norm    | Transformers, RNNs, NLP             | Spatial CNNs (less spatial meaning)   |
| Instance Norm | Style transfer, image generation    | Classification (removes useful info)  |
| Group Norm    | Object detection, small batch       | Must choose group size $G$            |

---

## 8. Backpropagation Through Convolution

### 8.1 Chain Rule on the Sliding Window

Consider the forward pass for a single-channel 2D convolution (stride 1, valid padding):

$$Y[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X[i+m, j+n] \cdot K[m, n] + b$$

Given the loss gradient $\delta Y[i, j] = \partial \mathcal{L} / \partial Y[i, j]$, we need:

1. $\partial \mathcal{L} / \partial X[p, q]$ — for the previous layer
2. $\partial \mathcal{L} / \partial K[m, n]$ — for weight updates
3. $\partial \mathcal{L} / \partial b$ — for bias update

### 8.2 Gradient w.r.t. Input (Transposed Convolution)

Input $X[p, q]$ contributes to output $Y[i, j]$ whenever $0 \le p - i < k$ and $0 \le q - j < k$, i.e., when the kernel centered at $(i, j)$ overlaps position $(p, q)$.

Applying the chain rule:

$$\frac{\partial \mathcal{L}}{\partial X[p, q]} = \sum_{i, j} \frac{\partial \mathcal{L}}{\partial Y[i, j]} \cdot \frac{\partial Y[i, j]}{\partial X[p, q]} = \sum_{i, j} \delta Y[i, j] \cdot K[p-i, q-j]$$

where the sum is over valid $(i, j)$ such that $0 \le p - i < k$ and $0 \le q - j < k$.

This is a convolution of $\delta Y$ with the kernel **rotated 180°**:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \delta\mathbf{Y} \star \mathbf{K}_{180°}$$

**This is exactly what transposed convolution (ConvTranspose2d) computes** — the backward pass of convolution is itself a convolution operation. This is why transposed convolution is sometimes called "deconvolution" (though that term is technically incorrect).

### 8.3 Gradient w.r.t. Kernel

$K[m, n]$ contributes to $Y[i, j]$ for every valid $(i, j)$:

$$\frac{\partial \mathcal{L}}{\partial K[m, n]} = \sum_{i, j} \frac{\partial \mathcal{L}}{\partial Y[i, j]} \cdot X[i+m, j+n] = \sum_{i, j} \delta Y[i, j] \cdot X[i+m, j+n]$$

This is a cross-correlation of the input $\mathbf{X}$ with the gradient $\delta\mathbf{Y}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \mathbf{X} \star \delta\mathbf{Y}$$

**Key insight:** Both gradients are convolutions. The entire backward pass through a convolutional layer consists of two convolution operations — meaning backprop through a CNN requires the same type of hardware optimisation as the forward pass.

**Gradient w.r.t. bias:**
$$\frac{\partial \mathcal{L}}{\partial b} = \sum_{i, j} \delta Y[i, j]$$

The bias gradient is simply the sum of all output gradients (consistent with the bias contributing equally to every output position).

### 8.4 Efficient Implementation via im2col / GEMM

Direct convolution iterates over spatial positions in nested loops — slow on GPUs. The standard approach converts convolution to matrix multiplication via **im2col** ("image to column"):

**im2col:** Reshape the input so that each $k \times k$ patch at every valid position becomes a column of a matrix $\mathbf{X}_{\text{col}} \in \mathbb{R}^{(C_{\text{in}} k^2) \times (H' W')}$.

**GEMM:** Reshape the kernel as $\mathbf{W}_{\text{mat}} \in \mathbb{R}^{C_{\text{out}} \times (C_{\text{in}} k^2)}$. The output is:

$$\mathbf{Y}_{\text{mat}} = \mathbf{W}_{\text{mat}} \cdot \mathbf{X}_{\text{col}} \in \mathbb{R}^{C_{\text{out}} \times (H' W')}$$

Reshape $\mathbf{Y}_{\text{mat}}$ back to $\mathbb{R}^{C_{\text{out}} \times H' \times W'}$.

**Why this works:** Matrix multiplication (GEMM) is extremely optimised on GPUs (cuBLAS achieves near-theoretical peak FLOPS). Expressing convolution as GEMM leverages decades of BLAS optimisation.

**Cost:** im2col requires $C_{\text{in}} k^2 H' W'$ memory (each patch stored separately), which can be $k^2$ times the input memory. Alternative: **Winograd's algorithm** reduces the arithmetic complexity without the memory overhead.

---

## 9. Depthwise Separable Convolution

### 9.1 Standard vs Depthwise Separable

**Standard convolution:** One $C_{\text{out}} \times C_{\text{in}} \times k \times k$ weight tensor performs spatial filtering and channel mixing simultaneously.

**Depthwise separable convolution** (Chollet, 2017; Howard et al., 2017) factorises this into two steps:

**Step 1 — Depthwise convolution:** Apply one $k \times k$ filter per input channel independently. Each channel is filtered separately (no channel mixing):

$$Y_{\text{dw}}[c, i, j] = \sum_{m, n} X[c, \, i \cdot s + m, \, j \cdot s + n] \cdot K_{\text{dw}}[c, m, n]$$

Weight tensor: $\mathbf{K}_{\text{dw}} \in \mathbb{R}^{C_{\text{in}} \times k \times k}$. Each input channel has its own spatial filter.

**Step 2 — Pointwise convolution:** Apply $C_{\text{out}}$ filters of size $1 \times 1$ to mix channels:

$$Y_{\text{pw}}[f, i, j] = \sum_{c=1}^{C_{\text{in}}} Y_{\text{dw}}[c, i, j] \cdot K_{\text{pw}}[f, c]$$

Weight tensor: $\mathbf{K}_{\text{pw}} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$. This is a linear recombination of channels at each spatial position.

### 9.2 Parameter Reduction

**Standard:** $C_{\text{out}} \times C_{\text{in}} \times k^2$ parameters.

**Depthwise separable:** $C_{\text{in}} \times k^2 + C_{\text{out}} \times C_{\text{in}}$ parameters.

**Reduction factor:**

$$\frac{C_{\text{in}} k^2 + C_{\text{out}} C_{\text{in}}}{C_{\text{out}} C_{\text{in}} k^2} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}$$

For typical values ($C_{\text{out}} = 256$, $k = 3$): $\frac{1}{256} + \frac{1}{9} \approx \frac{1}{9}$ — a **9× reduction in parameters**.

### 9.3 FLOPs Reduction

**Standard FLOPs:** $2 C_{\text{out}} C_{\text{in}} k^2 H' W'$ (factor 2 for multiply-add).

**Depthwise separable FLOPs:** $2 C_{\text{in}} k^2 H' W' + 2 C_{\text{out}} C_{\text{in}} H' W'$.

**Reduction factor** (same as parameters): $\approx \frac{1}{k^2}$ for large $C_{\text{out}}$.

This is the key insight of **MobileNet** and **Xception**: equivalent representational power (empirically) at a fraction of the computational cost.

### 9.4 MobileNet and the Efficiency Frontier

**MobileNet V1** (Howard et al., 2017) replaces all standard $3 \times 3$ convolutions with depthwise separable convolutions. Result: 8-9× fewer FLOPs with only ~1% ImageNet accuracy drop.

**MobileNet V2** adds **inverted residuals** (bottleneck blocks where the expansion is in the pointwise step) and **linear bottlenecks** (no ReLU at the bottleneck to preserve the linear subspace structure).

**The width multiplier** $\alpha \in (0, 1]$ scales all channel counts by $\alpha$, reducing FLOPs by $\alpha^2$. This provides a smooth accuracy-efficiency tradeoff along the Pareto frontier.

---

## 10. Dilated (Atrous) Convolution

### 10.1 Definition and Motivation

Dilated convolution inserts $d-1$ zeros between each kernel element (dilation rate $d$):

$$Y[i, j] = \sum_{m, n} X[i + m \cdot d, \; j + n \cdot d] \cdot K[m, n]$$

A $3 \times 3$ kernel with $d = 1$ (standard): covers a $3 \times 3$ area.
A $3 \times 3$ kernel with $d = 2$: covers a $5 \times 5$ area.
A $3 \times 3$ kernel with $d = 4$: covers a $9 \times 9$ area.

The **effective kernel size** is $k + (k-1)(d-1) = d(k-1) + 1$.

**Motivation:** Dense prediction tasks (semantic segmentation, depth estimation) require:

1. **Large receptive fields** — to understand global context.
2. **High spatial resolution** — to make predictions at the pixel level.

These are in tension: striding downsamples the resolution. Dilated convolutions expand the receptive field without downsampling.

### 10.2 Receptive Field of Dilated Stacks

A classic scheme (used in DeepLab, WaveNet, TCN) stacks convolutions with exponentially increasing dilation:

$$d = 1, 2, 4, 8, 16, 32, \ldots, 2^{L-1}$$

The receptive field after $L$ layers with a $3 \times 3$ kernel:

$$\text{RF}_L = 1 + \sum_{l=0}^{L-1} 2 \cdot 2^l = 1 + 2(2^L - 1) = 2^{L+1} - 1$$

After $L = 7$ layers: RF = $2^8 - 1 = 255$. After $L = 10$: RF = $2047$. The receptive field grows **exponentially** with depth — much faster than the linear growth of standard convolutions ($L(k-1) + 1$).

```text
DILATED CONVOLUTION RECEPTIVE FIELDS (3x3 kernel)
════════════════════════════════════════════════════════════════════════

  d=1: covers  3x3  (standard)
  d=2: covers  5x5  (with holes)
  d=4: covers  9x9  (with holes)
  d=8: covers 17x17 (with holes)

  Stack (d=1,2,4,8):

  Layer 1 (d=1): RF = 3
  Layer 2 (d=2): RF = 3 + 4 = 7
  Layer 3 (d=4): RF = 7 + 8 = 15
  Layer 4 (d=8): RF = 15 + 16 = 31

  After 4 layers: 31x31 receptive field with only 4×9=36 parameters!

════════════════════════════════════════════════════════════════════════
```

### 10.3 Dilated Causal Convolution (WaveNet)

**WaveNet** (van den Oord et al., 2016) generates raw audio waveforms using **dilated causal convolutions**. "Causal" means the output at time $t$ depends only on inputs at times $\le t$ (no future information leakage):

$$Y[t] = \sum_{m=0}^{k-1} X[t - m \cdot d] \cdot K[m]$$

The kernel looks backward in time. Stacking dilated causal convolutions with doublings ($d = 1, 2, 4, \ldots, 512$) achieves a receptive field of 1024 timesteps in just 10 layers. At 16kHz audio, this covers 64ms — enough to model short-term dependencies in speech.

**Connection to Transformers:** WaveNet's dilated causal convolution and GPT's causal self-attention solve the same problem — autoregressive generation with large context — using different mechanisms. CNNs (WaveNet) are faster at training (fully parallel) but have fixed receptive fields. Transformers have $O(n^2)$ attention cost but full context.

---

## 11. Transposed Convolution

### 11.1 Definition as Adjoint Operator

The **transposed convolution** (ConvTranspose2d) is the mathematical adjoint (transpose) of the standard convolution operator. If standard convolution maps $\mathbb{R}^{H \times W} \to \mathbb{R}^{H' \times W'}$, the transposed convolution maps $\mathbb{R}^{H' \times W'} \to \mathbb{R}^{H \times W}$ — an **upsampling** operation.

In the Toeplitz matrix view: if $\mathbf{y} = \mathbf{C} \mathbf{x}$, then the transposed convolution computes $\mathbf{x} = \mathbf{C}^\top \mathbf{y}$. The transpose of a Toeplitz matrix is another Toeplitz matrix (with the kernel flipped).

**Computation:** For each element $y[i, j]$ in the input, "stamp" the kernel multiplied by $y[i, j]$ onto the output, with stride $s$ spacing. Overlapping regions are summed.

**Use cases:**

1. Upsampling in encoder-decoder networks (U-Net, FCN for semantic segmentation).
2. Generative networks (VAE decoders, GANs) that need to upsample latent codes to images.
3. Visualising learned features by inverting the network.

**Transposed convolution is NOT a true inverse:** $\mathbf{C}^\top \mathbf{C} \neq \mathbf{I}$ in general. It is an adjoint, not an inverse. The output does not reconstruct the original input exactly.

### 11.2 Checkerboard Artifacts

When stride > 1, the transposed convolution kernel stamps overlap unevenly — some output pixels receive contributions from more kernel stamps than others, creating a periodic "checkerboard" pattern in the output.

**Root cause:** For stride $s = 2$ and kernel $k = 3$, the output coverage pattern is:

- Position 0: covered by 2 stamps
- Position 1: covered by 1 stamp
- Position 2: covered by 2 stamps
- Position 3: covered by 1 stamp ...

The alternating coverage creates a frequency-$s$ artifact. For $k = 4$ and $s = 2$: every position covered equally — no checkerboard! The condition for uniform coverage: $k$ divisible by $s$.

**Solutions:**

1. Use kernel sizes divisible by stride: $k = 4, s = 2$ or $k = 6, s = 3$.
2. Odena et al. (2016) showed that **bilinear upsampling followed by a standard convolution** (Section 11.3) eliminates checkerboard artifacts entirely.

### 11.3 Alternative: Bilinear Upsampling + Conv

The standard alternative to transposed convolution:

1. **Bilinear upsample** (factor of 2, no learnable parameters): each pixel is interpolated from its neighbours.
2. **Standard convolution** (learnable): refines the upsampled feature map.

This combination is now preferred in most segmentation architectures (DeepLab V3+, HRNet) because it avoids checkerboard artifacts and is often faster than transposed convolution.

---

## 12. 1D Convolution for Sequences

### 12.1 Text and Audio Processing

**1D convolution** applies a kernel of length $k$ to a sequence $\mathbf{x} \in \mathbb{R}^{C_{\text{in}} \times L}$ (length $L$, $C_{\text{in}}$ channels):

$$Y[f, t] = \sum_{c=0}^{C_{\text{in}}-1} \sum_{m=0}^{k-1} X[c, t+m] \cdot K[f, c, m]$$

This is exactly analogous to 2D convolution but with one spatial dimension.

**For text:** Input is a sequence of word embeddings $\mathbf{x} \in \mathbb{R}^{d \times L}$ where $d$ is the embedding dimension. 1D convolution with kernel size $k$ detects $k$-gram patterns. TextCNN (Kim, 2014) uses multiple kernel sizes (3, 4, 5) in parallel to capture different n-gram features.

**For audio:** Raw waveform at 16kHz, input $\in \mathbb{R}^{1 \times 16000}$. 1D convolution detects local acoustic patterns. WaveNet, SoundStream, and Encodec all use 1D convolutions.

### 12.2 Causal Convolution

A standard 1D convolution looks at context $[t, t+k-1]$ — both past and future relative to position $t$. For autoregressive generation, we need **causal convolution** that only looks at $[t-k+1, t]$:

$$Y[f, t] = \sum_{c, m} X[c, t - m] \cdot K[f, c, m]$$

Implementation: pad the input with $k-1$ zeros on the **left** (past), none on the right. This ensures $Y[t]$ depends only on $X[t], X[t-1], \ldots, X[t-k+1]$.

**Temporal Convolutional Networks (TCN, Bai et al., 2018)** combine dilated causal convolutions with residual connections, achieving performance competitive with LSTMs on sequential prediction tasks while being fully parallelisable.

### 12.3 Connection to Self-Attention

Both 1D convolution and self-attention process sequences. The key mathematical differences:

|                          | 1D Convolution   | Self-Attention       |
| ------------------------ | ---------------- | -------------------- |
| Context                  | Fixed window $k$ | Full sequence $L$    |
| Weights                  | Fixed (kernel)   | Input-dependent      |
| Complexity               | $O(L k C)$       | $O(L^2 C)$           |
| Translation equivariance | Yes (exact)      | Approximate (via PE) |
| Long-range dependencies  | Limited by $k$   | Exact                |

**Convolution as local, static attention:** The convolution kernel $K[m]$ is the attention weight for position offset $m$, applied identically at every position. Self-attention generalises this: the weight for position $j$ attending to position $i$ is $\text{softmax}(\mathbf{q}_j^\top \mathbf{k}_i / \sqrt{d_k})$ — **content-dependent and position-dependent**.

**Hybrid architectures** like ConvNeXt and local attention Transformers combine both: convolutions for local inductive biases, attention for global context.

---

## 13. CNN Architectures

### 13.1 AlexNet to VGG: Depth and Uniformity

**AlexNet** (Krizhevsky et al., 2012): 5 conv layers (11×11, 5×5, 3×3) + 3 FC layers. Used ReLU, dropout, data augmentation, and multi-GPU training. Top-5 error: 15.3% (vs 26.2% for second place).

**VGGNet** (Simonyan & Zisserman, 2014): All $3 \times 3$ convolutions, very deep (VGG-16: 16 weight layers, VGG-19: 19 layers). Key insight: two stacked $3 \times 3$ conv layers have the same receptive field as one $5 \times 5$ layer, but with fewer parameters ($2 \times 9C^2 = 18C^2 < 25C^2$) and more non-linearities. VGG is still widely used as a feature extractor.

### 13.2 ResNet: Residual Learning

**Problem:** Deep networks (>20 layers) were harder to train than shallow ones — not due to overfitting but **optimisation difficulty** (degradation problem). A 56-layer network had higher training error than a 20-layer network.

**ResNet** (He et al., 2015) adds **skip connections**:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

The residual $\mathcal{F}(\mathbf{x})$ is what the layers need to learn, rather than the full mapping $\mathbf{y}$. If the identity mapping is optimal, the network just pushes $\mathcal{F}(\mathbf{x}) \to \mathbf{0}$ — much easier than learning the identity through a stack of layers.

**Mathematical justification:** Skip connections create gradient highways — the gradient can flow directly through the skip connection without passing through the residual layers: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}\left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)$. The $+1$ ensures a non-vanishing gradient path.

**Bottleneck block** (ResNet-50): $1\times1$ (reduce channels) → $3\times3$ → $1\times1$ (restore channels). The $1\times1$ convolutions reduce computational cost while maintaining expressiveness.

### 13.3 DenseNet: Dense Connections

**DenseNet** (Huang et al., 2017) connects each layer to all subsequent layers:

$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{l-1}])$$

where $[\cdot, \cdot]$ denotes channel concatenation. Every layer receives feature maps from all preceding layers.

**Benefits:**

1. **Feature reuse:** Earlier features remain accessible throughout the network.
2. **Gradient flow:** Direct connections to every layer prevent gradient vanishing.
3. **Efficiency:** Growth rate (new channels per layer) can be very small ($k = 32$) because features accumulate via concatenation.

### 13.4 EfficientNet: Compound Scaling

**EfficientNet** (Tan & Le, 2019) systematically scales width, depth, and resolution together using a compound coefficient $\phi$:

$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (approximately doubling FLOPs per unit of $\phi$). The optimal coefficients $\alpha = 1.2, \beta = 1.1, \gamma = 1.15$ are found by grid search with $\phi = 1$.

**Why compound scaling works:** Scaling only depth (more layers) eventually saturates — deeper networks need wider layers and higher resolution to be effective. Scaling all three dimensions together in a principled ratio avoids these bottlenecks.

### 13.5 Inception: Multi-Scale Kernels

**Inception** (Szegedy et al., 2014) applies multiple kernel sizes in parallel and concatenates the results:

$$\text{output} = \operatorname{concat}[\text{Conv}_{1\times1}, \; \text{Conv}_{3\times3}, \; \text{Conv}_{5\times5}, \; \text{MaxPool}_{3\times3}]$$

Different scales capture different types of structure: $1 \times 1$ for channel mixing, $3 \times 3$ for local patterns, $5 \times 5$ for larger patterns. The multi-scale approach increases expressiveness without choosing a single kernel size.

**$1 \times 1$ convolutions** ("network-in-network") are used before the larger kernels to reduce channel dimensionality (bottleneck), dramatically reducing FLOPs. A $1 \times 1$ conv on a 256-channel input with 32 output channels reduces a subsequent $3 \times 3$ conv's FLOPs by $256/32 = 8\times$.

---

## 14. CNNs and Vision Transformers

### 14.1 Patch Embedding as Strided Convolution

**Vision Transformer (ViT)** (Dosovitskiy et al., 2020) divides an image into non-overlapping $P \times P$ patches and linearly projects each to a $d$-dimensional embedding:

$$\mathbf{z}^{(0)} = [\mathbf{x}^1_p E; \; \mathbf{x}^2_p E; \; \ldots; \; \mathbf{x}^{N}_p E] + \mathbf{E}_{\text{pos}}$$

where $E \in \mathbb{R}^{(P^2 C) \times d}$ is the projection matrix.

**This is exactly a strided convolution:**

$$\text{PatchEmbed} = \operatorname{Conv2d}(C_{\text{in}}, d, \text{kernel\_size}=P, \text{stride}=P)$$

A $P \times P$ convolution with stride $P$ produces one output per non-overlapping patch. The PyTorch ViT implementation uses this equivalence — `nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)`.

### 14.2 Inductive Biases Compared

| Inductive Bias           | CNN                           | ViT                                  |
| ------------------------ | ----------------------------- | ------------------------------------ |
| Translation equivariance | Built-in (exact)              | Learned (approximate via PE)         |
| Locality                 | Built-in (local kernels)      | None (global attention)              |
| Parameter sharing        | Across all spatial positions  | Only within attention heads          |
| Rotation equivariance    | None                          | None                                 |
| Scale invariance         | Approximate (via pooling)     | None                                 |
| Data requirement         | Low (works on small datasets) | High (needs large-scale pretraining) |

**When CNNs win:** Small datasets, tasks requiring precise spatial localisation (detection, segmentation), mobile/edge deployment.

**When ViTs win:** Large-scale pretraining, long-range dependency tasks, multimodal learning (CLIP uses a ViT image encoder), transfer learning flexibility.

### 14.3 ConvNeXt: Modernising the CNN

**ConvNeXt** (Liu et al., 2022) takes ResNet-50 and applies all the design decisions of ViT-style training to arrive at a pure CNN that matches ViT performance:

| Modification                | From ViT design                 | Effect |
| --------------------------- | ------------------------------- | ------ |
| $7 \times 7$ depthwise conv | Large receptive field per block | +0.1%  |
| Inverted bottleneck         | Wider hidden dim                | +0.4%  |
| Group norm (1 group = LN)   | Layer Norm-like                 | +0.1%  |
| GELU activation             | Transformer activation          | +0.0%  |
| Fewer normalisation layers  | ViT uses only pre-norm          | +0.1%  |
| Separate downsampling       | ViT has explicit stem           | +0.4%  |

The message: the performance gap between CNNs and ViTs was largely due to **training procedures and design conventions**, not the fundamental architecture. A well-trained CNN with modern design choices is competitive with a ViT.

---

## 15. Common Mistakes

1. **Confusing convolution and cross-correlation.** PyTorch's `Conv2d` computes cross-correlation (no kernel flip). For handcrafted filters from signal processing literature, flip the kernel before using it in PyTorch.

2. **Wrong output size calculation.** Always use the formula $\lfloor(H + 2p - d(k-1) - 1)/s\rfloor + 1$. Common mistake: forgetting the dilation term $d(k-1)$, or computing $\lfloor(H - k)/s\rfloor + 1$ (valid, no dilation) when dilation is non-zero.

3. **Misunderstanding weight sharing.** A convolutional layer with $C_{\text{out}} = 64$ filters does NOT apply 64 different operations to the same position — it applies 64 different kernels, each producing one feature map. The 64 filters detect 64 different patterns.

4. **BatchNorm at test time.** Never compute batch statistics during inference — use the running statistics. Forgetting `model.eval()` in PyTorch leaves BatchNorm in training mode (uses batch statistics), giving different results for different batch sizes at inference.

5. **BatchNorm placement.** The original paper places BatchNorm after Conv and before activation: Conv → BN → ReLU. Modern practice (pre-activation ResNet, Transformers) often uses BN/LN before the main operation. These are NOT equivalent.

6. **Receptive field vs effective receptive field.** The theoretical receptive field grows linearly with depth, but the effective receptive field (where most gradient signal comes from) is much smaller — empirically Gaussian-shaped and much smaller than theoretical. A 50-layer network does NOT uniformly aggregate the entire theoretical RF.

7. **Using transposed convolution for upsampling without controlling artifacts.** Always check: is $k$ divisible by $s$? If not, use bilinear upsample + conv instead.

8. **Forgetting to account for dilation in receptive field calculations.** A dilated convolution with $d = 4$ and $k = 3$ has effective kernel size $4(3-1) + 1 = 9$, not 3.

---

## 16. Exercises

1. **Implement 2D convolution from scratch.** Write a NumPy function `conv2d(X, K, stride=1, padding=0)` that computes 2D cross-correlation. Verify against `scipy.signal.correlate2d` and against `torch.nn.functional.conv2d`.

2. **Output size formula.** For each of the following configurations, compute the output size: (a) $H=32, k=3, p=0, s=1$; (b) $H=32, k=3, p=1, s=1$; (c) $H=32, k=3, p=0, s=2$; (d) $H=64, k=5, p=2, s=2$; (e) $H=28, k=3, p=0, s=1, d=2$.

3. **Backpropagation through convolution.** For a 1D convolution $\mathbf{y} = \mathbf{k} * \mathbf{x}$ with $\mathbf{x} = [1, 2, 3, 4, 5]$ and $\mathbf{k} = [1, 0, -1]$: (a) compute the output; (b) given $\partial \mathcal{L}/\partial \mathbf{y} = [1, 1, 1]$, compute $\partial \mathcal{L}/\partial \mathbf{x}$ using the Toeplitz transpose formula; (c) verify by finite differences.

4. **BatchNorm forward and backward.** Implement BatchNorm from scratch. Given input $\mathbf{x} = [2, 4, 6, 8]$ with $\gamma = 1.0$, $\beta = 0$, $\epsilon = 10^{-5}$: compute the normalised output. Then verify the backward pass matches autograd.

5. **Depthwise separable parameter count.** For a layer with $C_{\text{in}} = 512$, $C_{\text{out}} = 512$, $k = 3$: compute the exact parameter counts for (a) standard convolution, (b) depthwise separable convolution. What is the ratio?

6. **Dilated receptive field.** For a stack of 5 dilated $3 \times 3$ convolutions with rates $d = 1, 2, 4, 8, 16$: draw the receptive field of the final neuron (which input pixels affect it?). What is the total RF size? How many parameters are used?

7. **Transposed convolution.** For a 1D input $\mathbf{y} = [1, 2, 3]$ and kernel $\mathbf{k} = [1, 2]$ with stride $s = 2$: manually compute the transposed convolution output. Verify it is the backward pass of a standard convolution.

8. **ResNet skip connection gradient.** In a residual block $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$, derive $\partial \mathcal{L}/\partial \mathbf{x}$ using the chain rule. Show that the gradient always contains at least one direct path from $\partial \mathcal{L}/\partial \mathbf{y}$ to $\partial \mathcal{L}/\partial \mathbf{x}$.

9. **Patch embedding as convolution.** Show mathematically that a $P \times P$ non-overlapping patch projection with weight matrix $E \in \mathbb{R}^{(P^2 C) \times d}$ is exactly equivalent to `Conv2d(C, d, kernel_size=P, stride=P)`. Verify in PyTorch by comparing the outputs.

10. **Convolution theorem.** Given a $256 \times 256$ input and a $32 \times 32$ kernel, estimate the speedup from FFT-based convolution over direct convolution. Assume direct cost $O(n^2 k^2)$ and FFT cost $O(n^2 \log n)$ (where $n = 256$). For a $3 \times 3$ kernel, which is faster?

---

## 17. Why This Matters for AI (2026)

**CNNs are not dead.** Despite the ViT revolution, CNNs remain dominant in:

- **Mobile and edge inference** (MobileNet, EfficientNet-Lite, TFLite models)
- **Medical imaging** (limited data, spatial precision required)
- **Object detection** (YOLO uses CNNs; EfficientDet, DETR use CNN backbones)
- **Video processing** (3D CNNs, temporal shift modules)
- **Signal processing** (WaveNet, Encodec, SoundStream)

**Convolution in Transformers.** The connection between convolution and attention is deep and productive:

- Patch embedding = strided convolution.
- Local attention = convolution with learned, content-dependent weights.
- Convolutional position encodings (e.g., CPVT, LocalViT) inject CNN inductive biases into ViTs.
- **ConvNeXt** shows that a pure CNN with modern training can match ViT — suggesting the gap was always about training, not architecture.

**Depthwise separable convolutions are everywhere.** MobileNet's depthwise separable design directly influenced:

- **Xception:** Extends the idea to all layers.
- **Transformer FFN layers:** Can be viewed as $1 \times 1$ convolutions (mixing channels at each position).
- **EfficientNet:** Uses mobile inverted bottlenecks throughout.
- **On-device ML:** The 9× FLOPs reduction is critical for deployment on phones and embedded systems.

**Dilated convolutions power sequence modelling.** WaveNet's dilated causal convolutions are the direct precursor to temporal convolutional networks (TCN) and influenced the design of linear attention models that achieve sub-quadratic complexity.

---

## 18. Conceptual Bridge

**Where we have been:** This section developed the mathematics of convolution from the continuous integral definition through discrete 2D cross-correlation, the Toeplitz/circulant linear algebra, backpropagation, depthwise separable and dilated variants, and major architectural innovations.

**Where this connects:**

| Concept                   | Connects to                                        |
| ------------------------- | -------------------------------------------------- |
| Toeplitz matrix structure | Linear algebra, eigenvectors (Chapter 02)          |
| Convolution theorem (FFT) | Fourier analysis, signal processing                |
| Backprop through conv     | Automatic differentiation (Section 14-02)          |
| BatchNorm backward        | Computational graphs, chain rule                   |
| Residual connections      | Gradient flow, vanishing gradients (Section 14-02) |
| Translation equivariance  | Group theory, symmetry                             |
| Patch embedding as conv   | Transformer architecture (Section 14-05)           |
| Dilated causal conv       | WaveNet, sequence modelling (Section 14-04)        |
| Depthwise separable       | Tensor decomposition, matrix rank                  |
| EfficientNet scaling      | Pareto optimality, constrained optimisation        |

**What comes next:** Section 14-09 (Attention Mechanisms) generalises convolution: instead of fixed local kernels, attention computes input-dependent, global weights. Understanding convolution's weight-sharing, locality, and equivariance makes the design choices of Transformers (why positional encoding is needed, why local attention helps, why patch size matters) much clearer.

---

## References

1. LeCun, Y. et al. (1998). Gradient-based learning applied to document recognition. _Proceedings of the IEEE_, 86(11), 2278-2324.
2. Krizhevsky, A., Sutskever, I. & Hinton, G. (2012). ImageNet classification with deep convolutional neural networks. _NeurIPS_.
3. Simonyan, K. & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. _ICLR_.
4. Szegedy, C. et al. (2015). Going deeper with convolutions. _CVPR_.
5. He, K. et al. (2016). Deep residual learning for image recognition. _CVPR_.
6. Ioffe, S. & Szegedy, C. (2015). Batch normalisation: Accelerating deep network training. _ICML_.
7. Howard, A. et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision. _arXiv:1704.04861_.
8. Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. _CVPR_.
9. van den Oord, A. et al. (2016). WaveNet: A generative model for raw audio. _arXiv:1609.03499_.
10. Huang, G. et al. (2017). Densely connected convolutional networks. _CVPR_.
11. Tan, M. & Le, Q. (2019). EfficientNet: Rethinking model scaling for CNNs. _ICML_.
12. Dosovitskiy, A. et al. (2021). An image is worth 16x16 words. _ICLR_.
13. Liu, Z. et al. (2022). A ConvNet for the 2020s. _CVPR_.
14. Odena, A. et al. (2016). Deconvolution and checkerboard artifacts. _Distill_.
15. Luo, W. et al. (2017). Understanding the effective receptive field in DNNs. _NeurIPS_.
