[<- Back to Fourier Analysis](../README.md) | [Previous: Fourier Series <-](../01-Fourier-Series/notes.md) | [Next: DFT and FFT ->](../03-Discrete-Fourier-Transform-and-FFT/notes.md)

---

# Fourier Transform

> _"The Fourier transform is a mathematical prism: it refracts a signal into its constituent frequencies, revealing structure invisible in time."_
> - paraphrasing Norbert Wiener

## Overview

The Fourier Transform extends Fourier series to non-periodic signals by taking the period to infinity. Where a Fourier series decomposes a periodic function into a discrete sum of harmonics, the Fourier Transform decomposes an aperiodic function into a continuous integral over all frequencies. The result is the **spectrum** of a signal - a complete description of which frequencies are present, at what amplitude, and with what phase.

This extension is not merely a generalization for mathematical completeness. It is the foundation of modern signal processing, quantum mechanics, partial differential equations, and - crucially for this curriculum - large language models, neural operators, and kernel methods. The Fourier Transform converts convolution (an expensive $O(N^2)$ operation) into pointwise multiplication, converts differentiation into algebraic multiplication by frequency, and provides the mathematical basis for the uncertainty principle that constrains every time-frequency representation from spectrograms to attention windows.

We develop the theory in full: the $L^1$ definition and its limitations, the $L^2$ extension via Plancherel's theorem, the complete set of operational properties, the inversion theorem, the Heisenberg uncertainty principle with proof, the distribution-theoretic extension to handle Dirac deltas and periodic signals, and a thorough treatment of AI applications including FNet, random Fourier features, spectral normalization, and the Fourier Neural Operator.

## Prerequisites

- **Fourier Series** - complex Fourier coefficients, $L^2[-\pi,\pi]$, Parseval's identity, orthonormality of $\{e^{inx}\}$ ([Section 20-01 Fourier Series](../01-Fourier-Series/notes.md))
- **Complex analysis** - $e^{i\theta} = \cos\theta + i\sin\theta$, complex modulus, argument, conjugate ([Section 01 Mathematical Foundations](../../01-Mathematical-Foundations/README.md))
- **Integration** - improper integrals, Fubini's theorem, dominated convergence ([Section 04 Calculus Fundamentals](../../04-Calculus-Fundamentals/README.md))
- **$L^2$ function spaces** - inner product, norm, completeness, Hilbert space ([Section 12-02 Hilbert Spaces](../../12-Functional-Analysis/02-Hilbert-Spaces/notes.md))
- **Linear algebra** - unitary operators, isometries ([Section 03 Advanced Linear Algebra](../../03-Advanced-Linear-Algebra/README.md))

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive derivations: FT of standard functions, properties, uncertainty principle, Plancherel, FNet, random Fourier features |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems from computing transforms to proving inversion to implementing kernel approximation |

## Learning Objectives

After completing this section, you will:

1. Derive the Fourier Transform formula as the $T \to \infty$ limit of the Fourier series
2. Compute the Fourier Transform of the Gaussian, rectangular pulse, Lorentzian, and triangle function from the definition
3. Apply all ten standard properties (linearity, shift, scaling, differentiation, conjugation, duality, convolution) to evaluate transforms without integration
4. State and prove Plancherel's theorem and use it to evaluate $L^2$ norms in the frequency domain
5. State and prove the Heisenberg uncertainty principle and identify the Gaussian as the unique extremal function
6. Define the Schwartz space and tempered distributions; compute the Fourier Transform of $\delta(t)$, $1$, $\cos(2\pi\xi_0 t)$, and the Dirac comb
7. Explain the Poisson summation formula and its role in sampling theory
8. Describe how FNet (Lee-Thorp et al., 2022) replaces self-attention with Fourier transforms and analyze its computational tradeoffs
9. Implement random Fourier features (Rahimi & Recht, 2007) and verify the kernel approximation guarantee
10. Explain spectral normalization (Miyato et al., 2018) and how it uses the FT to enforce Lipschitz constraints on discriminators
11. Distinguish the three normalization conventions ($\omega$, $\xi$, $f$) and convert between them without error
12. Explain how the uncertainty principle constrains RoPE context length extension and attention window design

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 From Fourier Series to Fourier Transform](#11-from-fourier-series-to-fourier-transform)
  - [1.2 Frequency as a Continuous Variable](#12-frequency-as-a-continuous-variable)
  - [1.3 Why It Matters for AI (2026 Perspective)](#13-why-it-matters-for-ai-2026-perspective)
  - [1.4 Historical Timeline](#14-historical-timeline)
  - [1.5 What the Fourier Transform Does Geometrically](#15-what-the-fourier-transform-does-geometrically)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The Fourier Transform on $L^1(\mathbb{R})$](#21-the-fourier-transform-on-l1r)
  - [2.2 Convention War: $\omega$ vs $\xi$ vs $f$](#22-convention-war-omega-vs-xi-vs-f)
  - [2.3 Standard Examples](#23-standard-examples)
  - [2.4 The Inverse Fourier Transform](#24-the-inverse-fourier-transform)
  - [2.5 Non-Examples and the $L^2$ Extension](#25-non-examples-and-the-l2-extension)
- [3. Core Properties](#3-core-properties)
  - [3.1 Linearity and Conjugate Symmetry](#31-linearity-and-conjugate-symmetry)
  - [3.2 Time-Shift and Frequency-Shift (Modulation)](#32-time-shift-and-frequency-shift-modulation)
  - [3.3 Scaling and Dilation](#33-scaling-and-dilation)
  - [3.4 Time Reversal](#34-time-reversal)
  - [3.5 Differentiation and Integration](#35-differentiation-and-integration)
  - [3.6 The Master Properties Table](#36-the-master-properties-table)
  - [3.7 Convolution-Multiplication Duality - Preview](#37-convolution-multiplication-duality--preview)
- [4. The Fourier Inversion Theorem](#4-the-fourier-inversion-theorem)
  - [4.1 Statement and Sufficient Conditions](#41-statement-and-sufficient-conditions)
  - [4.2 Proof via Approximate Identity](#42-proof-via-approximate-identity)
  - [4.3 The Inversion Formula in Practice](#43-the-inversion-formula-in-practice)
  - [4.4 The Self-Dual Gaussian](#44-the-self-dual-gaussian)
- [5. Plancherel's Theorem and $L^2$ Theory](#5-plancherels-theorem-and-l2-theory)
  - [5.1 Statement: FT as a Unitary Isometry](#51-statement-ft-as-a-unitary-isometry)
  - [5.2 Parseval's Relation](#52-parsevals-relation)
  - [5.3 Proof Sketch: Extension from $L^1 \cap L^2$ to $L^2$](#53-proof-sketch-extension-from-l1-cap-l2-to-l2)
  - [5.4 Energy Conservation in Practice](#54-energy-conservation-in-practice)
- [6. The Heisenberg Uncertainty Principle](#6-the-heisenberg-uncertainty-principle)
  - [6.1 Time Spread and Frequency Spread](#61-time-spread-and-frequency-spread)
  - [6.2 Formal Statement: $\Delta t \cdot \Delta\xi \geq \frac{1}{4\pi}$](#62-formal-statement-delta-t-cdot-deltaxi-geq-frac14pi)
  - [6.3 Proof via Cauchy-Schwarz](#63-proof-via-cauchyschwarz)
  - [6.4 The Gaussian as the Unique Extremal Function](#64-the-gaussian-as-the-unique-extremal-function)
  - [6.5 Implications for Neural Architecture Design](#65-implications-for-neural-architecture-design)
- [7. Tempered Distributions and the Dirac Delta](#7-tempered-distributions-and-the-dirac-delta)
  - [7.1 Why Distributions Are Necessary](#71-why-distributions-are-necessary)
  - [7.2 Schwartz Space and Tempered Distributions](#72-schwartz-space-and-tempered-distributions)
  - [7.3 The Dirac Delta and its Fourier Transform](#73-the-dirac-delta-and-its-fourier-transform)
  - [7.4 FT of Constants, Sinusoids, and Periodic Signals](#74-ft-of-constants-sinusoids-and-periodic-signals)
  - [7.5 The Dirac Comb and Poisson Summation Formula](#75-the-dirac-comb-and-poisson-summation-formula)
  - [7.6 Unifying Fourier Series and Fourier Transform](#76-unifying-fourier-series-and-fourier-transform)
- [8. Applications in Machine Learning](#8-applications-in-machine-learning)
  - [8.1 FNet: Replacing Self-Attention with Fourier Transforms](#81-fnet-replacing-self-attention-with-fourier-transforms)
  - [8.2 Random Fourier Features](#82-random-fourier-features)
  - [8.3 Spectral Normalization](#83-spectral-normalization)
  - [8.4 Fourier Neural Operator - Preview](#84-fourier-neural-operator--preview)
  - [8.5 Bochner's Theorem and Shift-Invariant Kernels - Preview](#85-bochners-theorem-and-shift-invariant-kernels--preview)
  - [8.6 RoPE from the Continuous FT Perspective](#86-rope-from-the-continuous-ft-perspective)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)
- [13. Numerical Fourier Analysis: What the Grid Does to the Theorems](#13-numerical-fourier-analysis-what-the-grid-does-to-the-theorems)

---

## 1. Intuition

### 1.1 From Fourier Series to Fourier Transform

Recall from [Section 20-01 Fourier Series](../01-Fourier-Series/notes.md) that a $2L$-periodic function has a Fourier series in complex exponential form:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n\, e^{in\pi x/L}, \qquad c_n = \frac{1}{2L}\int_{-L}^{L} f(x)\,e^{-in\pi x/L}\,dx$$

The frequencies present are the **discrete** set $\{\ldots, -2/2L, -1/2L, 0, 1/2L, 2/2L, \ldots\}$, separated by spacing $\Delta\xi = 1/(2L)$.

Now ask: what happens as $L \to \infty$? The function is no longer required to repeat - it becomes an arbitrary function on all of $\mathbb{R}$. Three things change simultaneously:

1. The frequency spacing $\Delta\xi = 1/(2L) \to 0$: the discrete spectrum becomes a **continuum**.
2. The discrete sum $\sum_n$ becomes an integral $\int_{-\infty}^{\infty} d\xi$.
3. The coefficient $c_n$ (which scales as $1/(2L)$) becomes a density $\hat{f}(\xi)\,d\xi$.

Concretely, substitute $\xi_n = n/(2L)$ and let $L \to \infty$:

$$f(x) = \sum_{n=-\infty}^{\infty} \underbrace{\left(\int_{-L}^{L} f(t)\,e^{-2\pi i \xi_n t}\,dt\right)}_{\approx \hat{f}(\xi_n)} e^{2\pi i \xi_n x} \underbrace{\frac{1}{2L}}_{\Delta\xi}$$

$$\xrightarrow{L\to\infty} \int_{-\infty}^{\infty} \hat{f}(\xi)\, e^{2\pi i \xi x}\,d\xi$$

where the **Fourier Transform** (using the $\xi$-convention) is:

$$\hat{f}(\xi) = \int_{-\infty}^{\infty} f(t)\, e^{-2\pi i \xi t}\,dt$$

This derivation makes the Fourier Transform inevitable: it is not a definition pulled from thin air, but the natural limit of the Fourier series as periodicity is removed. The continuous spectrum $\hat{f}(\xi)$ is the **spectral density** of $f$ - the amplitude and phase contributed by frequency $\xi$ per unit frequency interval.

**Non-example:** The derivation above requires that the "coefficients" $\hat{f}(\xi)$ are well-defined as $L \to \infty$. For this, we need $\int_{-\infty}^{\infty} |f(t)|\,dt < \infty$, i.e., $f \in L^1(\mathbb{R})$. A constant function $f(t) = 1$ is NOT in $L^1(\mathbb{R})$, so it does not have a classical Fourier Transform - we must extend the theory to distributions (Section 7) to handle it.

### 1.2 Frequency as a Continuous Variable

In a Fourier series, the spectrum is a discrete set of spikes: the function has energy only at the harmonics $n/T$. In the Fourier Transform, the spectrum is a continuous function $\hat{f}(\xi)$, and the function can have energy spread continuously across all frequencies.

The **magnitude spectrum** $|\hat{f}(\xi)|$ tells you how much of each frequency is present. The **phase spectrum** $\angle\hat{f}(\xi)$ tells you the phase offset of each frequency component. Together they completely determine $f$ (via the inversion theorem). Some examples of what the spectrum looks like:

- A **pure sinusoid** $f(t) = \cos(2\pi\xi_0 t)$: spectrum is two spikes at $\pm\xi_0$ (as Dirac deltas).
- A **rectangular pulse** (nonzero on $[-T/2, T/2]$): spectrum is a sinc function, spreading out in frequency.
- A **Gaussian** $f(t) = e^{-\pi t^2}$: spectrum is another Gaussian (this is the self-dual case).
- A **chirp** (frequency increasing linearly with time): spectrum is spread over a range of frequencies, with the time-frequency tradeoff governed by the uncertainty principle.
- **White noise**: flat spectrum - equal energy at every frequency.

The key principle is the **time-frequency duality**: a signal that is concentrated in time (short duration) must have a broad spectrum, and a signal with a narrow spectrum (nearly monochromatic) must extend over a long time. This is not a limitation of our measurement apparatus - it is a mathematical theorem (the Heisenberg uncertainty principle, Section 6).

**For AI:** This duality directly constrains attention mechanisms. RoPE uses frequencies to encode position, and extending context length (longer time window) requires lower frequencies - which is why YaRN (Peng et al., 2023) and LongRoPE (Ding et al., 2024) rescale the frequency base to accommodate longer sequences. The uncertainty principle is the fundamental reason why this rescaling is necessary.

### 1.3 Why It Matters for AI (2026 Perspective)

The Fourier Transform is not peripheral to modern AI - it is structurally present in several of the most important systems:

**FNet** (Lee-Thorp et al., 2022): Replaces the self-attention sublayer in a Transformer with a 2D Fourier transform over the sequence and embedding dimensions. Achieves 92% of BERT's accuracy on GLUE benchmarks at 7 the training speed, because FFT is $O(N \log N)$ while attention is $O(N^2)$. The mathematical insight: the FT acts as a "global mixer" that combines all token representations - a weaker but cheaper version of attention.

**Random Fourier Features** (Rahimi & Recht, 2007, NeurIPS Best Paper): Every shift-invariant kernel $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{x} - \mathbf{y})$ is the Fourier Transform of a non-negative measure (Bochner's theorem). By sampling random frequencies from that measure, one gets a randomized feature map $\phi(\mathbf{x}) \in \mathbb{R}^D$ such that $k(\mathbf{x},\mathbf{y}) \approx \phi(\mathbf{x})^\top\phi(\mathbf{y})$. This reduces kernel machine computation from $O(N^2)$ to $O(ND)$.

**Spectral Normalization** (Miyato et al., 2018): To train stable GANs, the discriminator's weight matrices are normalized by their largest singular value (spectral radius). This enforces a Lipschitz constraint on the discriminator, stabilizing training. The "spectral" in the name refers to the spectrum of the weight matrix - a direct application of the FT of the weight matrices interpreted as linear operators.

**Fourier Neural Operator** (Li et al., 2021): Learns mappings between function spaces by applying a learnable linear transformation in Fourier space, then transforming back. Used for solving PDEs 1000 faster than traditional numerical methods. The key is that convolution in space = pointwise multiplication in Fourier space, so global dependencies can be captured efficiently.

**RoPE** (Su et al., 2021): Used in LLaMA-3, Mistral, Qwen, and virtually every modern LLM. Interprets each position as a rotation in the complex plane, using Fourier basis vectors $e^{im\theta_j}$ at different frequencies $\theta_j$ for different embedding dimensions. The relative-position property follows from the group structure of the complex exponential.

### 1.4 Historical Timeline

```
FOURIER TRANSFORM - HISTORICAL MILESTONES
========================================================================

  1822  Fourier's "Theorie analytique de la chaleur": introduces the
        integral formula that becomes the Fourier Transform
  1880s Rayleigh uses Fourier integrals in optics and acoustics
  1898  Heaviside operational calculus (early form of the Laplace/FT)
  1915  Plancherel proves the L2 isometry theorem (Plancherel's theorem)
  1927  Heisenberg formulates the uncertainty principle in quantum
        mechanics; Kennard gives the precise mathematical inequality
  1933  Norbert Wiener's "The Fourier Integral and Certain of its
        Applications" - rigorous L2 theory; lays groundwork for
        signal processing as a mathematical discipline
  1949  Shannon's sampling theorem (uses Poisson summation formula)
  1965  Cooley & Tukey publish the FFT - makes FT computable in O(NlogN)
  1966  Schwartz distributions: FT extended to , constants, sinusoids
  1975  FT enters digital signal processing (audio, radar, MRI)
  2007  Rahimi & Recht: Random Fourier Features (NeurIPS Best Paper)
  2017  Vaswani et al.: sinusoidal positional encodings in Transformers
  2018  Miyato et al.: spectral normalization for GANs
  2021  Li et al.: Fourier Neural Operator for PDE solving
  2021  Su et al.: RoPE - now in LLaMA-3, Mistral, Qwen, GPT-NeoX
  2022  Lee-Thorp et al.: FNet - Fourier-based alternative to attention
  2023  YaRN (Peng et al.): Fourier-based context length extension
  2024  LongRoPE (Ding et al.): progressive frequency rescaling to 2M
        context length; RWKV-7 uses state-space FT interpretation

========================================================================
```

### 1.5 What the Fourier Transform Does Geometrically

The most useful geometric picture is that the Fourier Transform **decomposes $f$ in an orthonormal basis of complex exponentials** - just as the Fourier series decomposed a periodic function in the trigonometric basis. The difference is that now the "basis" is uncountably infinite and the "coefficients" form a continuous function rather than a sequence.

Formally, the complex exponentials $\{e^{2\pi i \xi t}\}_{\xi \in \mathbb{R}}$ do not form a basis of $L^2(\mathbb{R})$ in the usual Hilbert space sense (they are not in $L^2(\mathbb{R})$ themselves - they have constant modulus 1 and are not square-integrable). The rigorous statement is that the Fourier Transform is a **unitary operator** on $L^2(\mathbb{R})$ (Plancherel's theorem), meaning it preserves inner products and norms:

$$\langle f, g \rangle_{L^2} = \langle \hat{f}, \hat{g} \rangle_{L^2}$$

Think of the Fourier Transform as a **rotation in an infinite-dimensional space**: it rotates the function $f$ into a new coordinate system (the frequency domain) where the same function looks different but has exactly the same "length" (energy). Just as rotating a vector in $\mathbb{R}^n$ preserves its norm but changes its coordinates, the Fourier Transform preserves energy but expresses it in frequency coordinates.

A second geometric picture: the Fourier Transform of the convolution $f * g$ is the pointwise product $\hat{f} \cdot \hat{g}$. Convolution in time/space corresponds to multiplication in frequency. This is the key theorem for signal processing and the foundation of efficient convolution in CNNs via the FFT. (Full treatment in [Section 20-04 Convolution Theorem](../04-Convolution-Theorem/notes.md).)

```
GEOMETRIC PICTURE OF THE FOURIER TRANSFORM
========================================================================

  Time domain                    Frequency domain
  -------------                  ----------------
  f(t): a function of time       f(): a function of frequency

  Signal duration T  <-->  Bandwidth ~ 1/T  (uncertainty principle)

  Convolution f*g    <-->  Pointwise product fg  (Convolution Theorem)

  Differentiation d/dt <-->  Multiplication by 2i

  Translation f(t-a) <-->  Modulation e^{-2ia}f()

  Scaling f(at)      <-->  Dilation (1/|a|)f(/a)

  The FT is a UNITARY OPERATOR on L2(R):
    f2 = f2  (Plancherel)
    f,g = f,g  (Parseval)

========================================================================
```

## 2. Formal Definitions

### 2.1 The Fourier Transform on $L^1(\mathbb{R})$

**Definition 2.1 (Fourier Transform on $L^1$).** For $f \in L^1(\mathbb{R})$ (i.e., $\int_{-\infty}^{\infty}|f(t)|\,dt < \infty$), the **Fourier Transform** of $f$ is the function $\hat{f}: \mathbb{R} \to \mathbb{C}$ defined by:

$$\hat{f}(\xi) = \int_{-\infty}^{\infty} f(t)\, e^{-2\pi i \xi t}\,dt$$

We write $\hat{f} = \mathcal{F}[f]$ or $f \mapsto \hat{f}$.

**Well-definedness.** For $f \in L^1(\mathbb{R})$ and any $\xi \in \mathbb{R}$:
$$|\hat{f}(\xi)| = \left|\int_{-\infty}^{\infty} f(t)\,e^{-2\pi i\xi t}\,dt\right| \leq \int_{-\infty}^{\infty}|f(t)|\,|e^{-2\pi i\xi t}|\,dt = \int_{-\infty}^{\infty}|f(t)|\,dt = \lVert f \rVert_1$$

So $\hat{f}$ is bounded: $\lVert\hat{f}\rVert_\infty \leq \lVert f \rVert_1$. Moreover, one can show (by dominated convergence) that $\hat{f}$ is **continuous** on $\mathbb{R}$, and by the Riemann-Lebesgue lemma, $\hat{f}(\xi) \to 0$ as $|\xi| \to \infty$.

**Theorem 2.1 (Riemann-Lebesgue Lemma).** If $f \in L^1(\mathbb{R})$, then $\hat{f} \in C_0(\mathbb{R})$ (continuous functions vanishing at infinity):
$$\lim_{|\xi|\to\infty} \hat{f}(\xi) = 0$$

*Proof sketch.* For step functions, the integral reduces to a finite sum of terms $\sim e^{-2\pi i\xi a}/(\xi)$, each going to 0. Approximate general $L^1$ functions by step functions and use the boundedness $\lVert\hat{f}\rVert_\infty \leq \lVert f \rVert_1$. $\square$

**What $\hat{f}(\xi)$ tells you.** The value $\hat{f}(\xi)$ is a complex number encoding:
- $|\hat{f}(\xi)|$: the **amplitude** of frequency $\xi$ in $f$
- $\arg\hat{f}(\xi)$: the **phase** of frequency $\xi$ in $f$

The **power spectrum** $|\hat{f}(\xi)|^2$ gives the energy density per unit frequency at $\xi$.

### 2.2 Convention War: $\omega$ vs $\xi$ vs $f$

The Fourier Transform appears in three notational conventions in the literature, all equivalent but related by rescaling:

| Convention | Transform formula | Inverse formula | Used in |
|-----------|------------------|-----------------|---------|
| **Frequency** $\xi$ (Hz) | $\hat{f}(\xi) = \int f(t)e^{-2\pi i\xi t}\,dt$ | $f(t) = \int \hat{f}(\xi)e^{2\pi i\xi t}\,d\xi$ | Signal processing, probability |
| **Angular freq** $\omega$ (rad/s), no $\frac{1}{2\pi}$ | $\hat{f}(\omega) = \int f(t)e^{-i\omega t}\,dt$ | $f(t) = \frac{1}{2\pi}\int \hat{f}(\omega)e^{i\omega t}\,d\omega$ | Physics, engineering |
| **Angular freq** $\omega$, symmetric | $\hat{f}(\omega) = \frac{1}{\sqrt{2\pi}}\int f(t)e^{-i\omega t}\,dt$ | $f(t) = \frac{1}{\sqrt{2\pi}}\int \hat{f}(\omega)e^{i\omega t}\,d\omega$ | Pure mathematics |

**This section uses the $\xi$-convention** (row 1). It is self-symmetric (no $2\pi$ factors), maps the Gaussian to itself, and is standard in modern ML papers (FNet, FNO, RFF all use it). The relationship between conventions: $\hat{f}_\omega(\omega) = \hat{f}_\xi(\omega/2\pi)$.

> **Warning:** The most common source of errors in Fourier analysis is mixing conventions. When reading a paper, identify the convention in the first equation before proceeding. Properties like "differentiation multiplies by $i\omega$" vs "multiplies by $2\pi i\xi$" depend entirely on which convention is in use.

### 2.3 Standard Examples

These four transforms appear constantly in applications and should be memorized.

**Example 1: Rectangular Pulse (rect function)**

$$\operatorname{rect}(t) = \begin{cases}1 & |t| \leq 1/2 \\ 0 & |t| > 1/2\end{cases}$$

$$\widehat{\operatorname{rect}}(\xi) = \int_{-1/2}^{1/2} e^{-2\pi i\xi t}\,dt = \frac{e^{\pi i\xi} - e^{-\pi i\xi}}{2\pi i\xi} = \frac{\sin(\pi\xi)}{\pi\xi} =: \operatorname{sinc}(\xi)$$

The sinc function oscillates and decays as $1/|\xi|$. The slow decay reflects the sharp discontinuity of $\operatorname{rect}$ - sharp features in time produce slow decay in frequency (this is the spectral analog of the Gibbs phenomenon from Section 20-01).

A general pulse of width $T$: $f(t) = \operatorname{rect}(t/T)$ has transform $\hat{f}(\xi) = T\,\operatorname{sinc}(T\xi)$. Wider pulse -> narrower, taller sinc lobe.

**Example 2: Gaussian**

$$f(t) = e^{-\pi t^2}$$

$$\hat{f}(\xi) = \int_{-\infty}^{\infty} e^{-\pi t^2} e^{-2\pi i\xi t}\,dt = e^{-\pi\xi^2}$$

The Gaussian is **self-dual** under the Fourier Transform with the $\xi$-convention: $\hat{f} = f$. This is unique among "nice" functions and makes the Gaussian the natural test function in Fourier analysis, the ground state in quantum mechanics, and the kernel of the heat equation.

*Derivation:* Complete the square in the exponent: $-\pi t^2 - 2\pi i\xi t = -\pi(t + i\xi)^2 - \pi\xi^2$. Then:
$$\hat{f}(\xi) = e^{-\pi\xi^2}\int_{-\infty}^{\infty}e^{-\pi(t+i\xi)^2}\,dt = e^{-\pi\xi^2} \cdot 1$$
where the integral equals 1 by contour integration (the Gaussian integral is analytic and the contour shift $t \to t + i\xi$ is justified because the integrand decays rapidly).

**Example 3: Lorentzian (Cauchy distribution)**

$$f(t) = \frac{1}{1 + (2\pi t)^2}$$

$$\hat{f}(\xi) = \frac{1}{2}e^{-|\xi|}$$

The exponential decay in frequency reflects the moderate smoothness of the Lorentzian (it is $C^\infty$ but not compactly supported). The relationship between smoothness and spectral decay is captured by the general theorem in Section 3.5.

**Example 4: One-Sided Exponential**

$$f(t) = e^{-at}\mathbb{1}_{t \geq 0}, \quad a > 0$$

$$\hat{f}(\xi) = \int_0^\infty e^{-at}e^{-2\pi i\xi t}\,dt = \frac{1}{a + 2\pi i\xi}$$

This has modulus $|\hat{f}(\xi)| = 1/\sqrt{a^2 + 4\pi^2\xi^2}$, which decays as $1/|\xi|$ for large $\xi$ - reflecting the discontinuity at $t = 0$.

**Non-example 1:** $f(t) = 1/(1+t^2)^{1/4}$. This is in $L^2(\mathbb{R})$ but not $L^1(\mathbb{R})$ (it decays too slowly), so the integral $\int f(t)e^{-2\pi i\xi t}\,dt$ does not converge absolutely and the definition above does not apply. The $L^2$ theory (Section 5) handles this case.

**Non-example 2:** $f(t) = \sin(2\pi\xi_0 t)$. This is bounded but not in $L^1(\mathbb{R})$ or $L^2(\mathbb{R})$ - it does not decay. Its Fourier Transform exists only as a distribution: $\mathcal{F}[\sin(2\pi\xi_0 t)] = \frac{i}{2}[\delta(\xi + \xi_0) - \delta(\xi - \xi_0)]$ (see Section 7).

### 2.4 The Inverse Fourier Transform

**Definition 2.2 (Inverse Fourier Transform).** For $g \in L^1(\mathbb{R})$:
$$\mathcal{F}^{-1}[g](t) = \int_{-\infty}^{\infty} g(\xi)\,e^{2\pi i\xi t}\,d\xi$$

Note: the inverse is the same as the forward transform except the sign in the exponent is flipped ($-2\pi i\xi t \to +2\pi i\xi t$). In the $\xi$-convention, this symmetric form is one of its advantages.

**The inversion problem:** Given $\hat{f}$, can we recover $f$? Yes, under mild conditions:
$$f(t) = \int_{-\infty}^{\infty} \hat{f}(\xi)\,e^{2\pi i\xi t}\,d\xi$$

but this requires knowing that $\hat{f} \in L^1(\mathbb{R})$ (which is not guaranteed by $f \in L^1(\mathbb{R})$ alone - for instance, $\hat{\operatorname{rect}}(\xi) = \operatorname{sinc}(\xi)$ is not in $L^1$). The full inversion theorem is treated rigorously in Section 4.

### 2.5 Non-Examples and the $L^2$ Extension

The classical $L^1$ definition fails for many important functions:

| Function | Why $L^1$ FT fails | Solution |
|---------|-------------------|---------|
| $f(t) = e^{-t^2/2}$ (broader Gaussian) | In $L^1$ - this one is fine | Not a failure |
| $f(t) = (1+t^2)^{-1/4}$ | Not in $L^1$ | Use $L^2$ Plancherel (Section 5) |
| $f(t) = 1$ | Not in $L^1$ or $L^2$ | Use distributions (Section 7) |
| $f(t) = \sin(2\pi\xi_0 t)$ | Not in $L^1$ or $L^2$ | Use distributions (Section 7) |
| $f(t) = \delta(t)$ | Not a function | Use distributions (Section 7) |

The **$L^2$ extension** proceeds via a density argument. The subspace $L^1(\mathbb{R}) \cap L^2(\mathbb{R})$ is dense in $L^2(\mathbb{R})$. For $f \in L^1 \cap L^2$, the classical integral defines $\hat{f}$. The Plancherel theorem (Section 5) then shows $\lVert\hat{f}\rVert_2 = \lVert f \rVert_2$, which allows extending $\mathcal{F}$ to all of $L^2$ by continuity. The resulting transform is a **unitary operator** on $L^2(\mathbb{R})$.

## 3. Core Properties

### 3.1 Linearity and Conjugate Symmetry

**Theorem 3.1 (Linearity).** For $f, g \in L^1(\mathbb{R})$ and $\alpha, \beta \in \mathbb{C}$:
$$\mathcal{F}[\alpha f + \beta g](\xi) = \alpha\hat{f}(\xi) + \beta\hat{g}(\xi)$$

*Proof:* Immediate from linearity of the integral. $\square$

**Theorem 3.2 (Conjugate Symmetry / Hermitian Property).** For real-valued $f \in L^1(\mathbb{R})$:
$$\hat{f}(-\xi) = \overline{\hat{f}(\xi)}$$

*Proof:* $\hat{f}(-\xi) = \int f(t)e^{2\pi i\xi t}\,dt = \overline{\int f(t)e^{-2\pi i\xi t}\,dt} = \overline{\hat{f}(\xi)}$, where the last step uses $f(t) = \overline{f(t)}$ (real-valued). $\square$

**Corollaries for real $f$:**
- $|\hat{f}(-\xi)| = |\hat{f}(\xi)|$: the magnitude spectrum is **even**
- $\arg\hat{f}(-\xi) = -\arg\hat{f}(\xi)$: the phase spectrum is **odd**
- If $f$ is also even, then $\hat{f}$ is real-valued and even
- If $f$ is also odd, then $\hat{f}$ is purely imaginary and odd

**For AI:** The Hermitian symmetry means that for real signals, half the spectrum is redundant - you only need frequencies $\xi \geq 0$. This is why the FFT of a real signal produces $N/2 + 1$ unique complex outputs from $N$ real inputs. Real-valued weight matrices in neural networks have Hermitian-symmetric spectra, which is exploited in efficient spectral computation.

### 3.2 Time-Shift and Frequency-Shift (Modulation)

**Theorem 3.3 (Time-Shift / Translation).** For $a \in \mathbb{R}$:
$$\mathcal{F}[f(t - a)](\xi) = e^{-2\pi i\xi a}\,\hat{f}(\xi)$$

*Proof:* $\int f(t-a)e^{-2\pi i\xi t}\,dt \stackrel{u=t-a}{=} \int f(u)e^{-2\pi i\xi(u+a)}\,du = e^{-2\pi i\xi a}\hat{f}(\xi)$. $\square$

**Interpretation:** Shifting a signal in time multiplies its spectrum by a complex phase factor. The **magnitude** spectrum $|\hat{f}(\xi)|$ is unchanged - a time shift does not affect which frequencies are present, only their phases. The phase changes linearly with frequency: $\phi \mapsto \phi - 2\pi\xi a$.

**Theorem 3.4 (Frequency-Shift / Modulation).** For $\xi_0 \in \mathbb{R}$:
$$\mathcal{F}[e^{2\pi i\xi_0 t}f(t)](\xi) = \hat{f}(\xi - \xi_0)$$

*Proof:* $\int e^{2\pi i\xi_0 t}f(t)e^{-2\pi i\xi t}\,dt = \int f(t)e^{-2\pi i(\xi - \xi_0)t}\,dt = \hat{f}(\xi - \xi_0)$. $\square$

**Interpretation:** Multiplying by a complex exponential (modulation) shifts the spectrum. This is the mathematical basis for **amplitude modulation (AM) radio** and for the **RoPE** positional encoding.

**For AI - RoPE connection:** In RoPE, the query at position $m$ is $\mathbf{q}_m = e^{im\theta}\mathbf{q}$ (rotation by $m\theta$ in the complex plane). The inner product between query at $m$ and key at $n$ is $\langle e^{im\theta}\mathbf{q},\, e^{in\theta}\mathbf{k}\rangle = e^{i(m-n)\theta}\langle\mathbf{q},\mathbf{k}\rangle$, depending only on relative position $m - n$. This is the modulation theorem in action.

### 3.3 Scaling and Dilation

**Theorem 3.5 (Scaling / Dilation).** For $a \in \mathbb{R}$, $a \neq 0$:
$$\mathcal{F}[f(at)](\xi) = \frac{1}{|a|}\hat{f}\!\left(\frac{\xi}{a}\right)$$

*Proof:* $\int f(at)e^{-2\pi i\xi t}\,dt \stackrel{u=at}{=} \frac{1}{|a|}\int f(u)e^{-2\pi i(\xi/a)u}\,du = \frac{1}{|a|}\hat{f}(\xi/a)$. $\square$

**Interpretation:** This is the **time-bandwidth product** in action:
- Compressing a signal in time ($a > 1$, so $f(at)$ is narrower): the spectrum stretches by factor $a$ and shrinks in amplitude by $1/a$
- Stretching a signal in time ($0 < a < 1$): the spectrum compresses

This confirms the time-frequency duality: **doubling the duration halves the bandwidth, and vice versa.** This is a hard mathematical constraint, not an engineering limitation.

### 3.4 Time Reversal

**Theorem 3.6 (Time Reversal).** For $f \in L^1(\mathbb{R})$:
$$\mathcal{F}[f(-t)](\xi) = \hat{f}(-\xi) = \overline{\hat{f}(\xi)} \text{ (for real } f\text{)}$$

*Proof:* $\int f(-t)e^{-2\pi i\xi t}\,dt \stackrel{u=-t}{=} \int f(u)e^{2\pi i\xi u}\,du = \hat{f}(-\xi)$. $\square$

**Duality:** There is a deeper symmetry: $\mathcal{F}[\hat{f}](t) = f(-t)$. Applying the Fourier Transform four times returns the original function: $\mathcal{F}^4 = I$. This means the FT has eigenvalues $\{1, -i, -1, i\}$ and the **Hermite functions** are its eigenfunctions (with the Gaussian as the eigenfunction for eigenvalue 1).

### 3.5 Differentiation and Integration

**Theorem 3.7 (Differentiation in Time).** If $f, f' \in L^1(\mathbb{R})$:
$$\mathcal{F}[f'(t)](\xi) = 2\pi i\xi\,\hat{f}(\xi)$$

More generally, $\mathcal{F}[f^{(n)}(t)](\xi) = (2\pi i\xi)^n\hat{f}(\xi)$.

*Proof:* Integration by parts: $\int f'(t)e^{-2\pi i\xi t}\,dt = [f(t)e^{-2\pi i\xi t}]_{-\infty}^{\infty} + 2\pi i\xi\int f(t)e^{-2\pi i\xi t}\,dt$. The boundary term vanishes since $f \in L^1$ implies $f(t) \to 0$ as $|t| \to \infty$. $\square$

**Theorem 3.8 (Differentiation in Frequency).** If $tf(t) \in L^1(\mathbb{R})$:
$$\frac{d}{d\xi}\hat{f}(\xi) = -2\pi i\,\widehat{tf(t)}(\xi)$$

Equivalently: $\mathcal{F}[(-2\pi it)^n f(t)](\xi) = \frac{d^n}{d\xi^n}\hat{f}(\xi)$.

**Key consequence for smoothness vs. spectral decay:**

| Smoothness of $f$ | Decay of $\hat{f}$ |
|------------------|-------------------|
| $f \in C^k$ and $f^{(k)} \in L^1$ | $|\hat{f}(\xi)| = O(|\xi|^{-k})$ as $|\xi| \to \infty$ |
| $f \in C^\infty$ (smooth) | $\hat{f}$ decays faster than any polynomial |
| $f$ has a jump discontinuity | $|\hat{f}(\xi)| = O(1/|\xi|)$ - slow decay |
| $f$ is analytic | $\hat{f}$ decays exponentially |

**For AI:** The differentiation property is why Fourier methods are efficient for solving differential equations. It converts a PDE $\partial_t u = a\partial_{xx}u$ into an ODE $\partial_t\hat{u} = -4\pi^2\xi^2 a\hat{u}$, which is solved by $\hat{u}(\xi, t) = \hat{u}(\xi, 0)e^{-4\pi^2 a\xi^2 t}$ - a simple exponential decay. The Fourier Neural Operator (Section 8.4) exploits this by learning the spectral solution operator directly.

### 3.6 The Master Properties Table

| Property | $f(t)$ | $\hat{f}(\xi)$ |
|---------|--------|----------------|
| Linearity | $\alpha f(t) + \beta g(t)$ | $\alpha\hat{f}(\xi) + \beta\hat{g}(\xi)$ |
| Time shift | $f(t - a)$ | $e^{-2\pi i\xi a}\hat{f}(\xi)$ |
| Frequency shift | $e^{2\pi i\xi_0 t}f(t)$ | $\hat{f}(\xi - \xi_0)$ |
| Scaling | $f(at)$ | $\frac{1}{\lvert a\rvert}\hat{f}(\xi/a)$ |
| Time reversal | $f(-t)$ | $\hat{f}(-\xi)$ |
| Conjugation | $\overline{f(t)}$ | $\overline{\hat{f}(-\xi)}$ |
| Hermitian (real $f$) | $f(t) \in \mathbb{R}$ | $\hat{f}(-\xi) = \overline{\hat{f}(\xi)}$ |
| Differentiation | $f^{(n)}(t)$ | $(2\pi i\xi)^n\hat{f}(\xi)$ |
| Freq. differentiation | $(-2\pi it)^n f(t)$ | $\hat{f}^{(n)}(\xi)$ |
| Convolution | $(f * g)(t)$ | $\hat{f}(\xi)\cdot\hat{g}(\xi)$ |
| Multiplication | $f(t)\cdot g(t)$ | $(\hat{f} * \hat{g})(\xi)$ |
| Duality | $\hat{f}(t)$ | $f(-\xi)$ |

### 3.7 Convolution-Multiplication Duality - Preview

The **Convolution Theorem** states that the Fourier Transform converts convolution into pointwise multiplication:

$$\mathcal{F}[f * g](\xi) = \hat{f}(\xi) \cdot \hat{g}(\xi)$$

where $(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t - \tau)\,d\tau$.

This is the most important property of the Fourier Transform for applications. It means that:
- **Linear filtering** (convolution with a filter $h$) corresponds to **pointwise multiplication** of spectra
- A filter's effect on frequency $\xi$ is simply multiplication by $\hat{h}(\xi)$ (the **frequency response**)
- Convolution of length-$N$ signals costs $O(N^2)$ directly but $O(N\log N)$ via FFT

> **Preview:** The full treatment of the convolution theorem - including circular convolution, cross-correlation, the Wiener-Khinchin theorem, and applications to CNNs, WaveNet, and Mamba - is in [Section 20-04 Convolution Theorem](../04-Convolution-Theorem/notes.md). Here we state the theorem and use it; the proof and all applications belong there.

## 4. The Fourier Inversion Theorem

### 4.1 Statement and Sufficient Conditions

The fundamental question: given $\hat{f}$, can we recover $f$? The answer is yes under appropriate conditions, but the precise statement requires care.

**Theorem 4.1 (Fourier Inversion Theorem).** Suppose $f \in L^1(\mathbb{R})$ and $\hat{f} \in L^1(\mathbb{R})$. Then for almost every $t \in \mathbb{R}$:

$$f(t) = \int_{-\infty}^{\infty} \hat{f}(\xi)\,e^{2\pi i\xi t}\,d\xi$$

Moreover, if $f$ is also continuous at $t$, equality holds exactly (not just a.e.).

**The subtlety:** The condition $\hat{f} \in L^1(\mathbb{R})$ is not automatic. If $f \in L^1(\mathbb{R})$, then $\hat{f}$ is bounded and continuous (by Riemann-Lebesgue), but not necessarily in $L^1$. For example, $f = \operatorname{rect}$ gives $\hat{f} = \operatorname{sinc}$, and $\operatorname{sinc} \notin L^1(\mathbb{R})$ (it decays too slowly: $\int_1^\infty \frac{|\sin(\pi\xi)|}{\pi\xi}d\xi = \infty$). The inversion of the rect function requires the $L^2$ theory.

**Sufficient condition for inversion:** If $f \in L^1(\mathbb{R})$ and $f$ has bounded variation near $t$ (e.g., $f$ is differentiable at $t$), then the principal value integral:
$$\lim_{R\to\infty}\int_{-R}^{R}\hat{f}(\xi)e^{2\pi i\xi t}\,d\xi$$
converges to $f(t)$ where $f$ is continuous, and to $\frac{1}{2}[f(t^+) + f(t^-)]$ at jump discontinuities - exactly as in Dirichlet's theorem for Fourier series.

### 4.2 Proof via Approximate Identity

The standard proof uses the concept of an **approximate identity** - a family of functions that converge to the Dirac delta as a parameter goes to zero.

**Definition (Approximate Identity).** A family $\{k_\epsilon\}_{\epsilon > 0}$ is an approximate identity if:
1. $k_\epsilon \geq 0$
2. $\int k_\epsilon(t)\,dt = 1$ for all $\epsilon > 0$
3. For every $\delta > 0$: $\int_{|t| > \delta} k_\epsilon(t)\,dt \to 0$ as $\epsilon \to 0$

The **Gauss-Weierstrass kernel** $k_\epsilon(t) = \frac{1}{\epsilon}e^{-\pi t^2/\epsilon^2}$ is an approximate identity.

**Proof sketch of inversion theorem:**

Define the **Gauss regularized inversion**:
$$f_\epsilon(t) = \int_{-\infty}^{\infty} \hat{f}(\xi)\,e^{2\pi i\xi t}\,e^{-\pi\epsilon^2\xi^2}\,d\xi$$

The factor $e^{-\pi\epsilon^2\xi^2}$ provides absolute convergence (the Gaussian is in $L^1$). Using Fubini's theorem to interchange integration order:
$$f_\epsilon(t) = \int_{-\infty}^{\infty} f(s) \underbrace{\int_{-\infty}^{\infty} e^{-\pi\epsilon^2\xi^2}e^{2\pi i\xi(t-s)}\,d\xi}_{= \frac{1}{\epsilon}e^{-\pi(t-s)^2/\epsilon^2}} ds = (f * k_\epsilon)(t)$$

Since $\{k_\epsilon\}$ is an approximate identity: $f_\epsilon(t) \to f(t)$ as $\epsilon \to 0$ at every point of continuity of $f$ (and in $L^1$ norm). Since $f_\epsilon(t) \to \int\hat{f}(\xi)e^{2\pi i\xi t}\,d\xi$ as $\epsilon \to 0$ when $\hat{f} \in L^1$, the result follows. $\square$

**Significance:** This proof reveals why the Gaussian plays a central role - it is the unique function (up to scaling) that is its own Fourier Transform, making the Gauss regularization self-consistent.

### 4.3 The Inversion Formula in Practice

The inversion formula is used to:

1. **Recover a signal from its spectrum:** Given $\hat{f}$, compute $f(t) = \int\hat{f}(\xi)e^{2\pi i\xi t}\,d\xi$
2. **Compute transforms of new functions from known transforms:** Use the FT table plus properties
3. **Solve PDEs:** Take FT, solve the resulting ODE, invert

**Example:** Compute $f(t)$ such that $\hat{f}(\xi) = e^{-2\pi|\xi|}$ (double-sided exponential spectrum).

By the inversion formula:
$$f(t) = \int_{-\infty}^{\infty} e^{-2\pi|\xi|}e^{2\pi i\xi t}\,d\xi = \int_0^\infty e^{-2\pi\xi(1-it)}\,d\xi + \int_{-\infty}^0 e^{2\pi\xi(1+it)}\,d\xi$$
$$= \frac{1}{2\pi(1-it)} + \frac{1}{2\pi(1+it)} = \frac{1}{\pi}\cdot\frac{1}{1+t^2}$$

So $\mathcal{F}[1/(\pi(1+t^2))](\xi) = e^{-2\pi|\xi|}$ - confirming the Lorentzian result from Section 2.3 (Example 3).

**Numerical verification:** The inversion theorem can be verified numerically via the FFT: compute $\hat{f}$ numerically, then invert, and check that you recover $f$ up to discretization error (this is Exercise 3 in Section 10).

### 4.4 The Self-Dual Gaussian

The Gaussian $g(t) = e^{-\pi t^2}$ satisfies $\hat{g} = g$, making it a **fixed point** of the Fourier Transform. This property makes it indispensable in both theory and practice.

**Generalized Gaussian family.** For $\sigma > 0$, define $g_\sigma(t) = e^{-\pi t^2/\sigma^2}$ (Gaussian of width $\sigma$). By the scaling theorem:
$$\hat{g}_\sigma(\xi) = \sigma\,e^{-\pi\sigma^2\xi^2} = \sigma\,g_{1/\sigma}(\xi)$$

Wide Gaussian ($\sigma$ large) -> narrow Fourier Transform (bandwidth $\sim 1/\sigma$). This is the uncertainty principle made explicit: the product of time-width $\sigma$ and frequency-width $1/\sigma$ equals 1, the minimum possible (see Section 6).

**Eigenfunctions of the FT:** The Fourier Transform has eigenvalues $\lambda \in \{1, -i, -1, i\}$. The corresponding eigenfunctions are the **Hermite functions**:
$$\psi_n(t) = H_n(\sqrt{2\pi}\,t)\,e^{-\pi t^2}$$
where $H_n$ is the $n$-th Hermite polynomial. For $n = 0$: $\psi_0(t) = e^{-\pi t^2}$ (eigenvalue 1, the self-dual Gaussian). For $n = 1$: $\psi_1(t) = 2\sqrt{2\pi}\,t\,e^{-\pi t^2}$ (eigenvalue $-i$).

**For AI:** The Gaussian window is the optimal time-frequency window (by the uncertainty principle), which is why Gaussian smoothing is used in signal preprocessing and why Gaussian attention kernels appear in some efficient attention variants.

```
GAUSSIAN SELF-DUALITY
========================================================================

  Time domain               Frequency domain
  -------------             ----------------
  g(t) = e^{-t2}   --FT--  g() = e^{-2}  <- same function!

  Width _t = 1/(2)       Width _ = 1/(2)
  Product: _t  _ = 1/(2) = MINIMUM (uncertainty bound)

  Scaling:
  f(t) = e^{-t2/2} --FT-- e^{-22}
  ^ wider in time               ^ narrower in frequency

========================================================================
```

## 5. Plancherel's Theorem and $L^2$ Theory

### 5.1 Statement: FT as a Unitary Isometry

**Theorem 5.1 (Plancherel's Theorem).** The Fourier Transform extends uniquely from $L^1(\mathbb{R}) \cap L^2(\mathbb{R})$ to a **unitary operator** on $L^2(\mathbb{R})$:
$$\mathcal{F}: L^2(\mathbb{R}) \to L^2(\mathbb{R}), \qquad \lVert \hat{f} \rVert_{L^2} = \lVert f \rVert_{L^2}$$

**Unitarity** means $\mathcal{F}^{-1} = \mathcal{F}^*$ (the adjoint equals the inverse), which in this case means $(\mathcal{F}^*)[\hat{f}](t) = \int \hat{f}(\xi)e^{+2\pi i\xi t}\,d\xi$ - the inverse FT.

**Why this is non-trivial:** For $f \in L^2(\mathbb{R})$ but $f \notin L^1(\mathbb{R})$, the defining integral $\int f(t)e^{-2\pi i\xi t}\,dt$ need not converge absolutely. Plancherel's theorem says we can still define $\hat{f} \in L^2$ as the $L^2$ limit:
$$\hat{f} = \lim_{R\to\infty} \int_{-R}^{R} f(t)e^{-2\pi i\xi t}\,dt \qquad \text{(limit in } L^2\text{ norm)}$$

The limit exists because the truncated transforms form a Cauchy sequence in $L^2$ (by the $L^1 \cap L^2$ case and density of that subspace in $L^2$).

### 5.2 Parseval's Relation

**Theorem 5.2 (Parseval / Plancherel Identity).** For $f, g \in L^2(\mathbb{R})$:
$$\int_{-\infty}^{\infty} f(t)\overline{g(t)}\,dt = \int_{-\infty}^{\infty} \hat{f}(\xi)\overline{\hat{g}(\xi)}\,d\xi$$

The special case $g = f$ gives the **energy conservation** (or Parseval's formula):
$$\int_{-\infty}^{\infty} |f(t)|^2\,dt = \int_{-\infty}^{\infty} |\hat{f}(\xi)|^2\,d\xi$$

**Interpretation:** The total energy of a signal is the same whether measured in the time domain or the frequency domain. No energy is created or destroyed by the Fourier Transform - it is a lossless change of representation.

**Using Parseval to compute integrals:** Sometimes $\int|f|^2\,dt$ is hard but $\int|\hat{f}|^2\,d\xi$ is easy (or vice versa). Example: compute $\int_{-\infty}^{\infty} \operatorname{sinc}^2(\xi)\,d\xi$.

We know $\mathcal{F}[\operatorname{rect}] = \operatorname{sinc}$, so by Parseval:
$$\int_{-\infty}^{\infty}|\operatorname{sinc}(\xi)|^2\,d\xi = \int_{-\infty}^{\infty}|\operatorname{rect}(t)|^2\,dt = \int_{-1/2}^{1/2}1\,dt = 1$$

**For AI:** Parseval's relation underpins the analysis of **spectral energy distribution** in neural networks. The WeightWatcher tool (Martin & Mahoney, 2021) analyzes the spectrum of weight matrices to diagnose training quality - a healthy model has a power-law spectral distribution. Parseval tells us the Frobenius norm $\lVert W \rVert_F^2 = \sum_i\sigma_i^2$ equals the energy in the spectral domain.

### 5.3 Proof Sketch: Extension from $L^1 \cap L^2$ to $L^2$

The proof of Plancherel's theorem follows a classic functional analysis pattern:

**Step 1:** Show the Parseval identity for $f \in L^1(\mathbb{R}) \cap L^2(\mathbb{R})$ (the intersection is dense in both spaces).

For $f \in L^1 \cap L^2$, use Fubini to compute:
$$\int |\hat{f}(\xi)|^2\,d\xi = \int\hat{f}(\xi)\overline{\hat{f}(\xi)}\,d\xi = \int\hat{f}(\xi)\left(\int f(t)e^{2\pi i\xi t}\,dt\right)d\xi$$
$$= \int f(t)\underbrace{\int\hat{f}(\xi)e^{2\pi i\xi t}\,d\xi}_{=f(t) \text{ by inversion}}\,dt = \int|f(t)|^2\,dt$$

**Step 2:** The isometry $\lVert\hat{f}\rVert_2 = \lVert f \rVert_2$ for $f \in L^1 \cap L^2$ shows $\mathcal{F}: L^1\cap L^2 \to L^2$ is a bounded operator with operator norm 1.

**Step 3:** By density ($L^1 \cap L^2$ is dense in $L^2$) and the fact that a bounded linear isometry extends uniquely to the closure, $\mathcal{F}$ extends to all of $L^2$ preserving the isometry.

**Step 4:** Show $\mathcal{F}$ is surjective by showing $\mathcal{F}^{-1}$ (the inverse FT) also extends and $\mathcal{F}\circ\mathcal{F}^{-1} = I$. $\square$

### 5.4 Energy Conservation in Practice

The Parseval identity has immediate computational consequences:

**Low-pass filtering:** A filter that removes frequencies $|\xi| > W$ (a rectangular window in frequency) preserves at most a fraction $2W\lVert\hat{f}\rVert_2^2/\lVert\hat{f}\rVert_2^2$ of the total energy. For a signal with most energy at low frequencies, this fraction is close to 1.

**Compression by truncation:** For $f \in L^2$, the best $k$-term approximation in the Fourier basis minimizes the $L^2$ error. The error is $\int_{|\xi| > W}|\hat{f}(\xi)|^2\,d\xi$ - the energy in the discarded frequencies. This is the Fourier analog of truncated SVD in matrix approximation.

**For AI:** The Fourier Neural Operator (Section 8.4) exploits this by keeping only the top-$k$ Fourier modes (lowest frequencies) of the input function and discarding the rest. Plancherel guarantees the error is exactly the discarded energy - a principled truncation criterion.

## 6. The Heisenberg Uncertainty Principle

### 6.1 Time Spread and Frequency Spread

To state the uncertainty principle precisely, we need quantitative measures of how "spread out" a signal is in time and frequency.

**Definition 6.1 (Time Center and Spread).** For $f \in L^2(\mathbb{R})$ with $\lVert f \rVert_2 = 1$:
$$\mu_t = \int_{-\infty}^{\infty} t\,|f(t)|^2\,dt \quad \text{(time center)}$$
$$\Delta t = \left(\int_{-\infty}^{\infty} (t - \mu_t)^2\,|f(t)|^2\,dt\right)^{1/2} \quad \text{(time spread / RMS duration)}$$

**Definition 6.2 (Frequency Center and Spread).** Similarly:
$$\mu_\xi = \int_{-\infty}^{\infty} \xi\,|\hat{f}(\xi)|^2\,d\xi \quad \text{(frequency center)}$$
$$\Delta\xi = \left(\int_{-\infty}^{\infty} (\xi - \mu_\xi)^2\,|\hat{f}(\xi)|^2\,d\xi\right)^{1/2} \quad \text{(frequency spread / bandwidth)}$$

Note: since $\lVert f \rVert_2^2 = \lVert\hat{f}\rVert_2^2 = 1$ (Plancherel), $|f(t)|^2$ and $|\hat{f}(\xi)|^2$ are probability densities. The time and frequency spreads are the standard deviations of these distributions.

### 6.2 Formal Statement: $\Delta t \cdot \Delta\xi \geq \frac{1}{4\pi}$

**Theorem 6.1 (Heisenberg-Kennard Uncertainty Principle).** For any $f \in L^2(\mathbb{R})$ with $\lVert f \rVert_2 = 1$ and $tf(t), \xi\hat{f}(\xi) \in L^2(\mathbb{R})$:

$$\Delta t \cdot \Delta\xi \geq \frac{1}{4\pi}$$

Equality holds if and only if $f$ is a Gaussian (up to time-shift, frequency-shift, and scaling):
$$f(t) = A\,e^{-\pi(t-t_0)^2/\sigma^2}\,e^{2\pi i\xi_0 t}$$
for some $A, t_0, \xi_0 \in \mathbb{R}$ and $\sigma > 0$.

**The bound is fundamental.** This is not a statement about measurement error or quantum physics - it is a purely mathematical theorem about functions and their Fourier Transforms. Any signal with time duration $\Delta t$ necessarily has bandwidth at least $1/(4\pi\Delta t)$. You cannot concentrate a signal to be both time-limited and bandwidth-limited simultaneously.

### 6.3 Proof via Cauchy-Schwarz

**Proof.** Without loss of generality, assume $\mu_t = \mu_\xi = 0$ (we can always translate in time and frequency without changing the spreads). Then:

$$\Delta t^2 \cdot \Delta\xi^2 = \left(\int t^2|f(t)|^2\,dt\right)\left(\int \xi^2|\hat{f}(\xi)|^2\,d\xi\right)$$

By Parseval and the differentiation property, $\int\xi^2|\hat{f}(\xi)|^2\,d\xi = \frac{1}{4\pi^2}\int|f'(t)|^2\,dt$ (using $\mathcal{F}[f'](\xi) = 2\pi i\xi\hat{f}(\xi)$ and Parseval).

So we need to show:
$$\left(\int t^2|f|^2\,dt\right)\left(\frac{1}{4\pi^2}\int|f'|^2\,dt\right) \geq \frac{1}{16\pi^2}$$

i.e., $\left(\int t^2|f|^2\right)\left(\int|f'|^2\right) \geq \frac{1}{4}$.

By the Cauchy-Schwarz inequality:
$$\int |f|^2\,dt = \int \lvert f\rvert \cdot 1\,dt \leq \cdots$$

More elegantly, integrate by parts and use Cauchy-Schwarz:
$$\frac{1}{2} = -\frac{1}{2}\int \frac{d}{dt}(t)|f|^2\,dt = -\frac{1}{2}\left[t|f|^2\right]_{-\infty}^\infty + \frac{1}{2}\int t\frac{d}{dt}|f|^2\,dt$$

Wait - the cleaner path: use $\frac{d}{dt}(t|f|^2) = |f|^2 + t \cdot 2\operatorname{Re}(f'\bar{f})$, integrate over $\mathbb{R}$, and the left side integrates to 0 (since $f \in L^2$ implies $t|f|^2 \to 0$):
$$0 = \int|f|^2\,dt + 2\int t\operatorname{Re}(f'\bar{f})\,dt = 1 + 2\operatorname{Re}\int t\,\bar{f}(t)f'(t)\,dt$$

So $\operatorname{Re}\int t\,\bar{f}f'\,dt = -1/2$. Then by Cauchy-Schwarz:
$$\frac{1}{4} = \left|\operatorname{Re}\int t\bar{f}f'\,dt\right|^2 \leq \left|\int t\bar{f}f'\,dt\right|^2 \leq \left(\int t^2|f|^2\,dt\right)\left(\int|f'|^2\,dt\right)$$

Dividing both sides by $4\pi^2$:
$$\Delta t^2 \cdot \Delta\xi^2 \geq \frac{1}{4} \cdot \frac{1}{4\pi^2} = \frac{1}{16\pi^2}$$
$$\therefore \quad \Delta t \cdot \Delta\xi \geq \frac{1}{4\pi} \qquad \square$$

### 6.4 The Gaussian as the Unique Extremal Function

Equality in Cauchy-Schwarz requires $f'(t) = \lambda t f(t)$ for some constant $\lambda$. This ODE has solution $f(t) = Ce^{\lambda t^2/2}$. For $f \in L^2(\mathbb{R})$, we need $\lambda < 0$, giving $f(t) = Ce^{-\alpha t^2}$ for $\alpha > 0$ - a Gaussian.

Adding back the time-shift $t_0$ and frequency-shift $\xi_0$:
$$f(t) = C\,e^{-\pi(t-t_0)^2/\sigma^2}\,e^{2\pi i\xi_0 t}, \qquad \Delta t = \sigma/\sqrt{2}, \quad \Delta\xi = 1/(2\sigma\sqrt{2})$$

Check: $\Delta t \cdot \Delta\xi = \sigma/\sqrt{2} \cdot 1/(2\sigma\sqrt{2}) = 1/(4) \cdot 1/\pi$... Actually with the normalization $g(t) = e^{-\pi t^2}$: $\Delta t = 1/(2\sqrt{\pi})$, $\Delta\xi = 1/(2\sqrt{\pi})$, product = $1/(4\pi)$ - exactly the bound.

**Implication:** The Gaussian is the **only signal that achieves perfect time-frequency concentration** in the sense of minimizing $\Delta t \cdot \Delta\xi$. Every other signal is "more spread out" in at least one of the two domains. This is why Gaussian windows are optimal for time-frequency analysis (Gabor atoms), and why the Gaussian noise model is natural in signal processing.

### 6.5 Implications for Neural Architecture Design

The uncertainty principle has direct, concrete implications for AI systems:

**RoPE and context length extension:** In RoPE, each dimension pair uses frequency $\theta_j = 10000^{-2j/d_{\text{model}}}$. The lowest frequencies $\theta_j$ encode the longest-range positional information. To extend context from 4K to 128K tokens (as in LLaMA-3 long context), the lowest frequencies must be able to distinguish positions up to 128K. But by the uncertainty principle, a signal that is distinguishable over a range $T$ in "position space" must have frequency resolution $\Delta\xi \leq 1/T$ - which requires the frequency to be low enough. YaRN (Peng et al., 2023) and LongRoPE (Ding et al., 2024) resolve this by rescaling $\theta_j$ to use lower frequencies for long-context models.

**Spectrogram and STFT:** Audio models like Whisper (Radford et al., 2022) use the **Short-Time Fourier Transform** - the FT applied to windowed segments. The window length controls the time-frequency tradeoff. Longer windows: better frequency resolution, worse time resolution. The Gaussian window is optimal but the Hamming window is commonly used for computational reasons.

**Attention window size:** Multi-head self-attention with local windows (as in LongFormer, BigBird) effectively applies a bandpass filter in "position space." The uncertainty principle says a window of size $W$ can resolve frequency up to $1/W$ - setting a fundamental limit on the position encodings that can be usefully resolved within the window.

```
UNCERTAINTY PRINCIPLE - ARCHITECTURE IMPLICATIONS
========================================================================

  Signal Analysis            Neural Architecture Analog
  -----------------          -------------------------
  Time duration T            Context window / sequence length
  Bandwidth B ~ 1/T          Position frequency resolution
  t   >= 1/(4)          Short context  coarse PE, long  fine PE

  STFT window size W:        Local attention window size W:
    -> freq res: ~ 1/W          -> PE resolution: ~ 1/W
    -> time res: ~ W            -> local context: ~ W

  Gaussian window:           Gaussian attention kernel:
    -> optimal concentration    -> used in some efficient attention variants

  LongRoPE: rescales j to lower freqs -> finer resolution at long range

========================================================================
```

## 7. Tempered Distributions and the Dirac Delta

### 7.1 Why Distributions Are Necessary

The $L^1$ and $L^2$ theories of the Fourier Transform handle a large class of functions, but they exclude some of the most important objects in signal processing and physics:

- $\delta(t)$ - the Dirac delta: not a function, but models an ideal impulse
- $f(t) = 1$ - the constant function: not in $L^1$ or $L^2$, but models a DC signal
- $f(t) = e^{2\pi i\xi_0 t}$ - a pure tone: not in $L^1$ or $L^2$, but is a fundamental signal
- $\sum_{n=-\infty}^{\infty}\delta(t - nT)$ - the Dirac comb: not a function, but models periodic sampling

All of these have Fourier Transforms in the **distributional sense** - and all appear in practical signal processing. The distribution framework, developed by Laurent Schwartz in the 1950s, extends the Fourier Transform to this broader class.

### 7.2 Schwartz Space and Tempered Distributions

**Definition 7.1 (Schwartz Space).** The **Schwartz space** $\mathcal{S}(\mathbb{R})$ is the space of all $C^\infty$ functions $\phi: \mathbb{R} \to \mathbb{C}$ such that for all $m, n \geq 0$:
$$\sup_{t \in \mathbb{R}} \left|t^m \phi^{(n)}(t)\right| < \infty$$

i.e., $\phi$ and all its derivatives decay faster than any polynomial. Examples: $e^{-t^2}$, $e^{-|t|}$ (the derivative condition fails at 0 - not Schwartz), Gaussian bump functions.

The Schwartz space is closed under the Fourier Transform: if $\phi \in \mathcal{S}$, then $\hat{\phi} \in \mathcal{S}$. This is the key property: the FT maps $\mathcal{S}$ to itself bijectively.

**Definition 7.2 (Tempered Distributions).** A **tempered distribution** $T$ is a continuous linear functional $T: \mathcal{S}(\mathbb{R}) \to \mathbb{C}$. We write $T(\phi) = \langle T, \phi \rangle$.

**Examples of tempered distributions:**

1. **Regular distributions:** Every function $f \in L^p(\mathbb{R})$ ($1 \leq p \leq \infty$) defines a tempered distribution $T_f(\phi) = \int f(t)\phi(t)\,dt$.

2. **Dirac delta:** $\langle\delta, \phi\rangle = \phi(0)$. The Dirac delta evaluates $\phi$ at 0. It is NOT a function.

3. **Derivative of delta:** $\langle\delta', \phi\rangle = -\phi'(0)$.

4. **Dirac comb:** $\langle\text{III}_T, \phi\rangle = \sum_{n=-\infty}^\infty \phi(nT)$.

**Fourier Transform of a tempered distribution:** Define $\langle\hat{T}, \phi\rangle = \langle T, \hat{\phi}\rangle$ for $\phi \in \mathcal{S}$. This is the "transpose" of the Fourier Transform.

### 7.3 The Dirac Delta and its Fourier Transform

**The Dirac delta** $\delta(t)$ models a unit impulse concentrated at $t = 0$:
$$\langle\delta, \phi\rangle = \phi(0) = \int_{-\infty}^{\infty} \delta(t)\phi(t)\,dt \quad \text{(formal)}$$

It is the limit of a sequence of ordinary functions: $\delta_\epsilon(t) = \frac{1}{\epsilon}e^{-\pi t^2/\epsilon^2}$ converges to $\delta$ in the distributional sense as $\epsilon \to 0$.

**Fourier Transform of $\delta$:** By the definition $\langle\hat{\delta}, \phi\rangle = \langle\delta, \hat{\phi}\rangle = \hat{\phi}(0) = \int\phi(t)\cdot 1\,dt = \langle 1, \phi\rangle$. Therefore:
$$\mathcal{F}[\delta](\xi) = 1$$

An ideal impulse at $t = 0$ has a **flat (white) spectrum** - it contains equal energy at all frequencies. This makes perfect physical sense: to create a perfect impulse, you need all frequencies constructively interfering at $t = 0$.

**Fourier Transform of a constant:** By duality (applying FT twice): $\mathcal{F}[1](\xi) = \delta(\xi)$. A constant signal (DC) lives entirely at frequency $\xi = 0$.

**The duality principle in full:** The FT maps $\delta(t) \leftrightarrow 1$ and $1 \leftrightarrow \delta(\xi)$ - a striking symmetry.

**More generally:** A delta at $t = a$:
$$\mathcal{F}[\delta(t-a)](\xi) = e^{-2\pi i\xi a}$$

This is the time-shift property: $\delta(t - a)$ shifted by $a$ gives a phase factor $e^{-2\pi i\xi a}$, but the magnitude is still flat: $|\mathcal{F}[\delta(t-a)]| = 1$.

### 7.4 FT of Constants, Sinusoids, and Periodic Signals

Using $\mathcal{F}[1] = \delta$ and the frequency-shift property $\mathcal{F}[e^{2\pi i\xi_0 t}f(t)](\xi) = \hat{f}(\xi - \xi_0)$:

**Complex exponential:**
$$\mathcal{F}[e^{2\pi i\xi_0 t}](\xi) = \delta(\xi - \xi_0)$$

**Cosine and Sine:**
$$\mathcal{F}[\cos(2\pi\xi_0 t)](\xi) = \frac{1}{2}[\delta(\xi - \xi_0) + \delta(\xi + \xi_0)]$$
$$\mathcal{F}[\sin(2\pi\xi_0 t)](\xi) = \frac{1}{2i}[\delta(\xi - \xi_0) - \delta(\xi + \xi_0)]$$

A pure cosine has a spectrum consisting of two Dirac deltas, one at $+\xi_0$ and one at $-\xi_0$ (the negative frequency is the mirror image required by Hermitian symmetry). The amplitude of each spike is $1/2$ because the energy is split equally between the two.

**General periodic signal:** A $T$-periodic function $f(t) = \sum_{n=-\infty}^\infty c_n e^{2\pi i nt/T}$ has Fourier Transform:
$$\hat{f}(\xi) = \sum_{n=-\infty}^\infty c_n\,\delta\!\left(\xi - \frac{n}{T}\right)$$

The Fourier Transform of a periodic signal is a discrete sum of Dirac deltas at the harmonics - precisely the discrete spectrum of the Fourier series! This shows that Fourier series and Fourier Transform are two aspects of the same theory, unified by the distributional framework.

### 7.5 The Dirac Comb and Poisson Summation Formula

**Definition 7.3 (Dirac Comb).** The $T$-periodic Dirac comb is:
$$\text{III}_T(t) = \sum_{n=-\infty}^{\infty} \delta(t - nT)$$

**Theorem 7.1 (FT of the Dirac Comb).**
$$\mathcal{F}[\text{III}_T](\xi) = \frac{1}{T}\,\text{III}_{1/T}(\xi) = \frac{1}{T}\sum_{n=-\infty}^{\infty} \delta\!\left(\xi - \frac{n}{T}\right)$$

A comb of spacing $T$ in time has a spectrum that is a comb of spacing $1/T$ in frequency. This is the time-frequency duality made concrete: fine temporal sampling -> coarse frequency sampling, and vice versa.

**Theorem 7.2 (Poisson Summation Formula).** For $f \in \mathcal{S}(\mathbb{R})$:
$$\sum_{n=-\infty}^{\infty} f(n) = \sum_{n=-\infty}^{\infty} \hat{f}(n)$$

More generally, for sampling at interval $T$:
$$\frac{1}{T}\sum_{n=-\infty}^{\infty} f\!\left(\frac{n}{T}\right) = \sum_{n=-\infty}^{\infty} \hat{f}(nT)$$

*Proof:* The function $g(t) = \sum_n f(t + n)$ is $1$-periodic, so it has a Fourier series with coefficients $c_k = \hat{f}(k)$ (by computing $\int_0^1 g(t)e^{-2\pi ikt}\,dt = \int_{-\infty}^\infty f(t)e^{-2\pi ikt}\,dt = \hat{f}(k)$). Evaluating at $t = 0$: $\sum_n f(n) = g(0) = \sum_k c_k = \sum_k\hat{f}(k)$. $\square$

**The Poisson summation formula is the key to sampling theory.** Shannon's sampling theorem (1949) follows directly: a signal $f$ with bandwidth $B$ (i.e., $\hat{f}(\xi) = 0$ for $|\xi| > B$) is completely determined by its samples at rate $\geq 2B$ (the Nyquist rate). The Poisson summation formula explains both the reconstruction formula and the aliasing that occurs when undersampling.

**For AI:** The discrete Fourier Transform (Section 03) is the computational realization of the Poisson summation formula. The FFT computes the sampled Fourier Transform exactly, which requires the signal to be band-limited (or periodic) - any violation causes aliasing, the discrete analog of spectral leakage.

### 7.6 Unifying Fourier Series and Fourier Transform

The distribution framework reveals that Fourier series and Fourier Transform are the same thing:

| Situation | Signal | Spectrum |
|-----------|--------|---------|
| General $L^2$ signal | Continuous, aperiodic | Continuous, aperiodic |
| Periodic signal (period $T$) | Continuous, periodic | Discrete (spacing $1/T$): Fourier series |
| Bandlimited (bandwidth $B$) | Discrete (rate $\geq 2B$): sampling | Periodic (period $\geq 2B$): aliases |
| Periodic AND bandlimited | Discrete and periodic | Discrete and periodic: DFT (Section 03) |

```
FOURIER TRANSFORM UNIFICATION
========================================================================

  Time domain           Frequency domain        Setting
  -----------           ----------------        -------
  Continuous            Continuous              L1/L2 Fourier Transform
  Continuous            Discrete                Fourier Series (periodic)
  Discrete              Continuous              DTFT (sampled signal)
  Discrete              Discrete                DFT / FFT -> see Section 20-03

  All four are the same theory viewed through the distributional lens.
  The Dirac comb + Poisson summation connects all four cases.

========================================================================
```

## 8. Applications in Machine Learning

### 8.1 FNet: Replacing Self-Attention with Fourier Transforms

**Paper:** Lee-Thorp, Ainslie, Eckstein, Ontanon (2022). "FNet: Mixing Tokens with Fourier Transforms." NAACL.

**The idea:** In a standard Transformer, the self-attention sublayer computes:
$$\text{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
which is $O(N^2 d)$ in sequence length $N$. FNet replaces this with a 2D DFT:
$$\text{FNet-mixing}(X) = \operatorname{Re}\!\left(\mathcal{F}_{\text{seq}}\!\left[\mathcal{F}_{\text{model}}[X]\right]\right)$$
where $\mathcal{F}_{\text{seq}}$ applies FFT along the sequence dimension and $\mathcal{F}_{\text{model}}$ along the embedding dimension. This is $O(N\log N \cdot d)$ - much faster.

**Why does it work?** The Fourier Transform is a global linear mixing operation: every output position depends on every input position (via the sum $\hat{x}_k = \sum_n x_n e^{-2\pi ink/N}$). This is similar to attention's global mixing, but unparameterized - the mixing weights are fixed Fourier basis functions rather than learned attention scores.

**Results:** FNet achieves 92-97% of BERT's performance on GLUE benchmarks. For tasks requiring fine-grained token interaction (e.g., extractive QA), the gap is larger; for tasks where global context suffices (e.g., sentence classification), FNet nearly matches BERT. Training speed is 7 faster on GPUs, 2 faster on TPUs.

**Mathematical insight:** The key theorem is that the DFT matrix $F$ satisfies $F = F^\top / N$ (circular DFT is symmetric up to normalization). Therefore $FF^\dagger = I$ - the DFT is unitary, just like the continuous FT by Plancherel. The real part operation ensures the output is real-valued.

**Code sketch:**
```python
import numpy as np

def fnet_mixing(X):
    """
    X: (batch, seq_len, d_model) array
    Returns: real-valued mixing output of same shape
    """
    X_fft = np.fft.fft(np.fft.fft(X, axis=1), axis=2)
    return np.real(X_fft)
```

### 8.2 Random Fourier Features

**Paper:** Rahimi & Recht (2007). "Random Features for Large-Scale Kernel Machines." NeurIPS (Best Paper Award).

**The setup:** Kernel methods (SVMs, Gaussian Processes) require computing $K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ for all $N$ training pairs - an $O(N^2)$ matrix that costs $O(N^3)$ to invert. For $N > 10^5$, this is prohibitive.

**Bochner's theorem:** A continuous, shift-invariant kernel $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{x} - \mathbf{y})$ is **positive definite** if and only if it is the Fourier Transform of a non-negative measure $p(\boldsymbol{\omega})$:
$$k(\mathbf{x} - \mathbf{y}) = \int p(\boldsymbol{\omega})\,e^{i\boldsymbol{\omega}^\top(\mathbf{x}-\mathbf{y})}\,d\boldsymbol{\omega} = \mathbb{E}_{\boldsymbol{\omega} \sim p}[e^{i\boldsymbol{\omega}^\top\mathbf{x}}\overline{e^{i\boldsymbol{\omega}^\top\mathbf{y}}}]$$

(Full treatment of Bochner's theorem in [Section 12-03 Kernel Methods](../../12-Functional-Analysis/03-Kernel-Methods/notes.md).)

**The approximation:** Sample $D$ frequencies $\boldsymbol{\omega}_1, \ldots, \boldsymbol{\omega}_D \sim p(\boldsymbol{\omega})$ and define the random feature map:
$$\phi(\mathbf{x}) = \sqrt{\frac{2}{D}}\left[\cos(\boldsymbol{\omega}_1^\top\mathbf{x} + b_1), \ldots, \cos(\boldsymbol{\omega}_D^\top\mathbf{x} + b_D)\right] \in \mathbb{R}^D$$
where $b_j \sim \mathcal{U}[0, 2\pi]$ are random phase offsets.

**Guarantee:** By the law of large numbers:
$$k(\mathbf{x}, \mathbf{y}) \approx \phi(\mathbf{x})^\top\phi(\mathbf{y}), \qquad \mathbb{E}[\phi(\mathbf{x})^\top\phi(\mathbf{y})] = k(\mathbf{x},\mathbf{y})$$

with concentration: $P\!\left(|\phi(\mathbf{x})^\top\phi(\mathbf{y}) - k(\mathbf{x},\mathbf{y})| > \epsilon\right) \leq 2\exp(-D\epsilon^2/4)$.

**For popular kernels:**
- **RBF kernel** $k(\mathbf{x},\mathbf{y}) = e^{-\lVert\mathbf{x}-\mathbf{y}\rVert^2/(2\sigma^2)}$: $p(\boldsymbol{\omega}) = \mathcal{N}(\mathbf{0}, \sigma^{-2}I)$
- **Laplace kernel** $k(\mathbf{x},\mathbf{y}) = e^{-\lVert\mathbf{x}-\mathbf{y}\rVert_1/\sigma}$: $p$ is a Cauchy distribution

**Impact:** Random Fourier Features reduce kernel SVM training from $O(N^3)$ to $O(ND + D^3)$, making kernel methods scalable. The same idea appears in **Performer** (Choromanski et al., 2021) - an efficient attention mechanism that approximates the attention kernel with random features, reducing attention to $O(N)$.

### 8.3 Spectral Normalization

**Paper:** Miyato, Kataoka, Koyama, Yoshida (2018). "Spectral Normalization for Generative Adversarial Networks." ICLR.

**The problem:** Training GANs is notoriously unstable because the discriminator can overfit, causing gradient vanishing for the generator. A Lipschitz constraint on the discriminator stabilizes training.

**The solution:** Normalize each weight matrix $W$ by its spectral norm $\lVert W \rVert_2 = \sigma_{\max}(W)$ (the largest singular value):
$$\tilde{W} = \frac{W}{\sigma_{\max}(W)}$$

This makes every linear layer 1-Lipschitz: $\lVert\tilde{W}\mathbf{x} - \tilde{W}\mathbf{y}\rVert \leq \lVert\mathbf{x} - \mathbf{y}\rVert$. The composition of 1-Lipschitz layers is 1-Lipschitz - so the entire discriminator is 1-Lipschitz.

**Computing $\sigma_{\max}$:** Power iteration gives an efficient approximation. Starting from a random unit vector $\tilde{\mathbf{v}}$:
1. $\tilde{\mathbf{u}} = W\tilde{\mathbf{v}} / \lVert W\tilde{\mathbf{v}} \rVert$
2. $\tilde{\mathbf{v}} = W^\top\tilde{\mathbf{u}} / \lVert W^\top\tilde{\mathbf{u}} \rVert$
3. $\hat{\sigma} = \tilde{\mathbf{u}}^\top W\tilde{\mathbf{v}}$
One iteration per training step suffices empirically.

**Why "spectral"?** The spectral norm $\lVert W \rVert_2$ is the largest eigenvalue of $\sqrt{W^\top W}$ - the spectrum of the operator $W$ viewed as a linear map. This connects to Fourier analysis: for a convolution layer with kernel $h$, the spectral norm is $\lVert\hat{h}\rVert_\infty = \max_\xi|\hat{h}(\xi)|$ - the supremum of the Fourier Transform of the filter.

**For AI:** Spectral normalization is now standard in GAN training. It also appears in **attention normalization** - normalizing the key/value projection matrices to control the Lipschitz constant of the attention map, which is critical for stable training of large models.

### 8.4 Fourier Neural Operator - Preview

The **Fourier Neural Operator (FNO)** (Li et al., 2021) learns a mapping between function spaces (e.g., PDE initial conditions -> solutions) by:

1. Lift input $\mathbf{v}(t)$ to a higher-dimensional representation $\mathbf{v}_0 = P(\mathbf{v})$
2. Apply $L$ Fourier integral operator layers: $\mathbf{v}_{l+1}(t) = \sigma(W\mathbf{v}_l(t) + \mathcal{F}^{-1}[R_\phi \cdot \mathcal{F}[\mathbf{v}_l]](t))$
3. Project to output: $u = Q(\mathbf{v}_L)$

The key operation $R_\phi$ is a **learnable complex-valued tensor** applied pointwise in Fourier space - a global convolution with a learned filter. Only the lowest $k_{\max}$ Fourier modes are kept (Plancherel guarantees the truncation error equals the discarded energy).

FNO achieves 1000 speedup over classical PDE solvers for problems like weather prediction and turbulence simulation.

> **Full treatment in [Section 20-03 DFT and FFT](../03-Discrete-Fourier-Transform-and-FFT/notes.md)** - the discretized version of FNO and its implementation via FFT is covered there.

### 8.5 Bochner's Theorem and Shift-Invariant Kernels - Preview

**Bochner's theorem** (1932) characterizes all positive-definite, shift-invariant kernels as Fourier Transforms of non-negative measures:

$$k(\mathbf{x} - \mathbf{y}) = \int_{\mathbb{R}^d} e^{i\boldsymbol{\omega}^\top(\mathbf{x}-\mathbf{y})}\,dp(\boldsymbol{\omega})$$

This is a profound connection between Fourier analysis and kernel methods. The kernel's shape in the spatial domain determines (and is determined by) its spectral density $p$.

> **Full treatment in [Section 12-03 Kernel Methods](../../12-Functional-Analysis/03-Kernel-Methods/notes.md)** - the RKHS theory, Mercer's theorem, and applications to SVMs, Gaussian Processes, and the Neural Tangent Kernel are covered there. Here we note that Bochner's theorem is the mathematical foundation for Random Fourier Features (Section 8.2).

### 8.6 RoPE from the Continuous FT Perspective

**Rotary Position Embedding (RoPE)** (Su, Lu, Pan, Meng, Luo, 2021) is now used in LLaMA-3, Mistral, Qwen, GPT-NeoX, and virtually every frontier LLM. Its mathematical foundation is the Fourier Transform on the circle group.

**Setup.** In attention, the score between query $\mathbf{q}$ at position $m$ and key $\mathbf{k}$ at position $n$ should depend only on the **relative position** $m - n$. This is a **translation invariance** requirement - exactly the condition Bochner's theorem characterizes.

**Construction.** Pair up embedding dimensions: $(q_{2j}, q_{2j+1})$ and $(k_{2j}, k_{2j+1})$ for $j = 0, \ldots, d/2 - 1$. Treat each pair as a complex number $q_j^{(c)} = q_{2j} + iq_{2j+1}$. Apply rotation by angle $m\theta_j$:

$$f_q(\mathbf{q}, m) = q_j^{(c)} \cdot e^{im\theta_j} \qquad \theta_j = 10000^{-2j/d_{\text{model}}}$$

The attention score becomes:
$$\langle f_q(\mathbf{q}, m), f_k(\mathbf{k}, n)\rangle = \operatorname{Re}\!\left[\sum_j q_j^{(c)}\overline{k_j^{(c)}} \cdot e^{i(m-n)\theta_j}\right]$$

which depends only on $m - n$ - the relative position. This is the Fourier modulation theorem: multiplying by $e^{im\theta_j}$ is a frequency shift by $m$ in the "position domain."

**The frequencies $\theta_j$:** The geometric spacing $\theta_j = 10000^{-2j/d}$ means each pair encodes a different "octave" of positional information. Low-index dimensions ($j$ small) use high frequencies $\theta_j \approx 1$ - sensitive to local position differences. High-index dimensions use low frequencies $\theta_j \approx 10000^{-1}$ - sensitive to global structure over thousands of positions. This is a **multi-resolution** decomposition - the Fourier Transform at multiple frequency scales.

**Why the $10000$ base?** The maximum useful context is $\sim 10000$ tokens at the original RoPE scale (each frequency completes at most one full rotation). For LLaMA-3 with 128K context, the effective base must be much larger - which is why LLaMA-3.1 uses $\theta_{\text{base}} = 500000$ instead of $10000$.

```
ROPE: FOURIER TRANSFORM IN ATTENTION
========================================================================

  f_q(q, m) = q  e^{im}   <- modulation by Fourier basis e^{im}

  f_q(q,m), f_k(k,n) = Re[q*k  e^{i(m-n)}]
                            ^ depends only on relative position m-n

  j = 10000^{-2j/d}:  j=0 -> approx1 (fast/local),  j=d/2 -> approx1/10000 (slow/global)

  Context extension:
    Original RoPE: base=10000, max context ~10K tokens
    LLaMA-3.1:     base=500K,  max context 128K tokens
    LongRoPE:      non-uniform rescaling up to 2M context

========================================================================
```

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | Confusing the $\omega$ and $\xi$ conventions and getting $2\pi$ factors wrong | $\mathcal{F}_\omega[f](f')=\hat{f}(\omega)$ has a $1/(2\pi)$ in the inverse; $\mathcal{F}_\xi$ does not. Mixing these adds spurious factors | Always identify the convention on first use; for this curriculum use the $\xi$-convention $\hat{f}(\xi)=\int f(t)e^{-2\pi i\xi t}\,dt$ |
| 2 | Assuming $\hat{f} \in L^1$ whenever $f \in L^1$ | $f=\operatorname{rect}$ has $\hat{f}=\operatorname{sinc} \notin L^1$. The Riemann-Lebesgue lemma only guarantees $\hat{f} \in C_0$, not $L^1$ | Check whether $f$ is smooth enough (e.g., $f \in L^1$ and Lipschitz) for $\hat{f} \in L^1$ to hold |
| 3 | Writing $\mathcal{F}[\delta](\xi) = 0$ | $\delta$ has a flat spectrum: $\mathcal{F}[\delta](\xi)=1$ for all $\xi$. A "pointlike" impulse contains all frequencies equally | Compute distributional FT: $\langle\hat{\delta},\phi\rangle=\langle\delta,\hat{\phi}\rangle=\hat{\phi}(0)=\int\phi\,dt$ |
| 4 | Thinking the uncertainty principle only limits our instruments | $\Delta t\cdot\Delta\xi\geq 1/(4\pi)$ is a theorem about functions, not about measurement devices. It holds for any $f\in L^2$, regardless of how it is measured | Accept the bound as a mathematical fact; designing around it requires fundamentally different time-frequency representations (wavelets, Section 05) |
| 5 | Applying the differentiation property to non-differentiable $f$ | $\mathcal{F}[f'](\xi)=2\pi i\xi\hat{f}(\xi)$ requires $f'\in L^1$. For $f=\operatorname{rect}$, $f'=\delta(t+1/2)-\delta(t-1/2)$ in the distributional sense | Work in the distributional setting when $f$ has jumps; the formula still holds but $f'$ is a distribution |
| 6 | Claiming Parseval says $\int|f|^2 = \int|\hat{f}|^2$ for $f\in L^1$ only | Parseval requires $f\in L^2$. A function in $L^1$ but not $L^2$ does not satisfy Parseval | Verify $f\in L^2$ before applying Parseval; for $f\in L^1\setminus L^2$, the $L^2$ norm may not exist |
| 7 | Mixing up the FT of a product vs. convolution | $\mathcal{F}[f\cdot g]=\hat{f}*\hat{g}$ (convolution in frequency), not $\hat{f}\cdot\hat{g}$. The multiplicative dual of the convolution theorem is often reversed | Memorize the duality table: convolution in time $\leftrightarrow$ product in frequency; product in time $\leftrightarrow$ convolution in frequency |
| 8 | Forgetting the $1/|a|$ in the scaling theorem | $\mathcal{F}[f(at)](\xi)=\frac{1}{|a|}\hat{f}(\xi/a)$, not $\hat{f}(\xi/a)$. The amplitude factor ensures Parseval is preserved under scaling | Derive it each time: change variables $u=at$, $du=a\,dt$, so $dt=du/|a|$ |
| 9 | Treating negative frequencies as "unphysical" | For real $f$, negative frequencies are redundant (Hermitian symmetry) but are not wrong. For complex $f$ (e.g., analytic signals), negative and positive frequencies are distinct and carry different information | Embrace negative frequencies; they simplify formulas and are essential for complex signals |
| 10 | Applying $\mathcal{F}^{-1}\circ\mathcal{F}=I$ pointwise instead of a.e. | Inversion holds almost everywhere (for $L^1\cap$ sufficient regularity). At a jump discontinuity, $\int\hat{f}(\xi)e^{2\pi i\xi t}\,d\xi=\frac{1}{2}[f(t^+)+f(t^-)]$ - the average of left and right limits | Account for the $1/2$-average behavior at discontinuities (same as Dirichlet's theorem in Section 01) |
| 11 | Using $\text{sinc}(x) = \sin(x)/x$ and $\operatorname{sinc}(x) = \sin(\pi x)/(\pi x)$ interchangeably | There are two conventions for sinc. The normalized sinc $\operatorname{sinc}(x)=\sin(\pi x)/(\pi x)$ satisfies $\mathcal{F}[\operatorname{rect}]=\operatorname{sinc}$ in the $\xi$-convention; the unnormalized $\sin(x)/x$ does not | In this curriculum: $\operatorname{sinc}(x) = \sin(\pi x)/(\pi x)$ (normalized), consistent with the $\xi$-convention FT |
| 12 | Thinking FNet replaces attention entirely | FNet is a weaker mixer than attention - it cannot learn data-dependent weights. It works well for classification but poorly for tasks requiring fine-grained token interactions (QA, coreference) | Use FNet as a fast baseline; switch to attention for tasks requiring selective, query-dependent mixing |


## 10. Exercises

**Exercise 1** * - Computing Fourier Transforms from Definition

**(a)** Compute $\mathcal{F}[f](\xi)$ for $f(t) = e^{-a|t|}$ with $a > 0$. Show all steps of integration. What type of function is $\hat{f}$?

**(b)** Compute $\mathcal{F}[\operatorname{rect}(t/T)](\xi)$ for a pulse of width $T > 0$. Express using the normalized sinc function.

**(c)** Verify the $L^1$ bound: show $|\hat{f}(\xi)| \leq \lVert f\rVert_1$ for your answer in (a).

**(d)** Confirm the Riemann-Lebesgue lemma numerically for $f(t) = e^{-t^2}$: plot $|\hat{f}(\xi)|$ and verify it decays to 0.

---

**Exercise 2** * - Applying Properties

**(a)** Given $\hat{f}(\xi) = e^{-\pi\xi^2}$, use the time-shift property to write the Fourier Transform of $g(t) = f(t - 3)$.

**(b)** Given $\hat{h}(\xi) = \operatorname{sinc}(\xi)$, find the Fourier Transform of $h(t - 2)\cos(4\pi t)$. (Use shift + modulation.)

**(c)** If $f(t)$ has $\hat{f}(\xi) = g(\xi)$, what is the FT of $f(3t + 1)$? (Use both shift and scaling.)

**(d)** Prove: if $f$ is real and even, then $\hat{f}$ is real and even.

---

**Exercise 3** * - Parseval's Relation Numerically

**(a)** For $f(t) = e^{-\pi t^2}$, compute $\lVert f\rVert_2^2 = \int|f|^2\,dt$ analytically.

**(b)** Compute $\lVert\hat{f}\rVert_2^2$ analytically and verify $\lVert f\rVert_2^2 = \lVert\hat{f}\rVert_2^2$.

**(c)** Numerically verify Parseval using the FFT: discretize $f$ on $[-5, 5]$ with $N = 1024$ points, apply FFT, and compare $\sum|f_n|^2/N$ vs $\sum|\hat{f}_k|^2/N$.

**(d)** Use Parseval to evaluate $\int_{-\infty}^\infty \frac{1}{(1 + 4\pi^2 t^2)^2}\,dt$.

---

**Exercise 4** ** - The Inversion Theorem

**(a)** For $f(t) = \operatorname{rect}(t)$ (not in $L^1$ after FT since $\operatorname{sinc} \notin L^1$), show that the principal value integral $\lim_{R\to\infty}\int_{-R}^{R}\operatorname{sinc}(\xi)e^{2\pi i\xi t}\,d\xi$ converges to $\operatorname{rect}(t)$ for $|t| \neq 1/2$ by evaluating the integral explicitly.

**(b)** Show that at $t = 1/2$ (the jump discontinuity), the principal value integral converges to $1/2 = \frac{1}{2}[\operatorname{rect}(1/2^-) + \operatorname{rect}(1/2^+)]$.

**(c)** This mirrors Dirichlet's theorem from Section 20-01. Identify the precise analog: what plays the role of the Dirichlet kernel in the Fourier integral case?

---

**Exercise 5** ** - Uncertainty Principle

**(a)** For $f(t) = e^{-\pi t^2/\sigma^2}$ (normalized to $\lVert f\rVert_2 = 1$ appropriately), compute $\Delta t$ (RMS time spread) and $\Delta\xi$ (RMS frequency spread). Verify $\Delta t\cdot\Delta\xi = 1/(4\pi)$.

**(b)** For $f(t) = \operatorname{rect}(t)$, compute $\Delta t$ and $\Delta\xi$ numerically. Is the uncertainty bound satisfied? Is it tight?

**(c)** A signal designer wants to transmit a pulse with duration $\Delta t \leq 1\,\mu$s using bandwidth $\Delta\xi \leq 100\,$kHz. Is this possible? What does the uncertainty principle say?

**(d)** Show that any signal of the form $f(t) = Ce^{-\alpha t^2}e^{2\pi i\xi_0 t}$ achieves equality in the uncertainty principle.

---

**Exercise 6** ** - Tempered Distributions

**(a)** Verify $\mathcal{F}[\delta(t - a)](\xi) = e^{-2\pi i\xi a}$ using the distributional definition $\langle\hat{T},\phi\rangle = \langle T,\hat{\phi}\rangle$.

**(b)** Compute $\mathcal{F}[\delta'(t)](\xi)$ (FT of the derivative of delta).

**(c)** Use the result from (a) to verify: $\mathcal{F}[\cos(2\pi\xi_0 t)](\xi) = \frac{1}{2}[\delta(\xi - \xi_0) + \delta(\xi + \xi_0)]$.

**(d)** Verify the Poisson summation formula numerically for $f(t) = e^{-\pi t^2}$: compare $\sum_{n=-N}^{N} f(n)$ with $\sum_{n=-N}^{N} \hat{f}(n)$ for $N = 5, 10, 20$.

---

**Exercise 7** *** - Random Fourier Features

**(a)** For the RBF kernel $k(\mathbf{x},\mathbf{y}) = e^{-\lVert\mathbf{x}-\mathbf{y}\rVert^2}$ (setting $\sigma = 1$), the spectral density is $p(\boldsymbol{\omega}) = (4\pi)^{-d/2}e^{-\lVert\boldsymbol{\omega}\rVert^2/4}$. Draw $D = 500$ frequencies $\boldsymbol{\omega}_j \sim \mathcal{N}(\mathbf{0}, 2I)$ for $d = 2$ and construct the random feature map $\phi(\mathbf{x}) \in \mathbb{R}^{2D}$ (using cos and sin).

**(b)** Generate 100 random pairs $(\mathbf{x}_i, \mathbf{y}_i)$ with $\mathbf{x}_i, \mathbf{y}_i \sim \mathcal{N}(\mathbf{0}, I)$. For each pair, compute the exact kernel value $k(\mathbf{x}_i, \mathbf{y}_i)$ and the RFF approximation $\phi(\mathbf{x}_i)^\top\phi(\mathbf{y}_i)$. Plot the approximation vs. exact values and report RMSE.

**(c)** Study the approximation quality as a function of $D \in \{10, 50, 100, 500, 1000\}$. How does RMSE scale with $D$? Is it consistent with the theoretical bound $\mathbb{E}[\text{error}^2] = O(D^{-1})$?

**(d)** Takeaway: explain in 2 sentences why RFF matters for LLM-scale systems where kernel computation would be infeasible.

---

**Exercise 8** *** - FNet vs Attention Experiment

**(a)** Implement a minimal FNet mixing layer: given $X \in \mathbb{R}^{N \times d}$, apply 2D FFT and take the real part. Implement a minimal attention mixing layer: $\text{Attn}(X) = \operatorname{softmax}(XX^\top/\sqrt{d})X$.

**(b)** For a synthetic task - "copy the token from position $k$ to the output" (a task requiring precise position tracking) - construct training pairs $(X, y)$ and train both mixers with a linear head on top. Compare accuracy and training speed.

**(c)** For a synthetic task - "output the mean of all input tokens" (a global aggregation task) - repeat the comparison. Which mixer is better suited? Why?

**(d)** Takeaway: connect your results to the mathematical properties of the FT (fixed unparameterized mixing vs. learned data-dependent mixing) and explain what types of tasks each mixer is appropriate for.


## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
|---------|-----------|
| Fourier Transform as unitary operator (Plancherel) | FNet (Lee-Thorp et al., 2022) uses FT as a token mixer - same information content, 7 faster than attention on GPU. The unitarity of FT ensures no information loss during mixing. |
| Convolution-multiplication duality | The theoretical basis for $O(N\log N)$ convolution via FFT - the reason CNNs can be trained efficiently. FlashConv (Fu et al., 2023) and Hyena (Poli et al., 2023) extend this to long-range sequence models. |
| Modulation theorem (time-shift  frequency-shift) | Foundation of RoPE (Su et al., 2021) - multiplying embeddings by $e^{im\theta}$ shifts the position representation. Now standard in LLaMA-3, Mistral, Qwen, GPT-NeoX. |
| Uncertainty principle $\Delta t\cdot\Delta\xi \geq 1/(4\pi)$ | Fundamental constraint on context length extension. YaRN (Peng et al., 2023) and LongRoPE (Ding et al., 2024) must lower the frequency $\theta_j$ to extend context - the UP dictates the minimum frequency needed. |
| Bochner's theorem + FT of spectral density | Random Fourier Features (Rahimi & Recht, 2007) reduce kernel computation from $O(N^2)$ to $O(ND)$. Performer (Choromanski et al., 2021) uses the same idea to get $O(N)$ attention. |
| Spectral norm $\lVert W\rVert_2 = \sigma_{\max}$ | Spectral normalization (Miyato et al., 2018) - dividing discriminator weights by their spectral norm enforces Lipschitz constraints. Standard in GAN training and used in diffusion model discriminators. |
| FT as basis for function space operators | Fourier Neural Operator (Li et al., 2021) - solves PDEs 1000 faster by learning in Fourier space. Used in climate modeling, fluid simulation, and materials science. |
| Differentiation in frequency domain ($f' \leftrightarrow 2\pi i\xi\hat{f}$) | Heat and wave equation solutions in closed form via FT. Kernel of the heat equation is a Gaussian - explains why diffusion models add Gaussian noise: it has maximal spectral support. |
| Gaussian as optimal time-frequency window | Gaussian embeddings in Gaussian attention (Tsai et al., 2019) - replacing dot-product attention with a Gaussian kernel gives an attention with explicit time-frequency interpretation. |
| Dirac comb + Poisson summation | Shannon sampling theorem - ensures that Whisper's mel-spectrogram preprocessing (16kHz sampling, 80-channel mel filterbank) captures all speech frequencies (up to 8kHz) without aliasing. |

## 12. Conceptual Bridge

### Looking Back: From Fourier Series to Fourier Transform

The Fourier series (Section 20-01) established that **periodic functions decompose into discrete harmonics**. The key insight was geometric: the trigonometric functions $\{e^{inx}\}$ form an orthonormal basis of $L^2[-\pi,\pi]$, and the Fourier coefficients are the projections of $f$ onto these basis vectors.

The Fourier Transform carries this insight to its natural conclusion. By letting the period $T \to \infty$, the discrete harmonic sum $\sum_n c_n e^{2\pi int/T}$ becomes the continuous integral $\int\hat{f}(\xi)e^{2\pi i\xi t}\,d\xi$, and the discrete spectrum $\{c_n\}$ becomes the continuous spectrum $\hat{f}(\xi)$. Everything from the Fourier series has an analog:
- Orthonormality -> Plancherel's theorem (Section 5)
- Parseval's identity -> Parseval's relation (Section 5.2)
- Dirichlet's theorem on convergence -> Fourier inversion theorem (Section 4)
- Gibbs phenomenon -> spectral leakage in DFT (Section 03)
- Smoothness  spectral decay -> differentiation property (Section 3.5)

The distribution theory in Section 7 reveals the deepest unity: Fourier series is just the Fourier Transform restricted to periodic distributions. The Dirac comb with spacing $T$ transforms to a Dirac comb with spacing $1/T$ - the very relationship between period and harmonic spacing that the Fourier series encodes.

### Looking Forward: Three Branches

From the Fourier Transform, the curriculum branches in three directions:

**Branch 1: Discretization -> Section 20-03 DFT and FFT.** The continuous FT must be made computable. Discretizing both time and frequency leads to the DFT, and the Cooley-Tukey algorithm (1965) computes it in $O(N\log N)$ instead of $O(N^2)$. The FFT is arguably the most important algorithm in scientific computing - it is what makes spectrograms, MRI, and FNet feasible. The DFT inherits all properties of the continuous FT but adds complications: aliasing (from discretizing frequency), spectral leakage (from finite duration), and the circular convolution structure.

**Branch 2: Convolution Theorem -> Section 20-04.** The most important property of the FT for applications is that convolution in time equals multiplication in frequency. This converts the $O(N^2)$ convolution operation into $O(N\log N)$ FFT-based multiplication. The convolution theorem is the mathematical foundation of CNNs, WaveNet, S4/Mamba, and Hyena - all architectures where the key operation is a learned filter applied by convolution. The full theory of LTI systems, frequency response, filter design, and the Wiener-Khinchin theorem belongs there.

**Branch 3: Time-Frequency Localization -> Section 20-05 Wavelets.** The FT has a fundamental limitation: $|\hat{f}(\xi)|$ tells you which frequencies are present but not *when* they occur. The uncertainty principle shows this is not fixable - any attempt to localize in both time and frequency is bounded by $\Delta t\cdot\Delta\xi \geq 1/(4\pi)$. Wavelets overcome this by accepting a principled tradeoff: high frequencies are analyzed at fine time resolution, low frequencies at coarse time resolution. This multi-resolution analysis (MRA) is the mathematical foundation of JPEG 2000, EEG analysis, and wavelet attention mechanisms.

```
POSITION IN THE FOURIER CURRICULUM
========================================================================

  Section 01 FOURIER SERIES                Section 02 FOURIER TRANSFORM (HERE)
  ---------------------             ---------------------------------
  Periodic functions                Aperiodic functions on R
  Discrete spectrum {cn}            Continuous spectrum f()
  Parseval for series               Plancherel's theorem
  Dirichlet convergence             Inversion theorem
  Gibbs phenomenon                  Spectral leakage (-> Section 03)
  RoPE derivation                   Full uncertainty principle
                   v                         v
  +----------------------------------------------------------------+
  |              Section 02 -> THREE BRANCHES                              |
  |                                                                |
  |  Discretize:      Convolve:         Time-localize:            |
  |  Section 03 DFT/FFT  ->  Section 04 Convolution  ->  Section 05 Wavelets            |
  |  $O(N\log N)$    CNNs, Mamba         MRA, multi-scale         |
  |  Whisper, FNO    WaveNet, Hyena      Scattering networks       |
  +----------------------------------------------------------------+
                          v
       Section 21 Statistical Learning Theory
       (spectral learning bounds, kernel methods)

========================================================================
```

### Key Takeaways

The Fourier Transform is characterized by three central theorems:

1. **Plancherel** - the FT is a unitary isometry on $L^2(\mathbb{R})$: it preserves energy and inner products. The Fourier Transform changes how you describe a signal, not how much information it contains.

2. **Uncertainty** - $\Delta t\cdot\Delta\xi \geq 1/(4\pi)$: time and frequency concentration are fundamentally coupled. This is a theorem about analysis, not about physics, and it constrains every signal processing and ML system that operates in both domains simultaneously.

3. **Inversion** - the FT is invertible: given $\hat{f}$, you can recover $f$ exactly. No information is lost in the Fourier domain representation - it is a complete, lossless change of basis.

Together, these theorems say: the Fourier Transform is an **invertible, energy-preserving, mathematically constrained** change of representation. It reveals structure (frequency content) that is invisible in the time domain, converts convolution to multiplication, and converts differentiation to algebra - making it the single most powerful analytical tool for understanding and processing signals, functions, and the weight matrices of neural networks.

---

[<- Back to Fourier Analysis](../README.md) | [Previous: Fourier Series <-](../01-Fourier-Series/notes.md) | [Next: DFT and FFT ->](../03-Discrete-Fourier-Transform-and-FFT/notes.md)

---

## 13. Numerical Fourier Analysis: What the Grid Does to the Theorems

### 13.1 The discrete grid is a modelling choice

Every numerical Fourier computation replaces $\hat f(\xi)=\int f(t)e^{-2\pi i\xi t}dt$ with a sum over a finite window $[-T/2,T/2]$ sampled at step $\Delta t$. Three artefacts follow immediately, and each maps to a theorem in Sections 4-7:

- **Truncation:** the window discards $|t|>T/2$. For slowly decaying $f$ (the sinc of Section 3) this convolves the true spectrum with the window's transform, producing ringing.
- **Sampling:** frequencies above the Nyquist frequency $1/(2\Delta t)$ alias into lower frequencies (Section 03 develops this for the DFT).
- **Periodization:** the FFT treats the windowed signal as periodic, so a jump between the window's ends leaks across all frequencies.

### 13.2 Checking inversion honestly

The inversion theorem (Section 4) holds at points of continuity and returns the midpoint $\tfrac12[f(t^+)+f(t^-)]$ at jumps. A numerical check must therefore:

1. Measure recovery error at interior points, away from jumps.
2. At a jump, compare against the straddle average of the recovered values on either side — not against the grid sample that happens to sit on the discontinuity.
3. Expect Gibbs ringing near jumps: overshoot that narrows but does not shrink as the grid refines (Section 01).

A check that demands pointwise equality at a jump is testing a statement the theorem never made (Common Mistakes, row 10).

### 13.3 Duality checks and jump bands

Duality pairs a smooth statement with a discontinuous one: $\mathcal{F}[\mathrm{sinc}]=\mathrm{rect}$ (theory notebook, Section 3.4 of the notebook). Numerically, error concentrates in a band around $|\xi|=0.5$ where rect jumps. The correct check excludes a band around the jump and verifies the plateau ($|\xi|<0.4$) and the exterior ($0.6<|\xi|<2$) separately; with a $\pm30$ window the plateau error is about $0.02$, while the jump band itself carries Gibbs-scale error that no refinement of $\Delta t$ at fixed window removes — only a wider window reduces it.

### 13.4 Order-four and unitary checks

The continuous transform satisfies $\mathcal{F}^4=I$ and $\mathcal{F}^2f(t)=f(-t)$. On a grid, repeated FFT applications accumulate window and scaling conventions, so the clean numerical witness is the unitary DFT: with the $1/\sqrt N$ normalization, applying the DFT four times returns the sample vector to machine precision (error $\sim10^{-16}$ in the theory notebook). State which object — continuous FT on a truncated grid, or unitary DFT — a numerical claim is about.

### 13.5 Solved problems

1. **Window choice.** A signal decays like $e^{-|t|}$. On $[-10,10]$ the discarded mass is about $e^{-10}\approx4.5\times10^{-5}$ of the peak scale: truncation error is negligible next to sampling error. For $\mathrm{sinc}(t)$, decaying like $1/t$, the same window discards far more, which is why Section 13.3's check needs the jump band excluded and a wide window.
2. **Straddle average.** Recovered values either side of a rect jump are $1.00$ and $0.00$ (to grid accuracy). The inversion prediction at the jump is their average, $0.5$ — the Dirichlet midpoint of Common Mistakes row 10.
3. **Convention audit.** A spectrum computed in the $\omega$-convention is larger by where the $2\pi$ sits: converting $\hat f_\omega(\omega)$ to the $\xi$-convention uses $\xi=\omega/(2\pi)$ with no extra amplitude factor in the unitary-$\xi$ convention of this curriculum (Section 2.2). Writing the convention on first use is the entire fix.
4. **Which artefact?** A spectrum shows ripples whose spacing narrows as the window widens but whose height persists near a jump: that is truncation/Gibbs behaviour, not aliasing — aliasing would move with $\Delta t$, not with window width.

### 13.6 Numerical checklist

- [ ] Window width justified by the signal's decay rate.
- [ ] Sampling step justified by the highest frequency of interest (Nyquist).
- [ ] Inversion checked at continuity points; jumps checked by straddle average.
- [ ] Duality/spectral checks exclude jump bands, with the band stated.
- [ ] Convention ($\xi$ vs $\omega$, sinc normalization) written on first use.
- [ ] Machine-precision claims made only for exact discrete identities (unitary DFT order four).

## Appendix A: Extended Properties and Derivations

### A.1 The Duality Theorem in Full

One of the most elegant properties of the Fourier Transform is **self-duality**: applying the FT twice returns the time-reversed original.

**Theorem A.1 (Duality).** For $f \in L^1(\mathbb{R})$:
$$\mathcal{F}[\hat{f}(t)](\xi) = f(-\xi)$$

Equivalently: if $\mathcal{F}[f](\xi) = g(\xi)$, then $\mathcal{F}[g](\xi) = f(-\xi)$.

*Proof:* $\mathcal{F}[\hat{f}(t)](\xi) = \int\hat{f}(t)e^{-2\pi i\xi t}\,dt = \int\left(\int f(s)e^{-2\pi ist}\,ds\right)e^{-2\pi i\xi t}\,dt$

By Fubini (justified since $f, \hat{f} \in L^1$):
$= \int f(s)\left(\int e^{-2\pi i(s+\xi)t}\,dt\right)ds = \int f(s)\delta(s+\xi)\,ds = f(-\xi)$ $\square$

**Corollary:** Applying the FT four times: $\mathcal{F}^4[f] = f$. The Fourier Transform has order 4 as an operator.

**Using duality to compute new transforms:**

If you know $\mathcal{F}[f](\xi) = g(\xi)$, then $\mathcal{F}[g](\xi) = f(-\xi)$.

Example: We know $\mathcal{F}[\operatorname{rect}(t)](\xi) = \operatorname{sinc}(\xi)$.
By duality: $\mathcal{F}[\operatorname{sinc}(t)](\xi) = \operatorname{rect}(-\xi) = \operatorname{rect}(\xi)$ (since rect is even).

So $\mathcal{F}[\operatorname{sinc}](\xi) = \operatorname{rect}(\xi)$: a sinc in time has a rectangular (bandlimited) spectrum. This is the **ideal low-pass filter** - a filter that passes frequencies $|\xi| \leq 1/2$ perfectly and rejects all others.

### A.2 Analytic Signals and the Hilbert Transform

**Definition A.1 (Analytic Signal).** For a real signal $f(t)$, its **analytic signal** is:
$$f_+(t) = f(t) + i\,\mathcal{H}[f](t)$$

where $\mathcal{H}[f]$ is the **Hilbert Transform**: $\mathcal{H}[f](t) = \frac{1}{\pi}\,\text{P.V.}\int_{-\infty}^\infty \frac{f(\tau)}{t-\tau}\,d\tau$.

**Fourier domain characterization:**
$$\hat{f}_+(\xi) = \begin{cases}2\hat{f}(\xi) & \xi > 0 \\ \hat{f}(0) & \xi = 0 \\ 0 & \xi < 0\end{cases}$$

The analytic signal has only non-negative frequencies: it is a **one-sided spectrum** signal. This is achieved by zeroing out the negative frequencies and doubling the positive ones - a frequency-domain operation.

**Why this matters:** The analytic signal enables the definition of **instantaneous frequency** $\omega(t) = \frac{d}{dt}\arg f_+(t)$ - the frequency of the signal at a given instant. This is essential for frequency-modulated signals (FM radio, chirps) and is used in **empirical mode decomposition** for analyzing non-stationary signals like EEG and speech.

**Hilbert Transform in spectral terms:** $\mathcal{F}[\mathcal{H}[f]](\xi) = -i\,\operatorname{sgn}(\xi)\,\hat{f}(\xi)$. The Hilbert Transform applies a $-\pi/2$ phase shift to positive frequencies and $+\pi/2$ to negative frequencies - it is a "90-degree phase rotator."

### A.3 The Fourier Transform on $\mathbb{R}^d$

The Fourier Transform extends naturally to $d$ dimensions. For $f: \mathbb{R}^d \to \mathbb{C}$ with $f \in L^1(\mathbb{R}^d)$:

$$\hat{f}(\boldsymbol{\xi}) = \int_{\mathbb{R}^d} f(\mathbf{t})\,e^{-2\pi i \boldsymbol{\xi}^\top\mathbf{t}}\,d\mathbf{t}$$

where $\boldsymbol{\xi} \in \mathbb{R}^d$ is the frequency vector. All properties generalize:
- Shift: $\mathcal{F}[f(\mathbf{t} - \mathbf{a})](\boldsymbol{\xi}) = e^{-2\pi i\boldsymbol{\xi}^\top\mathbf{a}}\hat{f}(\boldsymbol{\xi})$
- Scaling by matrix $A$: $\mathcal{F}[f(A\mathbf{t})](\boldsymbol{\xi}) = \frac{1}{|\det A|}\hat{f}(A^{-\top}\boldsymbol{\xi})$
- Differentiation: $\mathcal{F}[\partial_{t_j}f](\boldsymbol{\xi}) = 2\pi i\xi_j\,\hat{f}(\boldsymbol{\xi})$
- Plancherel: $\lVert\hat{f}\rVert_{L^2(\mathbb{R}^d)} = \lVert f\rVert_{L^2(\mathbb{R}^d)}$

**For AI:** The 2D Fourier Transform is used in convolutional networks for images. A 2D convolution with filter $h$ is, in Fourier space, pointwise multiplication: $\hat{y}(\boldsymbol{\xi}) = \hat{x}(\boldsymbol{\xi})\cdot\hat{h}(\boldsymbol{\xi})$. For FNet (Section 8.1), the 2D FT is applied over both the sequence dimension and the embedding dimension.

**Radial functions:** For radially symmetric $f(\mathbf{t}) = f_0(\lVert\mathbf{t}\rVert)$, the Fourier Transform is also radial: $\hat{f}(\boldsymbol{\xi}) = \hat{f}_0(\lVert\boldsymbol{\xi}\rVert)$. The RBF kernel $k(\mathbf{x}-\mathbf{y}) = e^{-\lVert\mathbf{x}-\mathbf{y}\rVert^2}$ is radially symmetric, so its spectral density $p(\boldsymbol{\omega}) = c_d e^{-\lVert\boldsymbol{\omega}\rVert^2/4}$ is also radially symmetric - which is why sampling $\boldsymbol{\omega} \sim \mathcal{N}(\mathbf{0}, \frac{1}{2}I)$ works for RFF with the RBF kernel.

### A.4 The Fourier Transform and PDEs

The differentiation property $\mathcal{F}[f^{(n)}](\xi) = (2\pi i\xi)^n\hat{f}(\xi)$ converts partial differential equations into algebraic equations in frequency space. This is the primary tool for solving linear constant-coefficient PDEs.

**Example 1: Heat Equation.** Solve $\partial_t u = \kappa\partial_{xx}u$, $u(x,0) = u_0(x)$.

Taking the Fourier Transform in $x$:
$$\partial_t\hat{u}(\xi,t) = \kappa(2\pi i\xi)^2\hat{u}(\xi,t) = -4\pi^2\kappa\xi^2\hat{u}(\xi,t)$$

This is a first-order ODE in $t$ for each fixed $\xi$: $\hat{u}(\xi,t) = \hat{u}_0(\xi)\,e^{-4\pi^2\kappa\xi^2 t}$.

Inverting: $u(x,t) = \int\hat{u}_0(\xi)\,e^{-4\pi^2\kappa\xi^2 t}\,e^{2\pi i\xi x}\,d\xi = (u_0 * G_t)(x)$

where $G_t(x) = \frac{1}{\sqrt{4\pi\kappa t}}e^{-x^2/(4\kappa t)}$ is the **Gaussian heat kernel** (a Gaussian with time-dependent width $\sigma = \sqrt{2\kappa t}$).

**Interpretation:** Heat diffusion = convolution with a Gaussian. High frequencies $\xi$ decay as $e^{-4\pi^2\kappa\xi^2 t}$ - much faster than low frequencies. Sharp features (high frequency content) are smoothed out rapidly; broad features (low frequency) persist.

**For AI - Diffusion Models:** The DDPM forward process $q(x_t|x_0) = \mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$ is exactly discrete heat diffusion: the signal is progressively smoothed by adding Gaussian noise (which has a flat spectrum = all frequencies). The network learns to reverse this process. The Fourier perspective explains why the early denoising steps (low noise) must recover high-frequency details while later steps (high noise) recover the coarse structure - the exact inverse of heat diffusion.

**Example 2: Wave Equation.** Solve $\partial_{tt}u = c^2\partial_{xx}u$, $u(x,0) = u_0(x)$, $\partial_t u(x,0) = v_0(x)$.

FT in $x$: $\partial_{tt}\hat{u} = (2\pi i\xi)^2 c^2\hat{u} = -4\pi^2c^2\xi^2\hat{u}$

Solution: $\hat{u}(\xi,t) = \hat{u}_0(\xi)\cos(2\pi c|\xi|t) + \frac{\hat{v}_0(\xi)}{2\pi ic|\xi|}\sin(2\pi c|\xi|t)$

By d'Alembert's formula (inverting): $u(x,t) = \frac{1}{2}[u_0(x+ct) + u_0(x-ct)] + \frac{1}{2c}\int_{x-ct}^{x+ct}v_0(s)\,ds$

The wave equation propagates each frequency at the same speed $c$ - no dispersion. In contrast, the heat equation attenuates high frequencies faster than low frequencies - strong dispersion (dissipation).

**Example 3: Schrodinger Equation.** The free-particle Schrodinger equation $i\hbar\partial_t\psi = -\frac{\hbar^2}{2m}\partial_{xx}\psi$ is formally a heat equation with imaginary time. Its solution $\hat{\psi}(\xi,t) = \hat{\psi}_0(\xi)e^{-i\hbar(2\pi\xi)^2t/(2m)}$ shows that each frequency component oscillates (rather than decays) with frequency $\hbar\xi^2/(2m)$ - quantum dispersion. The uncertainty principle $\Delta x\cdot\Delta p \geq \hbar/2$ (where $p = h\xi$ by de Broglie) is the physics version of the mathematical uncertainty principle in Section 6.

### A.5 Windowed Fourier Transform (STFT)

The continuous Fourier Transform gives global frequency information but no time localization: $|\hat{f}(\xi)|$ tells you whether frequency $\xi$ is present somewhere in $f$, but not *when*.

**Definition A.2 (Short-Time Fourier Transform / Spectrogram).** For a window function $g \in L^2(\mathbb{R})$ and signal $f \in L^2(\mathbb{R})$:
$$\text{STFT}_f(t, \xi) = \int_{-\infty}^{\infty} f(\tau)\,\overline{g(\tau - t)}\,e^{-2\pi i\xi\tau}\,d\tau$$

This localizes the analysis around time $t$ using the window $g$. The **spectrogram** is $|\text{STFT}_f(t,\xi)|^2$ - a time-frequency power map.

**The uncertainty principle applies:** The time resolution $\Delta t$ and frequency resolution $\Delta\xi$ of the STFT are determined by the window $g$:
- Narrow window $g$ -> good time resolution, poor frequency resolution
- Wide window $g$ -> poor time resolution, good frequency resolution

The Gaussian window $g(t) = e^{-\pi t^2/\sigma^2}$ minimizes the product $\Delta t\cdot\Delta\xi$ for a given $\sigma$, making it the **optimal STFT window** (the resulting time-frequency representation is called the **Gabor transform**).

**For AI - Audio Models:** Whisper (Radford et al., 2022) processes audio as mel-spectrograms: it applies a 25ms STFT window (adequate time resolution for phoneme detection) with 10ms hop, then maps the frequency axis to the mel scale (logarithmic, matching human perception), and feeds the result as a 2D "image" to a Vision Transformer. The STFT parameters are fixed - the uncertainty principle determines the time-frequency resolution tradeoff.

```
STFT TIME-FREQUENCY TRADEOFF (UNCERTAINTY PRINCIPLE IN PRACTICE)
========================================================================

  Short window (e.g., 2ms):            Long window (e.g., 50ms):
  -------------------------            --------------------------
  Fine time resolution (~2ms)          Coarse time resolution (~50ms)
  Poor freq resolution (~500 Hz)       Fine freq resolution (~20 Hz)
  Good for: transients, clicks         Good for: sustained tones, pitch

  Whisper (speech): 25ms window -> resolves individual phonemes (50-100ms)
  while achieving ~40Hz frequency resolution (enough for pitch/formants)

  The Gaussian window achieves the minimum t product - it is the
  optimal window under the uncertainty principle.

========================================================================
```

### A.6 Fourier Transform and Operator Theory

From the perspective of functional analysis, the Fourier Transform is a **unitary operator** on the Hilbert space $L^2(\mathbb{R})$. This abstract viewpoint connects Fourier analysis to the broader theory of self-adjoint operators.

**The Fourier operator $\mathcal{F}$:** $\mathcal{F}: L^2(\mathbb{R}) \to L^2(\mathbb{R})$ is unitary: $\mathcal{F}^*\mathcal{F} = \mathcal{F}\mathcal{F}^* = I$.

Since $\mathcal{F}^4 = I$, the eigenvalues satisfy $\lambda^4 = 1$: $\lambda \in \{1, -1, i, -i\}$.

The **spectral decomposition** of $\mathcal{F}$ in terms of its eigenspaces: $L^2(\mathbb{R}) = \mathcal{E}_1 \oplus \mathcal{E}_{-1} \oplus \mathcal{E}_i \oplus \mathcal{E}_{-i}$, where each $\mathcal{E}_\lambda$ is spanned by the Hermite functions with the appropriate eigenvalue.

**Connection to quantum mechanics:** The position operator $\hat{X}f(t) = tf(t)$ and momentum operator $\hat{P}f(t) = \frac{1}{2\pi i}\frac{df}{dt}$ (in natural units) are both self-adjoint operators on $L^2(\mathbb{R})$, related by $\hat{P} = \mathcal{F}\hat{X}\mathcal{F}^{-1}$. The commutator $[\hat{X},\hat{P}] = \frac{i}{2\pi}I$ implies the uncertainty principle $\Delta X\cdot\Delta P \geq \frac{1}{4\pi}$. The mathematical identity and the physics are the same theorem.

**For AI:** The language of operator theory is increasingly used to analyze neural networks. The weight matrix $W$ of a linear layer is a linear operator; its spectral norm $\lVert W\rVert_2$ is its "size" as an operator; its singular value decomposition expresses it in terms of rank-1 operators. The FT connects linear operator theory to signal processing, unifying the analysis of both classical filters and neural network layers.


---

## Appendix B: Worked Examples and Techniques

### B.1 The Fourier Transform Table - Complete Reference

The following table lists the most important Fourier Transform pairs, using the $\xi$-convention $\hat{f}(\xi) = \int f(t)e^{-2\pi i\xi t}\,dt$. All functions are assumed in $L^1 \cap L^2$ unless otherwise noted.

| # | $f(t)$ | $\hat{f}(\xi)$ | Notes |
|---|--------|----------------|-------|
| 1 | $e^{-\pi t^2}$ | $e^{-\pi\xi^2}$ | Self-dual Gaussian |
| 2 | $e^{-\pi t^2/\sigma^2}$ | $\sigma e^{-\pi\sigma^2\xi^2}$ | Scaled Gaussian |
| 3 | $e^{-a\lvert t\rvert}$, $a>0$ | $\frac{2a}{a^2+(2\pi\xi)^2}$ | Double-sided exponential |
| 4 | $e^{-at}\mathbb{1}_{t\geq0}$, $a>0$ | $\frac{1}{a+2\pi i\xi}$ | One-sided exponential |
| 5 | $\operatorname{rect}(t/T)$ | $T\operatorname{sinc}(T\xi)$ | Rectangular pulse of width $T$ |
| 6 | $\operatorname{sinc}(Wt)$ | $\frac{1}{W}\operatorname{rect}(\xi/W)$ | Ideal low-pass filter (by duality) |
| 7 | $\Lambda(t)=\max(1-\lvert t\rvert,0)$ | $\operatorname{sinc}^2(\xi)$ | Triangle function |
| 8 | $\frac{1}{1+(2\pi t)^2}$ | $\frac{1}{2}e^{-\lvert\xi\rvert}$ | Lorentzian / Cauchy |
| 9 | $\delta(t)$ | $1$ | Dirac impulse -> white spectrum |
| 10 | $1$ | $\delta(\xi)$ | Constant -> DC spike (distributional) |
| 11 | $\delta(t-a)$ | $e^{-2\pi i\xi a}$ | Shifted impulse |
| 12 | $e^{2\pi i\xi_0 t}$ | $\delta(\xi-\xi_0)$ | Pure tone (distributional) |
| 13 | $\cos(2\pi\xi_0 t)$ | $\frac{1}{2}[\delta(\xi-\xi_0)+\delta(\xi+\xi_0)]$ | Cosine |
| 14 | $\sin(2\pi\xi_0 t)$ | $\frac{1}{2i}[\delta(\xi-\xi_0)-\delta(\xi+\xi_0)]$ | Sine |
| 15 | $\sum_{n=-\infty}^\infty\delta(t-nT)$ | $\frac{1}{T}\sum_{n=-\infty}^\infty\delta(\xi-n/T)$ | Dirac comb |
| 16 | $\operatorname{sgn}(t)$ | $\frac{1}{\pi i\xi}$ (distributional) | Sign function |
| 17 | $\mathbb{1}_{t\geq0}$ (unit step) | $\frac{1}{2}\delta(\xi)+\frac{1}{2\pi i\xi}$ | Heaviside step (distributional) |
| 18 | $t^n e^{-at}$, $a>0$, $t\geq0$ | $\frac{n!}{(a+2\pi i\xi)^{n+1}}$ | Polynomial  exponential |

### B.2 Four Full Worked Examples

**Worked Example 1: Triangular Pulse**

The triangular function $\Lambda(t) = \max(1 - |t|, 0)$ is zero outside $[-1, 1]$. We compute $\hat{\Lambda}$ directly.

$$\hat{\Lambda}(\xi) = \int_{-1}^{0}(1+t)e^{-2\pi i\xi t}\,dt + \int_0^1(1-t)e^{-2\pi i\xi t}\,dt$$

For the first integral, let $u = -t$:
$$\int_{-1}^{0}(1+t)e^{-2\pi i\xi t}\,dt = \int_0^1(1-u)e^{2\pi i\xi u}\,du$$

Adding both integrals: $\hat{\Lambda}(\xi) = \int_0^1(1-t)(e^{2\pi i\xi t}+e^{-2\pi i\xi t})\,dt = 2\int_0^1(1-t)\cos(2\pi\xi t)\,dt$.

Integrating by parts:
$$= 2\left[\frac{(1-t)\sin(2\pi\xi t)}{2\pi\xi}\right]_0^1 + 2\int_0^1\frac{\sin(2\pi\xi t)}{2\pi\xi}\,dt = 0 + \frac{1}{\pi\xi}\left[-\frac{\cos(2\pi\xi t)}{2\pi\xi}\right]_0^1 = \frac{1-\cos(2\pi\xi)}{2\pi^2\xi^2}$$

Using $1 - \cos\theta = 2\sin^2(\theta/2)$:
$$\hat{\Lambda}(\xi) = \frac{2\sin^2(\pi\xi)}{2\pi^2\xi^2} = \left(\frac{\sin(\pi\xi)}{\pi\xi}\right)^2 = \operatorname{sinc}^2(\xi)$$

The triangular pulse is the FT of $\operatorname{sinc}^2$. Alternatively: $\Lambda = \operatorname{rect} * \operatorname{rect}$ (convolution of two rect functions), so $\hat{\Lambda} = \hat{\operatorname{rect}}\cdot\hat{\operatorname{rect}} = \operatorname{sinc}^2$ by the convolution theorem. The $\operatorname{sinc}^2$ spectrum decays faster than $\operatorname{sinc}$ (as $1/\xi^2$ vs $1/\xi$) because $\Lambda$ is continuous (no jumps), while $\operatorname{rect}$ has jump discontinuities.

**Worked Example 2: Gaussian via Differential Equation**

An elegant alternative proof that $\mathcal{F}[e^{-\pi t^2}] = e^{-\pi\xi^2}$ uses the fact that $f(t) = e^{-\pi t^2}$ satisfies $f'(t) = -2\pi t\,f(t)$.

Taking the FT of both sides and using the differentiation rules:
- LHS: $\mathcal{F}[f'](\xi) = 2\pi i\xi\,\hat{f}(\xi)$
- RHS: $\mathcal{F}[-2\pi t\,f(t)](\xi) = -2\pi\cdot\frac{1}{-2\pi i}\frac{d}{d\xi}\hat{f}(\xi) = -i\hat{f}'(\xi)$

Wait - use: $\mathcal{F}[tf(t)](\xi) = \frac{1}{-2\pi i}\frac{d}{d\xi}\hat{f}(\xi) = \frac{i}{2\pi}\hat{f}'(\xi)$

So $2\pi i\xi\,\hat{f}(\xi) = -2\pi\cdot\frac{i}{2\pi}\hat{f}'(\xi) = -i\hat{f}'(\xi)$.

This gives the ODE: $\hat{f}'(\xi) = -2\pi\xi\,\hat{f}(\xi)$, with solution $\hat{f}(\xi) = C\,e^{-\pi\xi^2}$.

To find $C$: $\hat{f}(0) = \int e^{-\pi t^2}\,dt = 1$ (Gaussian integral). So $C = 1$ and $\hat{f}(\xi) = e^{-\pi\xi^2}$. $\square$

This proof is superior for teaching: it shows that the Gaussian's self-duality follows from the fact that $e^{-\pi t^2}$ satisfies the simplest possible first-order ODE, and the FT converts this ODE into the same ODE in frequency space.

**Worked Example 3: FT of $e^{-a|t|}$ and a Contour Argument**

$$\hat{f}(\xi) = \int_{-\infty}^\infty e^{-a|t|}e^{-2\pi i\xi t}\,dt = \int_0^\infty e^{-at}e^{-2\pi i\xi t}\,dt + \int_{-\infty}^0 e^{at}e^{-2\pi i\xi t}\,dt$$

For the first integral ($t > 0$):
$$\int_0^\infty e^{-(a+2\pi i\xi)t}\,dt = \frac{1}{a+2\pi i\xi} \quad \text{(valid for } \operatorname{Re}(a+2\pi i\xi) = a > 0\text{)}$$

For the second integral, let $u = -t$ ($u > 0$, $t < 0$):
$$\int_0^\infty e^{-au}e^{2\pi i\xi u}\,du = \frac{1}{a-2\pi i\xi}$$

Therefore:
$$\hat{f}(\xi) = \frac{1}{a+2\pi i\xi} + \frac{1}{a-2\pi i\xi} = \frac{(a-2\pi i\xi)+(a+2\pi i\xi)}{a^2+(2\pi\xi)^2} = \frac{2a}{a^2+4\pi^2\xi^2}$$

This is a Lorentzian (Cauchy distribution) centered at $\xi = 0$ with half-width $a/(2\pi)$. As $a \to 0^+$, the time-domain signal $e^{-a|t|}$ approaches the constant $1$, and its Fourier Transform approaches $2/(4\pi^2\xi^2)\cdot2a \to 2\pi\delta(\xi)/\pi = 2\delta(\xi)$... Actually: $\frac{2a}{a^2+4\pi^2\xi^2} \to \frac{1}{2\pi}\cdot\frac{2\pi a}{\pi[a^2+4\pi^2\xi^2]/(2\pi)} \to$ ... more carefully, $\frac{a}{\pi(a^2+4\pi^2\xi^2)}\cdot 2\pi$ and $\frac{a}{\pi(a^2+\xi^2)} \to \delta(\xi)$ is the standard Poisson kernel. The limit confirms $\mathcal{F}[1] = \delta$.

**Worked Example 4: The FT and Convolution for the Heat Equation Kernel**

The heat kernel at time $t > 0$ is $G_t(x) = \frac{1}{\sqrt{4\pi\kappa t}}e^{-x^2/(4\kappa t)}$. We verify $\mathcal{F}[G_t](\xi) = e^{-4\pi^2\kappa t\xi^2}$ using the scaling theorem.

We know $\mathcal{F}[e^{-\pi x^2}](\xi) = e^{-\pi\xi^2}$. The heat kernel is:
$$G_t(x) = \frac{1}{\sqrt{4\pi\kappa t}}e^{-x^2/(4\kappa t)} = \frac{1}{\sqrt{4\pi\kappa t}}\cdot e^{-\pi(x/\sigma)^2/\pi}\cdot\frac{1}{\sigma}...$$

More directly: write $G_t(x) = \frac{1}{\sqrt{4\pi\kappa t}}e^{-x^2/(4\kappa t)}$. Setting $\sigma = \sqrt{2\kappa t}$:
$$G_t(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-x^2/(2\sigma^2)}$$

The FT of a Gaussian $\frac{1}{\sqrt{2\pi}\sigma}e^{-x^2/(2\sigma^2)}$ is $e^{-2\pi^2\sigma^2\xi^2}$... Using our convention:

$\mathcal{F}[e^{-\pi x^2/\sigma^2}](\xi) = \sigma e^{-\pi\sigma^2\xi^2}$, so $\mathcal{F}\!\left[\frac{1}{\sigma}e^{-\pi x^2/\sigma^2}\right](\xi) = e^{-\pi\sigma^2\xi^2}$.

For $G_t$: $G_t(x) = \frac{1}{\sigma_t}e^{-\pi x^2/\sigma_t^2}$ where $\sigma_t = 2\sqrt{\pi\kappa t}$... Let me just directly compute:

Write $G_t(x) = ce^{-\alpha x^2}$ with $c = (4\pi\kappa t)^{-1/2}$, $\alpha = 1/(4\kappa t)$. Using $\mathcal{F}[e^{-\alpha x^2}](\xi) = \sqrt{\pi/\alpha}\,e^{-\pi^2\xi^2/\alpha}$:
$$\mathcal{F}[G_t](\xi) = c\sqrt{\frac{\pi}{\alpha}}e^{-\pi^2\xi^2/\alpha} = \frac{1}{\sqrt{4\pi\kappa t}}\cdot\sqrt{4\pi\kappa t}\cdot e^{-\pi^2\xi^2\cdot 4\kappa t} = e^{-4\pi^2\kappa t\xi^2}$$

This confirms that in the $\xi$-convention, the heat kernel's FT is $e^{-4\pi^2\kappa t\xi^2}$ - exactly the solution we derived from the heat equation in Section A.4.

### B.3 Decay Rates and Smoothness: The Complete Picture

The relationship between smoothness of $f$ and decay of $\hat{f}$ is one of the deepest qualitative results in Fourier analysis. The complete picture:

**Theorem B.1 (Smoothness  Spectral Decay).** Let $f \in L^1(\mathbb{R})$.

| Condition on $f$ | Decay of $|\hat{f}(\xi)|$ as $|\xi| \to \infty$ |
|-----------------|----------------------------------------------|
| $f$ has a jump discontinuity | $O(1/|\xi|)$ |
| $f \in C^1$ (continuous, one derivative) | $o(1/|\xi|)$ |
| $f \in C^k$, $f^{(k)} \in L^1$ | $O(1/|\xi|^k)$ |
| $f \in C^\infty$ | Faster than any polynomial: $O(1/|\xi|^N)$ for all $N$ |
| $f$ analytic (holomorphic in a strip $|\operatorname{Im}(t)| < \delta$) | Exponential decay: $O(e^{-2\pi\delta|\xi|})$ |

*Proof sketch:* For the $k$-derivative case: $|\hat{f}(\xi)| = |\mathcal{F}[f^{(k)}](\xi)|/|2\pi i\xi|^k \leq \lVert f^{(k)}\rVert_1/(2\pi|\xi|)^k$. For the analytic case, shift the contour of integration to $\operatorname{Im}(t) = \pm\delta$ (for $\xi \gtrless 0$) gaining an exponential factor $e^{\mp 2\pi\xi\delta} = e^{-2\pi|\xi|\delta}$. $\square$

**Applications:**
- **Signal compression:** Smooth signals (like audio with no sharp transients) have rapidly decaying spectra -> can be compressed by truncating high frequencies with small error.
- **Neural network generalization:** Smooth target functions (natural image statistics) have rapidly decaying spectra -> a network with limited bandwidth can approximate them well. This connects to the spectral bias of neural networks (Rahaman et al., 2019): networks preferentially learn low-frequency components first.
- **Discretization error:** The aliasing error when sampling a signal at rate $T$ is bounded by $\sum_{n \neq 0} |\hat{f}(\xi + n/T)| \leq C/T^k$ if $f \in C^k$ - smoother signals alias less.

### B.4 The Fourier Transform in Different Function Spaces

A complete picture of where the FT lives:

```
DOMAIN OF DEFINITION - FOURIER TRANSFORM
========================================================================

  L1(R)    ->  C0(R)        (bounded, continuous, vanishes at infty)
               f_infty <= f_1

  L2(R)    ->  L2(R)        (Plancherel: FT is unitary)
               f_2 = f_2

  L1  L2  ->  L2           (both apply; easiest to work with)

  S(R)     ->  S(R)         (Schwartz space: FT preserves rapidly
              (bijection)    decreasing smooth functions)

  S'(R)    ->  S'(R)        (tempered distributions: largest domain)
              (bijection)    includes , constants, periodic signals

  Key inclusions:  S  L1  L2  L1  S'
                   S  L1  L2  L2  S'

  For practical computation: work in S' (distributions),
  which unifies all cases under one theory.

========================================================================
```

### B.5 Numerical Computation of the Fourier Transform

In practice, the Fourier Transform is computed numerically via the DFT/FFT (Section 20-03). Here we note the key relationships.

**Approximation scheme.** To numerically approximate $\hat{f}(\xi)$ for $f \in L^1(\mathbb{R})$:

1. **Truncate** the signal to $[-L, L]$ for large enough $L$ (error: $\int_{|t|>L}|f(t)|\,dt$)
2. **Sample** at rate $N/(2L)$ to get $N$ equally-spaced samples $f_n = f(nT)$ where $T = 2L/N$
3. **Apply DFT:** $\hat{f}_k = T\sum_{n=0}^{N-1}f_n e^{-2\pi ink/N}$
4. The result approximates $\hat{f}(k/(2L))$ for $k = 0, 1, \ldots, N/2$.

**Sources of error:**
- **Aliasing:** Undersampling causes high-frequency components to fold back - reduced by sampling faster (higher $N$)
- **Leakage:** Truncation at $\pm L$ multiplies by a rect function -> convolves spectrum with sinc -> spectral leakage. Reduced by windowing (multiplying by a smooth window before DFT)
- **Resolution:** The frequency resolution is $1/(2L)$ - the smallest frequency distinguishable in the DFT is $1/(2L)$. For finer resolution, use longer time windows.

All three error types are direct consequences of the Fourier Transform properties developed in this section. Full treatment in Section 20-03.


---

## Appendix C: Connections, Extensions, and Deep Dives

### C.1 The Fourier Transform on Locally Compact Abelian Groups

The Fourier Transform generalizes far beyond $\mathbb{R}$. For any locally compact abelian (LCA) group $G$ with Haar measure $\mu$, there is a Fourier Transform $\mathcal{F}: L^1(G) \to C_0(\hat{G})$ where $\hat{G}$ is the **Pontryagin dual group** (the group of continuous homomorphisms $G \to S^1$).

| Group $G$ | Dual $\hat{G}$ | Fourier Transform | Name |
|-----------|---------------|------------------|------|
| $\mathbb{R}$ | $\mathbb{R}$ | $\int f(t)e^{-2\pi i\xi t}\,dt$ | Fourier integral |
| $\mathbb{Z}$ | $S^1 = \mathbb{R}/\mathbb{Z}$ | $\sum_n f(n)e^{-2\pi i\xi n}$ | Discrete-time FT (DTFT) |
| $S^1$ | $\mathbb{Z}$ | $\int_0^1 f(e^{i\theta})e^{-in\theta}\,d\theta$ | Fourier series |
| $\mathbb{Z}/N\mathbb{Z}$ | $\mathbb{Z}/N\mathbb{Z}$ | $\sum_{k=0}^{N-1}f(k)e^{-2\pi ikm/N}$ | Discrete Fourier Transform (DFT) |
| $\mathbb{R}^d$ | $\mathbb{R}^d$ | $\int f(\mathbf{t})e^{-2\pi i\boldsymbol{\xi}^\top\mathbf{t}}\,d\mathbf{t}$ | Multidimensional FT |

Plancherel's theorem, the convolution theorem, and the inversion theorem all hold in this general setting. The unification of Fourier series and Fourier Transform as special cases of the same construction on different groups is one of the most elegant results in abstract harmonic analysis.

**For AI:** The group structure is what makes RoPE work. The circle group $S^1$ (the group of complex numbers of modulus 1 under multiplication) is a compact abelian group. Multiplying by $e^{im\theta}$ is the group action of rotation by $m\theta$. The relative-position property of RoPE $\langle e^{im\theta}q, e^{in\theta}k\rangle = e^{i(m-n)\theta}\langle q,k\rangle$ is the statement that the group action is transitive: the inner product only sees the group element $e^{i(m-n)\theta}$ corresponding to relative position.

### C.2 Non-Commutative Fourier Analysis

For non-abelian groups (e.g., $SO(3)$, permutation groups $S_N$), the Pontryagin duality breaks down but a Fourier Transform still exists, with irreducible unitary representations playing the role of frequencies.

**Graph Fourier Transform:** For a graph $G = (V, E)$ with Laplacian $L = D - A$ (where $D$ is the degree matrix), the eigenvectors $L\mathbf{u}_k = \lambda_k\mathbf{u}_k$ play the role of complex exponentials. The **graph Fourier Transform** is $\hat{\mathbf{f}} = U^\top\mathbf{f}$ where $U = [\mathbf{u}_1,\ldots,\mathbf{u}_N]$ is the matrix of eigenvectors. The eigenvalues $\lambda_k$ represent "graph frequencies" - how rapidly the eigenvector oscillates across the graph.

This is the foundation of **spectral graph neural networks** (Bruna et al., 2014; Defferrard et al., 2016 ChebNet; Kipf & Welling, 2017 GCN). The spectral GNN applies a learned filter $h_\theta(\Lambda)$ in graph Fourier space: $\hat{h}_\theta * \mathbf{f} = U\,h_\theta(\Lambda)\,U^\top\mathbf{f}$.

> **Full treatment in [Section 11-04 Spectral Graph Theory](../../11-Graph-Theory/04-Spectral-Methods/notes.md).**

### C.3 The Fourier Transform and Learning Theory

The Fourier Transform connects deeply to several foundational results in learning theory:

**Spectral Bias (Rahaman et al., 2019):** Neural networks trained by gradient descent learn the Fourier components of the target function in order of increasing frequency. This is because the NTK (Neural Tangent Kernel) has eigenvalues that decay rapidly with frequency - the network converges faster on low-frequency components. Consequence: networks generalize well on smooth targets and may overfit on high-frequency features.

**Frequency Principle:** Let $f^*$ be the target function and $f_t$ the network at training step $t$. Then $|\hat{f}_t(\xi) - \hat{f}^*(\xi)|$ decreases at rate $\propto \lambda_\xi$ (the NTK eigenvalue at frequency $\xi$), and $\lambda_\xi$ decreases with $|\xi|$. High-frequency components therefore converge last.

**Implications:**
- Data augmentation that adds high-frequency perturbations (e.g., random cropping, noise injection) improves generalization precisely because it forces the network to represent these frequencies.
- Adversarial examples often correspond to high-frequency perturbations of inputs that fool the network but are invisible to humans - the network hasn't fully learned the high-frequency structure.
- Neural networks with spectral inductive bias (e.g., SIREN - sinusoidal activation functions) can represent high-frequency targets more efficiently.

**Rademacher Complexity and Fourier Norms:** The generalization error of a hypothesis class can be bounded by its **Fourier $\ell^1$ norm**: $\sum_\xi |\hat{f}(\xi)|$. Functions with small Fourier $\ell^1$ norm have low complexity and generalize well. This connects to the theory of Barron functions (Barron, 1993): functions $f$ with $\int|\hat{f}(\xi)||\xi|\,d\xi < C$ can be approximated to accuracy $\epsilon$ by a two-layer neural network with $O(C^2/\epsilon^2)$ neurons.

### C.4 Phase Retrieval - When Magnitude Alone Is Insufficient

The FT $\hat{f}(\xi) = |\hat{f}(\xi)|e^{i\phi(\xi)}$ consists of a magnitude $|\hat{f}(\xi)|$ and a phase $\phi(\xi)$. **Phase retrieval** asks: can you recover $f$ from $|\hat{f}|$ alone (without the phase)?

In general, the answer is no - many different signals can have the same magnitude spectrum. For example, $f(t)$ and $f(-t)$ have the same magnitude spectrum $|\hat{f}(\xi)|$ but may be entirely different. More strikingly, phase-randomized images (keeping $|\hat{f}|$ but using random $\phi(\xi)$) look like noise - the phase carries most of the perceptual information.

**Phase is more important than magnitude for perception:** An experiment: take image $A$ and image $B$, swap their phase spectra (keep $A$'s magnitude, use $B$'s phase), and reconstruct. The result looks like image $B$ - demonstrating that phase dominates perception.

**For AI:** This has implications for:
- **Adversarial robustness:** Many adversarial attacks perturb the high-frequency phase of input images - imperceptible to humans but highly disruptive to CNNs.
- **Generative models:** Diffusion models and GANs must learn both the magnitude and phase of the data distribution's spectrum to generate perceptually realistic images.
- **FNet limitation:** FNet takes the real part of the FT output, effectively discarding half the phase information. This is one reason FNet performs worse than attention on tasks requiring precise token interactions.

### C.5 Fast Algorithms Beyond FFT

The FFT computes the $N$-point DFT in $O(N\log N)$ operations, a vast improvement over the naive $O(N^2)$. Several modern algorithms push further:

**Sparse FFT:** If the signal has only $k \ll N$ significant frequencies, the **sparse FFT** (Hassanieh et al., 2012; adopted in MIT SFFT) computes the DFT in $O(k\log N)$ - sublinear in $N$. Applications: energy-efficient spectrum sensing, MRI acceleration.

**Non-Uniform FFT (NUFFT):** The standard FFT requires uniformly-spaced samples. The NUFFT handles non-uniform grids with $O(N\log N)$ complexity, using oversampling and interpolation. Used in MRI reconstruction (non-Cartesian k-space trajectories) and radio astronomy.

**Butterfly factorization (Monarch matrices):** The FFT matrix can be written as a product of $O(\log N)$ sparse matrices (butterfly factors), each requiring $O(N)$ operations. **Monarch matrices** (Dao et al., 2022) generalize this structure for learned efficient linear layers - they can represent any matrix with $O(N\sqrt{N})$ parameters that can be applied in $O(N\sqrt{N})$ time, interpolating between dense matrices ($O(N^2)$) and diagonal ($O(N)$).

> **The FFT algorithm and its applications** - Cooley-Tukey, Bluestein, Rader - are covered in full in [Section 20-03 DFT and FFT](../03-Discrete-Fourier-Transform-and-FFT/notes.md).

### C.6 Summary: The Central Theorems of the Fourier Transform

This section has developed the following central results, in order of importance:

**1. Definition and well-posedness** (Section 2):
$$\hat{f}(\xi) = \int_{-\infty}^\infty f(t)e^{-2\pi i\xi t}\,dt, \qquad f \in L^1(\mathbb{R})$$
Well-defined, bounded, continuous. Riemann-Lebesgue: $\hat{f}(\xi) \to 0$ as $|\xi| \to \infty$.

**2. Properties** (Section 3): Linearity, shift, modulation, scaling, time reversal, differentiation, convolution-multiplication duality. The Master Properties Table encodes all relationships between time and frequency operations.

**3. Inversion** (Section 4): Under sufficient conditions, $f(t) = \int\hat{f}(\xi)e^{2\pi i\xi t}\,d\xi$. The Gauss-Weierstrass regularization gives a constructive proof.

**4. Plancherel** (Section 5): $\lVert\hat{f}\rVert_2 = \lVert f\rVert_2$. The FT is a unitary isometry on $L^2(\mathbb{R})$ - energy is preserved.

**5. Uncertainty Principle** (Section 6): $\Delta t\cdot\Delta\xi \geq 1/(4\pi)$, with equality iff $f$ is a Gaussian. Time-frequency concentration is fundamentally limited.

**6. Distribution extension** (Section 7): The FT extends to tempered distributions, giving $\mathcal{F}[\delta] = 1$, $\mathcal{F}[1] = \delta$, $\mathcal{F}[e^{2\pi i\xi_0 t}] = \delta(\xi-\xi_0)$, and the Dirac comb formula. Poisson summation unifies Fourier series and Fourier Transform.

These six results form a complete, self-consistent theory that has shaped mathematics, physics, engineering, and AI for two centuries - and shows no sign of becoming less central.


---

## Appendix D: Machine Learning Deep Dives

### D.1 FNet Architecture - Full Mathematical Analysis

We give a complete mathematical analysis of the FNet architecture (Lee-Thorp et al., 2022) using the formalism developed in this section.

**Standard Transformer token mixing (attention):**

Given token matrix $X \in \mathbb{R}^{N \times d}$ (sequence length $N$, embedding dimension $d$):
$$\text{Attn}(X) = \operatorname{softmax}\!\left(\frac{XW_Q W_K^\top X^\top}{\sqrt{d_k}}\right)XW_V$$

This is $O(N^2 d_k)$ in memory and time (the $N \times N$ attention matrix dominates for large $N$). The attention matrix $A_{ij} = \operatorname{softmax}(\text{score})_{ij}$ is **data-dependent**: each row is a different weighted sum of the value vectors, where the weights depend on the specific input.

**FNet token mixing:**
$$\text{FNetMix}(X) = \operatorname{Re}(\mathcal{F}_\text{seq}[\mathcal{F}_\text{model}[X]])$$

where $\mathcal{F}_\text{seq}$ applies the DFT along the sequence dimension (rows) and $\mathcal{F}_\text{model}$ along the embedding dimension (columns).

**The DFT as a matrix multiplication:**

The 1D DFT can be written as $\hat{\mathbf{x}} = F\mathbf{x}$ where $F \in \mathbb{C}^{N \times N}$ with $F_{kn} = e^{-2\pi ikn/N}/\sqrt{N}$ (unitary DFT matrix). The 2D FNet mixing is:
$$\text{FNetMix}(X) = \operatorname{Re}(F_N X F_d^\top)$$
where $F_N \in \mathbb{C}^{N\times N}$ and $F_d \in \mathbb{C}^{d\times d}$ are DFT matrices.

**Key properties:**

1. **Unitarity:** $F_N F_N^* = I_N$ and $F_d F_d^* = I_d$. The FNet mixing is an isometry (before taking real part): $\lVert F_N X F_d^\top\rVert_F = \lVert X\rVert_F$ by Plancherel.

2. **Global mixing:** Each output position $Y_{kj} = \sum_n\sum_m X_{nm}(F_N)_{kn}(F_d)_{jm}$ depends on all $N \times d$ input values. FNet is a global mixer - exactly like attention.

3. **No parameters:** Unlike attention, FNet has zero parameters in the mixing layer ($F_N$ and $F_d$ are fixed). All learning happens in the feed-forward network after mixing.

4. **Complexity:** $O(Nd\log(Nd))$ via 2D FFT, vs $O(N^2 d)$ for attention. For $N = 512$, $d = 768$: FNet is $\sim 500\times$ fewer operations in the mixing layer.

**Why taking real part?** The DFT output is complex even for real inputs. Taking $\operatorname{Re}(\cdot)$ projects back to real numbers (required since the subsequent feed-forward layers expect real inputs). This discards the imaginary part - half the phase information. This is one reason FNet is weaker than attention: it cannot separately process magnitude and phase information.

**Theoretical analysis:** The FNet paper shows that attention with random (untrained) weights performs similarly to FNet. This suggests the attention mechanism's power comes primarily from its (trained) data-dependent mixing, not merely from being a global mixer. FNet is the "global but data-independent" baseline.

**When to use FNet vs Attention:**

| Task type | FNet | Attention | Reason |
|-----------|------|-----------|--------|
| Sentence classification | ~92-97% BERT | 100% | Global mixing is sufficient |
| Named entity recognition | ~90% | 100% | Moderate position sensitivity |
| Extractive QA (SQuAD) | ~80% | 100% | Requires precise token matching |
| Speed-critical inference | +7 | baseline | FNet wins when quality tradeoff acceptable |

### D.2 Random Fourier Features - Implementation Details

**Setup for RBF kernel:** $k(\mathbf{x},\mathbf{y}) = e^{-\gamma\lVert\mathbf{x}-\mathbf{y}\rVert^2}$.

The spectral density is $p(\boldsymbol{\omega}) = \left(\frac{1}{2\pi/(2\gamma)}\right)^{d/2}e^{-\lVert\boldsymbol{\omega}\rVert^2/(4\gamma)} = \mathcal{N}(\mathbf{0}, 2\gamma I)$.

(With $\gamma = 1/2$: $p(\boldsymbol{\omega}) = \mathcal{N}(\mathbf{0}, I)$, which is the standard form.)

**Feature map construction (two variants):**

*Random Fourier Features (RFF):* Sample $\boldsymbol{\omega}_j \sim p$, $b_j \sim \mathcal{U}[0,2\pi]$:
$$\phi(\mathbf{x}) = \sqrt{2/D}\,[\cos(\boldsymbol{\omega}_1^\top\mathbf{x}+b_1),\ldots,\cos(\boldsymbol{\omega}_D^\top\mathbf{x}+b_D)]$$

*Random Fourier Features (paired, no bias):* Sample $\boldsymbol{\omega}_j \sim p$:
$$\phi(\mathbf{x}) = \frac{1}{\sqrt{D}}[\cos(\boldsymbol{\omega}_1^\top\mathbf{x}),\sin(\boldsymbol{\omega}_1^\top\mathbf{x}),\ldots,\cos(\boldsymbol{\omega}_D^\top\mathbf{x}),\sin(\boldsymbol{\omega}_D^\top\mathbf{x})]$$

Both give $\mathbb{E}[\phi(\mathbf{x})^\top\phi(\mathbf{y})] = k(\mathbf{x},\mathbf{y})$. The paired version has slightly lower variance.

**Concentration bound:**
$$P\!\left(\sup_{\mathbf{x},\mathbf{y}}\left|\phi(\mathbf{x})^\top\phi(\mathbf{y}) - k(\mathbf{x},\mathbf{y})\right| > \epsilon\right) \leq 2^8\left(\frac{\sigma_p R}{\epsilon}\right)^2 e^{-D\epsilon^2/8}$$

where $\sigma_p = \sqrt{\mathbb{E}_p[\lVert\boldsymbol{\omega}\rVert^2]}$ and $R = \max_i\lVert\mathbf{x}^{(i)}\rVert$.

**Variance reduction via Quasi-Monte Carlo:** Instead of random $\boldsymbol{\omega}_j$, use quasi-random sequences (Sobol, Halton) to improve coverage of frequency space. Gives $O(D^{-1}\log^d D)$ error vs $O(D^{-1/2})$ for random - significant improvement in practice.

**Orthogonal Random Features (ORF) (Yu et al., 2016):** Sample $\boldsymbol{\omega}_j$ as rows of $G = \frac{1}{\sqrt{D}}HDS$ where $H$ is a Hadamard matrix, $D$ is diagonal $\pm1$, $S$ is a diagonal scaling. This gives:
- Same unbiasedness: $\mathbb{E}[\phi(\mathbf{x})^\top\phi(\mathbf{y})] = k(\mathbf{x},\mathbf{y})$
- Lower variance: $O(1/D)$ vs $O(1/D)$ but with better constants
- Faster computation: $O(D\log D)$ using fast Hadamard transform

**Connection to Performer:** The Performer attention (Choromanski et al., 2021) approximates $\exp(q^\top k/\sqrt{d})$ (the unnormalized attention softmax kernel) using orthogonal random features:
$$\text{softmax}(q^\top k/\sqrt{d}) \approx \phi(q)^\top\phi(k), \quad \phi(\mathbf{x}) = \frac{e^{-\lVert\mathbf{x}\rVert^2/2}}{\sqrt{D}}[e^{\omega_1^\top\mathbf{x}},\ldots,e^{\omega_D^\top\mathbf{x}}]$$

This reduces attention from $O(N^2 d)$ to $O(N D d)$ - linear in sequence length $N$.

### D.3 Spectral Analysis of Neural Network Weight Matrices

The **spectral norm** $\lVert W\rVert_2 = \sigma_{\max}(W)$ and the **spectral distribution** of weight matrices encode important information about the training state of a neural network.

**Heavy-tailed self-regularization (Martin & Mahoney, 2021):** In well-trained neural networks, the eigenvalue distribution of $W^\top W$ follows a power law: $p(\lambda) \propto \lambda^{-\alpha}$ for some $\alpha \in [2, 6]$. Under-trained or over-regularized networks have bulk (Marchenko-Pastur) distributions. This is detected by fitting a power law to the sorted singular values.

**WeightWatcher:** The open-source tool `weightwatcher` analyzes the spectral distribution of weight matrices in PyTorch/TensorFlow models and reports $\alpha$ values as a proxy for model quality - without needing test data. Well-trained frontier models (GPT-4, LLaMA) have $\alpha \approx 2$-4; poorly trained models have $\alpha > 6$ or non-power-law distributions.

**Spectral norm and Lipschitz constant:** For a multi-layer network $f = f_L\circ\cdots\circ f_1$, the Lipschitz constant satisfies:
$$\text{Lip}(f) \leq \prod_{l=1}^L\lVert W^{[l]}\rVert_2\cdot\text{Lip}(\sigma)$$

For 1-Lipschitz activations ($\sigma = \operatorname{ReLU}$, $\text{Lip}(\sigma) = 1$): $\text{Lip}(f) \leq \prod_l\lVert W^{[l]}\rVert_2$.

Spectral normalization (Section 8.3) sets each $\lVert W^{[l]}\rVert_2 = 1$, making the entire network 1-Lipschitz.

**For LLMs:** Analyzing the spectral properties of attention weight matrices ($W_Q, W_K, W_V, W_O$) is an active area of mechanistic interpretability research. Low-rank structure in these matrices (small effective rank = few large singular values) indicates specialized computation. Anthropic's interpretability work has found "singular value signatures" of specific computational patterns in attention heads.

### D.4 The Fourier Perspective on Attention

Attention and the Fourier Transform are both **global linear mixing operations** on sequences. Understanding their relationship clarifies when each is appropriate.

**Attention as a data-dependent filter:** The output of attention can be written as:
$$y_i = \sum_j A_{ij}v_j, \quad A_{ij} = \frac{\exp(q_i^\top k_j/\sqrt{d_k})}{\sum_\ell\exp(q_i^\top k_\ell/\sqrt{d_k})}$$

For fixed query $i$, this is a weighted average of value vectors with data-dependent weights $A_{ij}$. In signal processing terms, it is a **position-dependent filter**: the filter coefficients $A_{ij}$ change for each output position $i$.

**Fourier Transform as a fixed filter bank:** FNet applies the DFT, which is a fixed (non-data-dependent) filter: the output at frequency $k$ is always $\hat{x}_k = \sum_n x_n e^{-2\pi ikn/N}$, regardless of the content of $x$. Each DFT frequency is a specific pattern of oscillation at rate $k/N$.

**Comparison table:**

| Property | Self-Attention | FNet (DFT) |
|---------|----------------|------------|
| Filter type | Data-dependent | Fixed |
| Parameters in mixer | $3d^2$ (W_Q, W_K, W_V) | 0 |
| Computational complexity | $O(N^2 d)$ | $O(Nd\log(Nd))$ |
| Expressive power | High (can select specific tokens) | Low (global uniform mixing) |
| Position awareness | Via PE or RoPE | Via DFT phases |
| Best for | Selective retrieval tasks | Global mixing tasks |
| Example success | Machine translation, QA | Text classification, embedding |

**The analogy:** Attention is like a learnable **short-time Fourier Transform with adaptive windows** - for each output position, it selects which "frequencies" (patterns) to attend to. The DFT in FNet is like a fixed global frequency analysis with no adaptivity. The power of attention comes from its ability to compute different effective filters for different positions and different inputs.


---

## Appendix E: Reference Summary and Quick Checks

### E.1 The Six Most Important Facts

If you retain only six things from this section, retain these:

**1. Definition:** $\hat{f}(\xi) = \int_{-\infty}^\infty f(t)e^{-2\pi i\xi t}\,dt$ (use the $\xi$-convention; no $2\pi$ in the inverse).

**2. Self-dual Gaussian:** $\mathcal{F}[e^{-\pi t^2}](\xi) = e^{-\pi\xi^2}$. The Gaussian maps to itself. All other transforms can be derived from this one via properties.

**3. Plancherel:** $\lVert\hat{f}\rVert_2 = \lVert f\rVert_2$. Energy is conserved. The FT is a unitary operator on $L^2$.

**4. Uncertainty principle:** $\Delta t\cdot\Delta\xi \geq 1/(4\pi)$, equality iff Gaussian. You cannot have both time and frequency concentration.

**5. Distributions:** $\mathcal{F}[\delta] = 1$, $\mathcal{F}[1] = \delta$, $\mathcal{F}[e^{2\pi i\xi_0 t}] = \delta(\xi-\xi_0)$. Pure tones map to spikes. Spikes have flat spectra.

**6. AI:** The FT is used in RoPE (modulation theorem), FNet (global mixing via DFT), RFF (Bochner + spectral sampling), spectral normalization (spectral norm of weight matrices), and FNO (learn in Fourier space).

### E.2 Quick Reference: Which Theorem to Use When

| Problem | Theorem/Tool |
|---------|-------------|
| Compute FT of $e^{-a|t|}$ | Direct integration + complex exponential |
| Compute FT of $f(t-a)g(t)$ | Time-shift + modulation properties |
| Show $\int|f|^2 = \int|\hat{f}|^2$ | Plancherel / Parseval |
| Find $\int_{-\infty}^\infty\operatorname{sinc}^2(\xi)\,d\xi$ | Parseval applied to rect |
| Show $\hat{f}(\xi)\to 0$ as $|\xi|\to\infty$ | Riemann-Lebesgue lemma |
| Compute FT of $f'(t)$ | Differentiation property: multiply by $2\pi i\xi$ |
| Compute FT of $\cos(2\pi\xi_0 t)$ | Modulation theorem or distribution: two delta spikes |
| Verify $f$ recoverable from $\hat{f}$ | Inversion theorem (check $f,\hat{f}\in L^1$ or use $L^2$ theory) |
| Lower bound on signal bandwidth | Uncertainty principle: $\Delta\xi \geq 1/(4\pi\Delta t)$ |
| FT of a periodic signal $\sum c_n e^{2\pi int/T}$ | Distribute over sum: $\sum c_n\delta(\xi - n/T)$ |
| FT of $f(at)$ (rescaled signal) | Scaling theorem: $(1/|a|)\hat{f}(\xi/a)$ |
| Show kernel $k(\mathbf{x}-\mathbf{y})$ is PD | Bochner: show spectral density $p\geq0$ |

### E.3 Sign Convention Reference

Different sources use different sign conventions. This table gives the conversion formulas.

| Convention | Forward FT | Inverse FT | $e^{-\pi t^2}$ maps to |
|-----------|-----------|-----------|----------------------|
| $\xi$ (this section) | $\int f(t)e^{-2\pi i\xi t}\,dt$ | $\int\hat{f}(\xi)e^{+2\pi i\xi t}\,d\xi$ | $e^{-\pi\xi^2}$ |
| $\omega$, no $\frac{1}{2\pi}$ | $\int f(t)e^{-i\omega t}\,dt$ | $\frac{1}{2\pi}\int F(\omega)e^{+i\omega t}\,d\omega$ | $\sqrt{2\pi}e^{-\omega^2/2}$ |
| $\omega$, symmetric | $\frac{1}{\sqrt{2\pi}}\int f(t)e^{-i\omega t}\,dt$ | $\frac{1}{\sqrt{2\pi}}\int F(\omega)e^{+i\omega t}\,d\omega$ | $e^{-\omega^2/2}$ |

**Conversion:** If $F_\xi(\xi)$ is the transform in the $\xi$-convention, then $F_\omega(\omega) = F_\xi(\omega/(2\pi))$ for the non-symmetric $\omega$-convention.

**Properties in the $\omega$-convention (no $1/(2\pi)$):**
- Differentiation: $\mathcal{F}_\omega[f'](\omega) = i\omega F(\omega)$ (not $2\pi i\xi$)
- Time-shift: $\mathcal{F}_\omega[f(t-a)](\omega) = e^{-i\omega a}F(\omega)$ (not $e^{-2\pi i\xi a}$)
- Parseval: $\int|f|^2 = \frac{1}{2\pi}\int|F|^2$ (extra $1/(2\pi)$ factor!)

The $\xi$-convention is self-symmetric and avoids all these $2\pi$ factors - strongly recommended for all new work.

### E.4 Prerequisites Revisited: What You Now Know

This section assumed you knew Fourier series (Section 01) and functional analysis basics. Here is what each prerequisite contributed:

| Prerequisite | How it was used |
|-------------|----------------|
| Complex exponentials $e^{i\theta}$ | The basis functions $e^{-2\pi i\xi t}$ of the FT; the Euler formula is used constantly |
| $L^2$ inner product | Plancherel's theorem is the FT version of Parseval from Fourier series |
| Orthonormality of $\{e^{inx}\}$ | Generalized: the FT is the "orthonormal expansion" in the continuous limit |
| Parseval's identity (series) | Generalized to Plancherel's theorem in Section 5 |
| Gibbs phenomenon | Reinterpreted as spectral leakage - the frequency-domain manifestation of truncation |
| Dirichlet conditions | Sufficient conditions for pointwise inversion (Section 4.1) |
| Integration / IBP | Proofs of differentiation theorem, inversion via approximate identity |
| Convergence of series | $L^2$ convergence in Plancherel proof; distributional convergence in Section 7 |

### E.5 Glossary

| Term | Definition |
|------|-----------|
| Fourier Transform | $\hat{f}(\xi) = \int f(t)e^{-2\pi i\xi t}\,dt$; maps time-domain functions to frequency-domain functions |
| Spectrum | The function $\hat{f}(\xi)$; describes which frequencies are present and at what amplitude/phase |
| Power spectrum | $|\hat{f}(\xi)|^2$; energy density per unit frequency |
| Plancherel theorem | $\lVert\hat{f}\rVert_2 = \lVert f\rVert_2$; FT is a unitary isometry on $L^2$ |
| Parseval's relation | $\int f\bar{g} = \int\hat{f}\bar{\hat{g}}$; the FT preserves inner products |
| Uncertainty principle | $\Delta t\cdot\Delta\xi \geq 1/(4\pi)$; time and frequency spread cannot simultaneously be small |
| Riemann-Lebesgue lemma | $\hat{f}(\xi)\to0$ as $|\xi|\to\infty$ for $f\in L^1$ |
| Dirac delta $\delta(t)$ | Distributional unit impulse; $\int\delta(t)\phi(t)\,dt = \phi(0)$; FT is constant 1 |
| Tempered distribution | Continuous linear functional on Schwartz space; includes all $L^p$ functions, $\delta$, and more |
| Schwartz space $\mathcal{S}$ | Smooth functions decaying faster than any polynomial; FT maps $\mathcal{S}$ to $\mathcal{S}$ bijectively |
| Dirac comb | $\sum_n\delta(t-nT)$; its FT is another comb with spacing $1/T$ |
| Poisson summation | $\sum_n f(n) = \sum_n\hat{f}(n)$; connects Fourier series and FT |
| Modulation theorem | $\mathcal{F}[e^{2\pi i\xi_0 t}f](\xi) = \hat{f}(\xi-\xi_0)$; multiplication by tone = frequency shift |
| Sinc function | $\operatorname{sinc}(\xi) = \sin(\pi\xi)/(\pi\xi)$; FT of rectangular pulse; ideal low-pass filter kernel |
| Spectral norm | $\lVert W\rVert_2 = \sigma_{\max}(W)$; largest singular value; Lipschitz constant of linear map $W$ |
| RFF | Random Fourier Features; $k(\mathbf{x}-\mathbf{y})\approx\phi(\mathbf{x})^\top\phi(\mathbf{y})$ via sampled frequencies from spectral density |
| STFT | Short-Time Fourier Transform; localizes FT analysis to a time window; gives spectrogram |
| FNet | Transformer variant replacing attention with 2D DFT; $O(N\log N)$ mixing |
| FNO | Fourier Neural Operator; learns PDE solutions by parameterizing the spectral operator |
| RoPE | Rotary Position Embedding; encodes position as rotation $e^{im\theta}$ in complex plane |


---

## Appendix F: The Fourier Transform in Practice - Case Studies

### F.1 Case Study: Audio Preprocessing for Whisper

OpenAI's Whisper (Radford et al., 2022) is a speech recognition model that accepts raw audio and produces text. Its preprocessing pipeline is a direct application of STFT and spectral analysis.

**Step 1: Resampling.** Raw audio is resampled to 16kHz. By Shannon's sampling theorem (Poisson summation): a signal sampled at 16kHz can faithfully represent frequencies up to 8kHz - which covers all speech frequencies (human voice: 80Hz-8kHz, formants: 200Hz-4kHz).

**Step 2: STFT.** Apply a windowed Fourier Transform with:
- Window: Hann window (smooth tapering reduces spectral leakage)
- Window length: 25ms  16kHz = 400 samples
- Hop length: 10ms  16kHz = 160 samples (75% overlap)
- FFT size: 512 points (zero-padded for frequency resolution)

This gives a spectrogram of shape $(\text{frames} \times 257)$ where 257 = 512/2 + 1 unique frequencies.

**Step 3: Mel filterbank.** Map the 257 linear frequencies to 80 mel-scale frequency bins using triangular filters. The mel scale $m = 2595\log_{10}(1 + f/700)$ is perceptually uniform - equal mel intervals are equally distinguishable to humans. The uncertainty principle is no longer limiting here because the window (25ms) is longer than the smallest perceptible phoneme duration (~10-20ms), so time resolution is slightly sacrificed for good frequency resolution.

**Step 4: Log compression.** Take $\log(\max(\text{mel spectrogram}, 10^{-10}))$. The log operation compresses the dynamic range of audio energy (from ~120 dB to ~12 dB).

**Result:** $(\text{frames} \times 80)$ log-mel spectrogram, treated as a 2D "image" and fed to a CNN + Transformer encoder. The entire preprocessing is fixed (not learned) - it embeds centuries of signal processing knowledge into the model architecture.

**Fourier analysis used:**
- Fourier Transform -> STFT window analysis
- Uncertainty principle -> window length tradeoff
- Sampling theorem -> 16kHz minimum rate
- Spectral leakage -> Hann window choice

### F.2 Case Study: LLaMA-3 Context Extension via RoPE Scaling

LLaMA-3 uses RoPE with base $\theta_{\text{base}} = 500{,}000$ and supports up to 128K token context (compared to LLaMA-2's 4K with $\theta_{\text{base}} = 10{,}000$).

**The problem:** Standard RoPE with $\theta_j = 10000^{-2j/d}$ has frequencies ranging from $\theta_0 = 1$ (dimension pair 0) to $\theta_{d/2-1} = 10^{-4}$ (last dimension pair). At frequency $\theta_j$, the RoPE encoding completes one full rotation (period = $2\pi/\theta_j$ tokens). The maximum distinguishable context is the longest period: $T_{\max} = 2\pi/\theta_{\min} = 2\pi\times 10^4 \approx 62{,}832$ tokens for $d = 4096$ - explaining why LLaMA-2 degrades beyond ~4K (it uses $\theta_{\text{base}} = 10000$, so the longest period is $2\pi\times 10^4/d$... the exact computation depends on $d$).

**The Fourier interpretation:** Each dimension pair is an oscillator at frequency $\theta_j$. For two positions $m$ and $n$ to be distinguishable, their phase difference $|(m-n)\theta_j|$ must be $\neq 0$ mod $2\pi$ for at least one $j$. If $|m-n|$ is larger than $2\pi/\theta_{\min}$, all dimensions have wrapped around - positions become identical. This is aliasing in the temporal domain, caused by the frequency $\theta_{\min}$ being too high.

**The fix:** Reduce the minimum frequency by increasing $\theta_{\text{base}}$:
$$\theta_j = \theta_{\text{base}}^{-2j/d}, \quad \theta_{\min} = \theta_{\text{base}}^{-1}, \quad T_{\max} = 2\pi\cdot\theta_{\text{base}}$$

LLaMA-3 uses $\theta_{\text{base}} = 500{,}000$, giving $T_{\max} = 2\pi\times 500{,}000 \approx 3.14\times 10^6$ - more than enough for 128K context.

**The cost:** Higher $\theta_{\text{base}}$ means the low-frequency dimensions ($j$ large) oscillate very slowly, providing little positional information for short contexts. There is a fundamental tradeoff between context length and short-range positional precision - a manifestation of the uncertainty principle in the discrete position domain.

**YaRN (Peng et al., 2023):** Instead of uniformly rescaling all frequencies, YaRN uses a non-uniform rescaling: high-frequency dimensions (small $j$) are scaled minimally (they handle short-range position fine), while low-frequency dimensions (large $j$) are scaled aggressively. The interpolation factor $s = \text{target context} / \text{original context}$ is applied per-frequency, weighted by the "needed" extension. This maintains short-range precision while extending long-range capacity - a principled application of the time-frequency tradeoff.

### F.3 Case Study: Spectral Normalization in Stable Diffusion

Modern diffusion models (Stable Diffusion, DALL-E 3, Flux) train a discriminator or guidance network to distinguish real from generated data. Spectral normalization (Section 8.3) is used to stabilize this training.

**Setup:** The discriminator $D: \mathbb{R}^{H\times W\times 3} \to [0,1]$ has $L$ convolutional layers. Each convolution layer has weight $W^{[l]} \in \mathbb{R}^{C_\text{out}\times C_\text{in}\times k\times k}$. For a 2D convolution layer, the relevant spectral norm is:

$$\lVert W^{[l]}\rVert_2 = \max_{\mathbf{h}:\lVert\mathbf{h}\rVert=1}\lVert W^{[l]}*\mathbf{h}\rVert$$

For a convolution with kernel $h(k_x, k_y)$, the spectral norm equals $\max_{\xi_x,\xi_y}|\hat{h}(\xi_x,\xi_y)|$ - the maximum of the 2D Fourier Transform of the kernel. This connects spectral normalization directly to the FT: normalizing by the spectral norm makes the filter's frequency response bounded by 1 at every frequency.

**Effect on generated images:** A spectrally-normalized discriminator applies Lipschitz constraints that prevent it from exploiting arbitrary high-frequency artifacts in generated images. This forces the generator to eliminate high-frequency artifacts (checkerboard patterns from transposed convolutions, aliasing from bilinear upsampling) - improving sample quality.

**Alias-free GAN / StyleGAN3 (Karras et al., 2021):** Takes spectral control further by enforcing alias-free processing throughout the generator using carefully designed low-pass filters at every upsampling step. The key insight from Fourier analysis: any nonlinearity applied to a bandlimited signal creates harmonics beyond the original bandwidth - these must be filtered out before downsampling to avoid aliasing. StyleGAN3's architecture enforces this principle throughout, producing significantly more temporally consistent video outputs.


---

## Appendix G: Further Reading and References

### G.1 Foundational Texts

**For the classical theory:**
- Stein, E.M. & Weiss, G. (1971). *Introduction to Fourier Analysis on Euclidean Spaces*. Princeton University Press. - The standard reference for rigorous Fourier analysis.
- Katznelson, Y. (2004). *An Introduction to Harmonic Analysis*, 3rd ed. Cambridge. - Concise and rigorous; excellent for the abstract perspective.
- Dym, H. & McKean, H.P. (1972). *Fourier Series and Integrals*. Academic Press. - Beautiful, accessible treatment with applications to probability.
- Korner, T.W. (1988). *Fourier Analysis*. Cambridge. - The most readable and historically motivated account.

**For distributions and Schwartz space:**
- Schwartz, L. (1950). *Theorie des Distributions*. Herman, Paris. - The original; defines tempered distributions.
- Hormander, L. (1990). *The Analysis of Linear Partial Differential Operators I*. Springer. - Comprehensive; connects distributions to PDE theory.

**For signal processing:**
- Oppenheim, A.V. & Schafer, R.W. (2009). *Discrete-Time Signal Processing*, 3rd ed. Prentice Hall. - Standard DSP textbook.
- Mallat, S. (2008). *A Wavelet Tour of Signal Processing*, 3rd ed. Academic Press. - Bridges Fourier analysis, wavelets, and sparse signal processing.

### G.2 Machine Learning Papers

**FNet:**
- Lee-Thorp, J., Ainslie, J., Eckstein, I. & Ontanon, S. (2022). FNet: Mixing Tokens with Fourier Transforms. *NAACL-HLT*. arXiv:2105.03824.

**Random Fourier Features:**
- Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. *NeurIPS*. [Best Paper Award]
- Yu, F.X. et al. (2016). Orthogonal Random Features. *NeurIPS*.
- Choromanski, K. et al. (2021). Rethinking Attention with Performers. *ICLR*. arXiv:2009.14794.

**Spectral Normalization:**
- Miyato, T., Kataoka, T., Koyama, M. & Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*. arXiv:1802.05957.

**Fourier Neural Operator:**
- Li, Z. et al. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. *ICLR*. arXiv:2010.08895.

**RoPE and Context Extension:**
- Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
- Peng, B. et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models. arXiv:2309.00071.
- Ding, Y. et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens. arXiv:2402.13753.

**Spectral Bias:**
- Rahaman, N. et al. (2019). On the Spectral Bias of Neural Networks. *ICML*. arXiv:1806.08734.
- Tancik, M. et al. (2020). Fourier Features Let Networks Learn High Frequency Functions. *NeurIPS*. arXiv:2006.10739.

**Audio Models:**
- Radford, A. et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). arXiv:2212.04356.

**Alias-free GAN:**
- Karras, T. et al. (2021). Alias-Free Generative Adversarial Networks. *NeurIPS*. arXiv:2106.12423.

### G.3 Online Resources

- MIT OCW 18.103 (Fourier Analysis) - rigorous mathematical treatment with lecture notes
- Stanford EE261 (The Fourier Transform and its Applications) - engineering-oriented, excellent visualization
- 3Blue1Brown "But what is the Fourier Transform?" - geometric intuition for absolute beginners
- Seeing Theory (Brown University) - interactive visualizations of Fourier series and transforms


---

## Appendix H: Self-Assessment Questions

These questions test conceptual understanding without computation. Answer each in 2-3 sentences.

1. **Explain in words why the Fourier Transform of a spike (Dirac delta) is a constant (flat spectrum).** What physical interpretation does this have?

2. **A signal engineer claims: "If I double the duration of my pulse, I can halve its bandwidth." Is this claim correct, and what theorem guarantees it?** What is the fundamental limit on simultaneous time and frequency concentration?

3. **What is the difference between the Parseval identity from Fourier series (Section 01) and Plancherel's theorem from this section?** Are they the same result in different forms?

4. **Why does the Fourier Transform of a real-valued function $f$ satisfy $\hat{f}(-\xi) = \overline{\hat{f}(\xi)}$? What does this mean for the magnitude and phase spectra?**

5. **Explain why FNet achieves near-BERT performance on classification tasks but not on extractive QA.** What mathematical property of the DFT explains this performance gap?

6. **In RoPE, the frequencies $\theta_j = 10000^{-2j/d}$ are chosen geometrically. Why geometric spacing? What would happen if the frequencies were linearly spaced?**

7. **Bochner's theorem says a shift-invariant kernel is PD iff it is the FT of a non-negative measure. What breaks down if the spectral density is not non-negative?** Give an example.

8. **Explain the connection between the Poisson summation formula and Shannon's sampling theorem.** Why does undersampling (below the Nyquist rate) cause aliasing?

9. **The heat equation solution is a convolution with a Gaussian that spreads over time. In the Fourier domain, how does the solution evolve?** Which frequencies decay fastest and why?

10. **The differentiation property states $\mathcal{F}[f'](\xi) = 2\pi i\xi\hat{f}(\xi)$. Use this to explain why smoother functions have faster spectral decay.** What happens for a function with a jump discontinuity?

11. **In spectral normalization, the discriminator is normalized by $\sigma_{\max}(W)$. Why does this enforce a Lipschitz constraint?** How does the Lipschitz constant relate to training stability?

12. **The Gaussian $e^{-\pi t^2}$ is the unique minimizer of the uncertainty bound. Does this mean we should always use Gaussian signals in practice?** What are the tradeoffs?


---

*End of Section 20-02 Fourier Transform.*

*Next: [Section 20-03 DFT and FFT](../03-Discrete-Fourier-Transform-and-FFT/notes.md) - discretizing the Fourier Transform for computation via the Cooley-Tukey algorithm.*

