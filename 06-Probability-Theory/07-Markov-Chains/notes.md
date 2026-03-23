[← Back to Chapter 6: Probability Theory](../README.md) | [Next: Chapter 7 Statistics →](../../07-Statistics/README.md)

---

# Markov Chains

> _"A sequence of experiments forms a simple Markov chain if the conditional distribution of each experiment, given all preceding ones, depends only on the immediately preceding experiment."_
> — Andrei Markov, 1906

## Overview

A **Markov chain** is a stochastic process with a single, powerful property: the future depends on the past only through the present. This memorylessness — the **Markov property** — is both a mathematical simplification and a profound modelling insight. It says: knowing the current state is a complete sufficient statistic for predicting what comes next.

This simplicity conceals enormous richness. Markov chains model language (every autoregressive LLM generates tokens one at a time, conditioning on the current state of the context), web structure (PageRank computes the stationary distribution of a Markov chain on the web graph), Bayesian computation (MCMC algorithms construct Markov chains whose stationary distribution is the target posterior), reinforcement learning (MDPs are Markov chains with actions), and speech recognition (HMMs). The mathematics of Markov chains — transition matrices, stationary distributions, mixing times — is the engine beneath all of these.

This section builds the complete theory from first principles: formal definitions, state classification, existence and uniqueness of stationary distributions via Perron-Frobenius, convergence to stationarity and mixing times, reversibility and detailed balance, continuous-time chains, and the MCMC algorithms central to Bayesian ML.

## Prerequisites

- [§06 Stochastic Processes](../06-Stochastic-Processes/notes.md) — filtrations, adapted processes, the Markov property preview, random walks
- [§03 Joint Distributions](../03-Joint-Distributions/notes.md) — conditional distributions, conditional independence, Bayes' theorem
- [§04 Expectation and Moments](../04-Expectation-and-Moments/notes.md) — conditional expectation, tower property
- [§02 Common Distributions](../02-Common-Distributions/notes.md) — Poisson distribution (for CTMC); Gaussian (for MCMC targets)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations: transition matrices, power iteration, classification, stationary distributions, mixing times, Metropolis-Hastings, Gibbs, PageRank, HMMs |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from state classification to PageRank and HMM decoding |

## Learning Objectives

After completing this section, you will be able to:

1. State the Markov property and express it as conditional independence
2. Write the transition matrix of a Markov chain and compute $n$-step probabilities via $P^n$
3. Classify states as communicating, recurrent/transient, periodic/aperiodic, and absorbing
4. State the Perron-Frobenius theorem and use it to establish existence and uniqueness of stationary distributions for ergodic chains
5. Compute stationary distributions via linear systems and power iteration
6. State the detailed balance equations and explain how they relate to reversibility
7. Define total variation distance and mixing time; bound mixing time via the spectral gap
8. Write the generator matrix of a CTMC and derive its stationary distribution
9. Implement Metropolis-Hastings and verify it satisfies detailed balance for the target distribution
10. Implement Gibbs sampling and explain why it is a special case of Metropolis-Hastings
11. Compute PageRank via power iteration with teleportation
12. Describe the HMM forward algorithm and Viterbi decoding

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is a Markov Chain?](#11-what-is-a-markov-chain)
  - [1.2 Everywhere in AI](#12-everywhere-in-ai)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 Taxonomy of Markov Chains](#14-taxonomy-of-markov-chains)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The Markov Property](#21-the-markov-property)
  - [2.2 Transition Matrix](#22-transition-matrix)
  - [2.3 Chapman-Kolmogorov Equations](#23-chapman-kolmogorov-equations)
  - [2.4 Standard Examples](#24-standard-examples)
- [3. Classification of States](#3-classification-of-states)
  - [3.1 Communicating Classes and Irreducibility](#31-communicating-classes-and-irreducibility)
  - [3.2 Recurrence and Transience](#32-recurrence-and-transience)
  - [3.3 Periodicity](#33-periodicity)
  - [3.4 Absorbing States and Absorption Probabilities](#34-absorbing-states-and-absorption-probabilities)
  - [3.5 Ergodic Chains](#35-ergodic-chains)
- [4. Stationary Distributions](#4-stationary-distributions)
  - [4.1 Definition and Existence](#41-definition-and-existence)
  - [4.2 Perron-Frobenius Theorem](#42-perron-frobenius-theorem)
  - [4.3 Convergence to Stationarity](#43-convergence-to-stationarity)
  - [4.4 Null-Recurrent Chains](#44-null-recurrent-chains)
  - [4.5 Computing Stationary Distributions](#45-computing-stationary-distributions)
- [5. Detailed Balance and Reversibility](#5-detailed-balance-and-reversibility)
  - [5.1 Detailed Balance Equations](#51-detailed-balance-equations)
  - [5.2 Reversible Markov Chains](#52-reversible-markov-chains)
  - [5.3 Birth-Death Chains](#53-birth-death-chains)
  - [5.4 Spectral Decomposition of Reversible Chains](#54-spectral-decomposition-of-reversible-chains)
- [6. Mixing Times and Convergence Rates](#6-mixing-times-and-convergence-rates)
  - [6.1 Total Variation Distance](#61-total-variation-distance)
  - [6.2 Mixing Time](#62-mixing-time)
  - [6.3 Spectral Gap](#63-spectral-gap)
  - [6.4 Coupling Arguments](#64-coupling-arguments)
  - [6.5 AI Relevance: Fast Mixing and MCMC Warm Starts](#65-ai-relevance-fast-mixing-and-mcmc-warm-starts)
- [7. Continuous-Time Markov Chains](#7-continuous-time-markov-chains)
  - [7.1 Generator Matrix](#71-generator-matrix)
  - [7.2 Kolmogorov Equations](#72-kolmogorov-equations)
  - [7.3 Stationary Distribution of CTMCs](#73-stationary-distribution-of-ctmcs)
  - [7.4 Poisson Process as CTMC](#74-poisson-process-as-ctmc)
- [8. MCMC: Markov Chain Monte Carlo](#8-mcmc-markov-chain-monte-carlo)
  - [8.1 The Sampling Problem](#81-the-sampling-problem)
  - [8.2 Metropolis-Hastings](#82-metropolis-hastings)
  - [8.3 Gibbs Sampling](#83-gibbs-sampling)
  - [8.4 Convergence Diagnostics](#84-convergence-diagnostics)
  - [8.5 Hamiltonian Monte Carlo](#85-hamiltonian-monte-carlo)
- [9. ML Deep Dive](#9-ml-deep-dive)
  - [9.1 Language Model Generation](#91-language-model-generation)
  - [9.2 PageRank](#92-pagerank)
  - [9.3 Markov Decision Processes](#93-markov-decision-processes)
  - [9.4 Hidden Markov Models](#94-hidden-markov-models)
  - [9.5 Diffusion Models as CTMCs](#95-diffusion-models-as-ctmcs)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [13. Conceptual Bridge](#13-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is a Markov Chain?

A **Markov chain** is a sequence of random variables $X_0, X_1, X_2, \ldots$ taking values in a state space $\mathcal{S}$, where the future is independent of the past given the present. Formally, for every $n \geq 0$ and every state $j \in \mathcal{S}$:
$$P(X_{n+1} = j \mid X_0, X_1, \ldots, X_n) = P(X_{n+1} = j \mid X_n)$$

This is the **Markov property**: the current state $X_n$ is a sufficient statistic for predicting $X_{n+1}$. No matter how long the history, knowing only the present gives as much information about the future as knowing the entire past trajectory.

**Why memorylessness is powerful.** At first glance, the Markov property looks like a restriction — real systems do have memory. But three things make it useful:

1. **Many systems genuinely satisfy it.** A fair die, re-rolled at each step, is Markov (no memory). A chess position captures all relevant game state (Markov). An LLM's key-value cache provides sufficient statistics for the next token (Markov, given the KV cache).

2. **Higher-order Markov chains reduce to first-order.** A chain where $X_{n+1}$ depends on $X_n$ and $X_{n-1}$ can be converted to a first-order chain on the augmented state $(X_n, X_{n-1})$. So first-order is universal given a rich enough state space.

3. **The mathematics is tractable.** Transition probabilities only involve pairs of states, so the full dynamics are captured by a matrix rather than an exponentially large conditional probability table.

**The mental picture:** Imagine a particle hopping between states on a graph. At each step, it looks only at which node it currently occupies, then jumps to a neighbouring node with probability given by the edge weights. The history of how it got to the current node is irrelevant — only where it is now matters.

**For AI:** Every autoregressive language model generates tokens one at a time, each conditioned on all previous tokens. Given a context window of length $n$, the distribution over the next token is fully specified by the current state (the context). This is precisely the Markov property with state space = all possible contexts of length $\leq n$.

### 1.2 Everywhere in AI

Markov chains appear as the backbone of several major AI paradigms:

**Language generation:** GPT, LLaMA, and every autoregressive transformer generates text via a Markov chain. The state is the current context (the key-value cache), the transition is the softmax over the vocabulary, and the stationary distribution — if it exists — characterises what the model "talks about" at equilibrium.

**PageRank:** Google's original ranking algorithm computes the stationary distribution of a Markov chain on the web graph, where each page links to others. A page is important if the random surfer spends a lot of time there at stationarity. Power iteration on the transition matrix gives the PageRank vector.

**Reinforcement learning:** A Markov Decision Process (MDP) is a Markov chain with actions. The agent's policy induces a Markov chain over states, and the value function is the expected discounted return from each state — computable from the stationary properties of this chain. TD-learning, Q-learning, and policy gradient methods all exploit the Markov structure.

**Bayesian inference via MCMC:** When the posterior distribution $\pi(\theta | \mathcal{D})$ is intractable, MCMC constructs a Markov chain whose stationary distribution is exactly $\pi$. Running the chain for long enough generates samples that represent the posterior. Metropolis-Hastings and Gibbs sampling are the two canonical MCMC algorithms.

**Speech and sequence modelling:** Hidden Markov Models (HMMs) model sequences where the true state (e.g., phoneme) is latent. The forward-backward algorithm and Viterbi decoder exploit the Markov structure to perform efficient inference.

**Diffusion models:** The DDPM forward process $x_0 \to x_1 \to \cdots \to x_T$ is a Markov chain where each step adds Gaussian noise. The learned reverse process is a Markov chain running backwards in time. The SDE perspective ([§06 Stochastic Processes](../06-Stochastic-Processes/notes.md)) gives the continuous-time version.

### 1.3 Historical Timeline

| Year | Contributor | Result |
| --- | --- | --- |
| 1906 | Andrei Markov | Defined Markov chains; proved the law of large numbers for dependent sequences |
| 1907 | Andrei Markov | First application: statistical analysis of letters in Pushkin's *Eugene Onegin* |
| 1913 | Paul Ehrenfest | Urn model illustrating approach to equilibrium; connection to statistical mechanics |
| 1928 | Andrei Kolmogorov | Backward equations for Markov processes; rigorous foundation |
| 1936 | Oskar Perron, Georg Frobenius | Perron-Frobenius theorem (positive matrices have unique dominant eigenvector) |
| 1950 | Nicholas Metropolis et al. | Monte Carlo method; Metropolis algorithm for statistical mechanics |
| 1953 | Metropolis, Rosenbluth×2, Teller×2 | Metropolis-Hastings algorithm published; first MCMC |
| 1970 | W.K. Hastings | Generalized Metropolis to asymmetric proposals |
| 1984 | Geman & Geman | Gibbs sampling for image restoration |
| 1996 | Levin, Peres, Wilmer | Systematic mixing time theory; coupling and spectral gap bounds |
| 1998 | Page, Brin, Motwani, Winograd | PageRank: web ranking via Markov chain stationary distribution |
| 2015 | Ho, Jain & Abbeel | DDPM: diffusion as Markov chain with learned reverse |
| 2022 | Various | LLMs as Markov chains; stationary distribution theory for language models |

### 1.4 Taxonomy of Markov Chains

Markov chains are classified along two dimensions:

**State space:** Finite ($\mathcal{S} = \{1,\ldots,N\}$), countably infinite ($\mathcal{S} = \mathbb{Z}^+$), or continuous ($\mathcal{S} = \mathbb{R}^d$). Most of this section treats finite or countably infinite state spaces; MCMC typically targets continuous spaces.

**Time:** Discrete ($n = 0, 1, 2, \ldots$) or continuous ($t \geq 0$). Discrete-time chains are developed in §2–§6; continuous-time chains in §7.

| | Discrete Time | Continuous Time |
| --- | --- | --- |
| **Finite states** | Finite DTMC — PageRank, HMM | CTMC — queueing theory |
| **Countable states** | SRW, birth-death chain | Poisson process |
| **Continuous states** | MCMC on $\mathbb{R}^d$ | Diffusion SDE (§06) |

**Time-homogeneous vs. inhomogeneous:** If transition probabilities do not depend on $n$ — $P(X_{n+1}=j|X_n=i) = P_{ij}$ for all $n$ — the chain is **time-homogeneous**. Almost all chains in this section are time-homogeneous.

---

## 2. Formal Definitions

### 2.1 The Markov Property

> **Recall:** Conditional independence $X \perp Y \mid Z$ was defined in [§03 Joint Distributions](../03-Joint-Distributions/notes.md#conditional-independence). The Markov property is conditional independence of the future from the past, given the present.

**Definition (Discrete-Time Markov Chain).** A sequence $\{X_n\}_{n \geq 0}$ of random variables on a measurable state space $(\mathcal{S}, \mathcal{B})$ is a **(time-homogeneous) Markov chain** if for all $n \geq 0$ and all measurable $B \subseteq \mathcal{S}$:
$$P(X_{n+1} \in B \mid X_0, X_1, \ldots, X_n) = P(X_{n+1} \in B \mid X_n) =: K(X_n, B)$$

The function $K : \mathcal{S} \times \mathcal{B} \to [0,1]$ is called the **transition kernel** or **transition probability**. For finite $\mathcal{S} = \{1,\ldots,N\}$, it is encoded by the $N \times N$ **transition matrix** $P$ with $P_{ij} = P(X_{n+1}=j \mid X_n=i)$.

**Equivalent formulation via conditional independence:** The Markov property is equivalent to:
$$X_{n+1} \perp (X_0, \ldots, X_{n-1}) \mid X_n$$
This is the cleanest statement: given the present $X_n$, the future $X_{n+1}$ is independent of the entire past $(X_0,\ldots,X_{n-1})$. More generally, the **strong Markov property** asserts this holds for stopping times: $X_{\tau+1} \perp \mathcal{F}_\tau \mid X_\tau$ for any stopping time $\tau$.

**Initial distribution:** The full law of the chain is specified by:
1. The initial distribution $\mu_0$ with $\mu_0(i) = P(X_0 = i)$
2. The transition matrix $P$ with $P_{ij} \geq 0$ and $\sum_j P_{ij} = 1$ for all $i$

Together, $(\mu_0, P)$ determines all finite-dimensional distributions.

### 2.2 Transition Matrix

For a finite state space $\mathcal{S} = \{1,\ldots,N\}$, the transition matrix $P$ is an $N \times N$ **stochastic matrix**: every row sums to 1, all entries are non-negative. The $(i,j)$ entry is the probability of jumping from state $i$ to state $j$ in one step.

**Distribution at step $n$:** If $\mu^{(n)}$ is the row vector of probabilities at step $n$, then:
$$\mu^{(n)} = \mu^{(0)} P^n$$

The distribution evolves by left-multiplying by $P$ at each step. In component form: $\mu^{(n+1)}_j = \sum_i \mu^{(n)}_i P_{ij}$.

**$n$-step transition probabilities:** The probability of going from $i$ to $j$ in exactly $n$ steps is the $(i,j)$ entry of $P^n$:
$$P^{(n)}_{ij} = P(X_n = j \mid X_0 = i) = (P^n)_{ij}$$

Computing $P^n$ via matrix exponentiation (eigendecomposition or repeated squaring) is the core computational task for Markov chains.

**Non-example:** A matrix with a row summing to more than 1, or with any negative entry, is NOT a valid transition matrix. A doubly-stochastic matrix (both rows and columns sum to 1) is a transition matrix, and its stationary distribution is uniform.

**For AI:** In a language model with vocabulary size $V$, the transition matrix $P$ has size $V \times V$ where $P_{ij}$ = probability of token $j$ following token $i$ (bigram model). Real LLMs use much richer state spaces (the context window), but the Markov structure is the same.

### 2.3 Chapman-Kolmogorov Equations

**Theorem (Chapman-Kolmogorov).** For any $m, n \geq 0$ and states $i, j$:
$$P^{(m+n)}_{ij} = \sum_{k \in \mathcal{S}} P^{(m)}_{ik} P^{(n)}_{kj}$$

In matrix form: $P^{m+n} = P^m P^n$. This is simply the rule for matrix multiplication — to go from $i$ to $j$ in $m+n$ steps, sum over all intermediate states $k$ at the midpoint.

**Proof:** By the law of total probability and the Markov property:
$$P(X_{m+n}=j \mid X_0=i) = \sum_k P(X_{m+n}=j \mid X_m=k) P(X_m=k \mid X_0=i) = \sum_k P^{(n)}_{kj} P^{(m)}_{ik}$$

Chapman-Kolmogorov is the reason matrix multiplication is the right operation for composing Markov chain transitions. It also says: to predict $n+m$ steps ahead, you can decompose the prediction at any intermediate time $m$.

### 2.4 Standard Examples

**Example 1: Two-state chain (weather model)**

$$P = \begin{pmatrix} 1-p & p \\ q & 1-q \end{pmatrix}$$

State 1 = sunny, state 2 = rainy. Sunny stays sunny with probability $1-p$, turns rainy with probability $p$. Rainy turns sunny with probability $q$. The stationary distribution is $\pi = (q/(p+q),\ p/(p+q))$.

**Example 2: Simple Random Walk on $\{0,1,\ldots,N\}$ with reflecting barriers**

$$P_{ij} = \begin{cases} 1/2 & |i-j|=1,\ 0 < i < N \\ 1 & i=0, j=1 \text{ or } i=N, j=N-1 \end{cases}$$

The particle bounces off the walls at 0 and $N$. Stationary distribution: uniform.

**Example 3: PageRank chain**

$$P_{ij} = \alpha \cdot \frac{A_{ij}}{\text{out-degree}(i)} + (1-\alpha) \cdot \frac{1}{N}$$

where $A$ is the web adjacency matrix and $\alpha \in (0,1)$ is the damping factor (usually $\alpha=0.85$). The teleportation term $(1-\alpha)/N$ ensures the chain is irreducible and aperiodic. The stationary distribution is the PageRank vector.

**Example 4: Gibbs chain on $\{0,1\}^2$**

For the Ising model on two spins, $P$ is a $4 \times 4$ matrix constructed by randomly updating one spin at a time conditional on the other. This is the simplest MCMC example.

---

## 3. Classification of States

### 3.1 Communicating Classes and Irreducibility

**Definition.** State $j$ is **accessible** from state $i$ (written $i \to j$) if there exists $n \geq 0$ such that $P^{(n)}_{ij} > 0$. States $i$ and $j$ **communicate** (written $i \leftrightarrow j$) if $i \to j$ and $j \to i$.

Communication is an equivalence relation, and its equivalence classes are called **communicating classes**. A communicating class $C$ is **closed** if no state outside $C$ is accessible from any state inside $C$: $i \in C$ and $i \to j$ implies $j \in C$.

**Definition (Irreducible Chain).** A Markov chain is **irreducible** if it has exactly one communicating class — every state communicates with every other. Equivalently, for every pair $(i,j)$, there exists $n$ with $P^{(n)}_{ij} > 0$.

Irreducibility is the "connectivity" condition for Markov chains. An irreducible chain cannot get trapped in a subset of states. Most of the theory (stationary distributions, convergence) requires irreducibility.

**Decomposition theorem:** Any finite Markov chain decomposes into transient states (a particle eventually leaves forever) and one or more closed communicating classes of recurrent states. Once the chain enters a closed class, it stays there forever.

**For AI:** A language model's transition matrix is nearly irreducible: any token sequence of sufficient length can be generated (in principle). Practical degeneracy (zero-probability transitions due to softmax temperatures → 0) creates absorbing states.

### 3.2 Recurrence and Transience

**Definition.** Define the **first return time** to state $i$: $T_i = \min\{n \geq 1 : X_n = i\}$, with $T_i = \infty$ if the chain never returns. The **return probability** is:
$$f_i = P(T_i < \infty \mid X_0 = i)$$

- State $i$ is **recurrent** if $f_i = 1$ (return is certain)
- State $i$ is **transient** if $f_i < 1$ (positive probability of never returning)

**Theorem (Recurrence criterion).** State $i$ is recurrent if and only if $\sum_{n=0}^\infty P^{(n)}_{ii} = \infty$. It is transient if and only if $\sum_{n=0}^\infty P^{(n)}_{ii} < \infty$.

*Proof sketch:* Let $R_i = \sum_{n=1}^\infty \mathbf{1}[X_n=i]$ be the number of returns to $i$. Then $\mathbb{E}[R_i \mid X_0=i] = \sum_{n=1}^\infty P^{(n)}_{ii}$. If $f_i=1$, returns are geometrically distributed with mean $\infty$, so the sum diverges. If $f_i < 1$, the number of returns is geometric with finite mean, so the sum converges.

**Positive vs. null recurrence:** Among recurrent states, define the **mean return time** $\mu_i = \mathbb{E}[T_i \mid X_0=i]$.

- **Positive recurrent**: $f_i = 1$ and $\mu_i < \infty$ (returns in finite average time)
- **Null recurrent**: $f_i = 1$ but $\mu_i = \infty$ (certain to return, but takes forever on average)

All recurrent states in a finite irreducible chain are positive recurrent (finite chains cannot be null-recurrent). The simple random walk on $\mathbb{Z}$ is null-recurrent in 1D and 2D, transient in 3D and above (Pólya's theorem).

**For AI:** In MCMC, we want the chain to be recurrent (it returns to every region of the posterior). A transient MCMC chain would drift away from the posterior, giving biased samples.

### 3.3 Periodicity

**Definition.** The **period** of state $i$ is:
$$d(i) = \gcd\{n \geq 1 : P^{(n)}_{ii} > 0\}$$

State $i$ is **aperiodic** if $d(i) = 1$; otherwise it is **periodic** with period $d(i)$.

**Key facts:**
1. All states in the same communicating class have the same period
2. A chain is called **aperiodic** if all states have period 1
3. If $P_{ii} > 0$ for some $i$, then $d(i) = 1$ (self-loops force aperiodicity)
4. An irreducible aperiodic chain converges to its stationary distribution (§4.3)

**Example:** A two-state chain $0 \leftrightarrow 1$ with $P_{01}=P_{10}=1$ is periodic with period 2: the chain alternates 0,1,0,1,... and $P^{(n)}_{00} = 1$ only for even $n$.

**Periodicity breaks convergence:** If a chain has period $d > 1$, then $P^{(n)}_{ij}$ does not converge as $n \to \infty$ — it oscillates. Adding any self-loop probability $\varepsilon > 0$ makes the chain aperiodic.

**For AI:** Metropolis-Hastings with a symmetric proposal that includes the possibility of staying put is automatically aperiodic (self-loops happen when a proposal is rejected). This is why MCMC chains converge.

### 3.4 Absorbing States and Absorption Probabilities

**Definition.** State $i$ is **absorbing** if $P_{ii} = 1$ (the chain stays at $i$ forever once it arrives). A chain with at least one absorbing state is called **absorbing**.

**Canonical form:** For an absorbing chain with $r$ absorbing states and $t$ transient states, the transition matrix can be written in canonical form:
$$P = \begin{pmatrix} Q & R \\ 0 & I_r \end{pmatrix}$$

where $Q$ is the $t \times t$ sub-matrix of transitions among transient states, $R$ is the $t \times r$ matrix of transitions to absorbing states, and $I_r$ is the identity on absorbing states.

**Fundamental matrix:** $N = (I - Q)^{-1}$. The $(i,j)$ entry of $N$ is the expected number of times the chain visits transient state $j$ before absorption, starting from transient state $i$.

**Absorption probabilities:** $B = NR$ gives the probability of being absorbed into each absorbing state. The gambler's ruin formula $P(\text{reach } N \mid \text{start at } a) = a/N$ (derived via OST in §06) follows directly from this formula.

### 3.5 Ergodic Chains

**Definition (Ergodic Chain).** A Markov chain is **ergodic** if it is:
1. **Irreducible** — every state communicates with every other
2. **Positive recurrent** — mean return time is finite for all states
3. **Aperiodic** — all states have period 1

Ergodic chains are the "nice" chains: they have a unique stationary distribution, and the chain converges to it from any starting state.

**For finite chains:** An irreducible chain on a finite state space is automatically positive recurrent. So for finite chains, ergodic = irreducible + aperiodic.

**Ergodic theorem:** For an ergodic chain, the time-average converges to the space-average:
$$\frac{1}{n}\sum_{k=0}^{n-1} f(X_k) \xrightarrow{n\to\infty} \sum_i \pi_i f(i) = \mathbb{E}_\pi[f] \quad \text{a.s.}$$

This is the Markov chain version of the law of large numbers. It is why MCMC works: time-averaging over the chain's trajectory approximates the posterior expectation.

---

## 4. Stationary Distributions

### 4.1 Definition and Existence

**Definition (Stationary Distribution).** A probability distribution $\pi$ on $\mathcal{S}$ is a **stationary distribution** (also called **invariant distribution** or **steady-state distribution**) for a Markov chain with transition matrix $P$ if:
$$\pi P = \pi \qquad \text{i.e.,} \qquad \pi_j = \sum_i \pi_i P_{ij} \text{ for all } j$$

Equivalently, $\pi$ is a left eigenvector of $P$ with eigenvalue 1. If the chain starts in distribution $\pi$, it stays in distribution $\pi$ for all time: $\mu^{(0)} = \pi \Rightarrow \mu^{(n)} = \pi$ for all $n$.

**Interpretation:** $\pi_i$ represents the long-run fraction of time the chain spends in state $i$. For PageRank, $\pi_i$ is the importance score of page $i$. For MCMC, $\pi$ is the target posterior distribution.

**Existence:** Every finite irreducible Markov chain has a unique stationary distribution $\pi$ with $\pi_i > 0$ for all $i$. For infinite chains, existence requires positive recurrence.

**Non-uniqueness when reducible:** A chain with multiple closed communicating classes has infinitely many stationary distributions — any convex combination of the stationary distributions within each closed class is stationary.

**The mean return time formula:** For a positive recurrent chain, $\pi_i = 1/\mu_i$ where $\mu_i = \mathbb{E}[T_i \mid X_0=i]$ is the mean return time. States visited more often (smaller $\mu_i$) get higher stationary probability.

### 4.2 Perron-Frobenius Theorem

The Perron-Frobenius theorem is the fundamental result guaranteeing existence and uniqueness of the stationary distribution.

**Theorem (Perron-Frobenius for Stochastic Matrices).** Let $P$ be an irreducible stochastic matrix. Then:
1. $\lambda = 1$ is an eigenvalue of $P$ (always true for stochastic matrices, since $P\mathbf{1}=\mathbf{1}$)
2. $|\lambda| \leq 1$ for all other eigenvalues $\lambda$
3. There exists a unique probability vector $\pi > 0$ (componentwise) with $\pi P = \pi$
4. $\pi$ is the unique (up to scaling) left eigenvector with eigenvalue 1

If additionally $P$ is **primitive** (irreducible and aperiodic), then $\lambda = 1$ is the **unique** eigenvalue on the unit circle, and $|\lambda_2| < 1$ strictly. This gap drives convergence.

**Proof sketch (uniqueness):** Suppose $\pi$ and $\nu$ are both stationary. Then $\mu = (\pi - \nu)$ satisfies $\mu P = \mu$ with $\sum_i \mu_i = 0$. If any $\mu_i \neq 0$, irreducibility implies all $\mu_i \neq 0$. But then $\pi - \nu$ cannot be a probability vector minus a probability vector with the right sign everywhere — contradiction.

**For AI:** PageRank power iteration converges because the damped PageRank matrix is primitive (the teleportation term makes it irreducible and the self-loops make it aperiodic). The dominant eigenvalue is 1, and all other eigenvalues are $< \alpha < 1$ in magnitude, guaranteeing geometric convergence.

### 4.3 Convergence to Stationarity

**Theorem (Convergence for Ergodic Chains).** Let $P$ be irreducible and aperiodic with stationary distribution $\pi$. Then for any initial distribution $\mu^{(0)}$:
$$\|\mu^{(0)} P^n - \pi\|_{\text{TV}} \xrightarrow{n\to\infty} 0$$

Moreover, for any starting state $i$: $P^{(n)}_{ij} \to \pi_j$ as $n \to \infty$ for all $j$.

**Rate of convergence:** For a reversible chain with eigenvalues $1 = \lambda_1 > \lambda_2 \geq \cdots \geq \lambda_N \geq -1$, the convergence rate is governed by $\lambda_* = \max(|\lambda_2|, |\lambda_N|)$:
$$\|P^n(i,\cdot) - \pi\|_{\text{TV}} \leq \frac{1}{2\sqrt{\pi_i}} \lambda_*^n$$

The rate $\lambda_*$ is called the **second largest eigenvalue modulus (SLEM)**. The **spectral gap** is $1 - \lambda_*$; a large gap means fast convergence.

**Periodic chains:** If $P$ is irreducible but periodic with period $d$, then $P^{(n)}_{ij}$ does not converge — it cycles. However, $P^d$ (the $d$-step chain) is aperiodic and does converge. In practice, a small perturbation (adding laziness) kills periodicity.

**The lazy chain:** Replace $P$ with $(I + P)/2$. This halves the spectral gap but makes the chain aperiodic with all eigenvalues non-negative, ensuring monotone convergence.

### 4.4 Null-Recurrent Chains

The simple random walk on $\mathbb{Z}$ returns to the origin with probability 1 (recurrent) but takes infinitely long on average (null-recurrent). Such chains have no stationary probability distribution — the "stationary measure" $\pi_i$ is not normalisable.

The 1D random walk has $\pi_i = 1$ for all $i$, which gives infinite total mass $\sum_i \pi_i = \infty$. The chain visits every state infinitely often, but spends a vanishing fraction of time at any fixed state.

**Implications for AI:** Infinite-state chains that model text generation (treating the entire prefix as state) may be null-recurrent or transient. In practice, language models cap context length, making the state space finite and all states positive recurrent.

### 4.5 Computing Stationary Distributions

**Method 1: Solve the linear system**
$$\pi P = \pi, \quad \|\pi\|_1 = 1, \quad \pi_i \geq 0$$

This is an $(N+1)$-dimensional system (the normalisation replaces one of the $N$ redundant equations $\pi P = \pi$). For small $N$, direct linear algebra is exact.

**Method 2: Power iteration**
$$\pi^{(k+1)} = \pi^{(k)} P$$

Starting from any distribution $\pi^{(0)} > 0$, iterating converges to $\pi$ at rate $\lambda_*^k$. Power iteration is the PageRank algorithm.

**Method 3: Eigendecomposition**
Compute the left eigenvector of $P$ corresponding to eigenvalue 1. For symmetric or reversible chains, this is equivalent to finding the dominant eigenvector.

**Method 4: MCMC (for continuous state spaces)**
For distributions $\pi(x) \propto \exp(-f(x))$ on $\mathbb{R}^d$, construct a Markov chain with stationary distribution $\pi$ and run it — the empirical distribution of the chain approximates $\pi$ by the ergodic theorem.

---

## 5. Detailed Balance and Reversibility

### 5.1 Detailed Balance Equations

**Definition (Detailed Balance).** A distribution $\pi$ satisfies the **detailed balance equations** with respect to transition matrix $P$ if:
$$\pi_i P_{ij} = \pi_j P_{ji} \quad \text{for all } i, j \in \mathcal{S}$$

Detailed balance says: the probability flux from $i$ to $j$ at stationarity equals the flux from $j$ to $i$. This is a pointwise balance of probability flows, stronger than the global balance equations $\pi P = \pi$.

**Theorem.** If $\pi$ satisfies detailed balance with $P$, then $\pi$ is a stationary distribution for $P$.

*Proof:*
$$\sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} = \pi_j \sum_i P_{ji} = \pi_j \cdot 1 = \pi_j$$

So $\pi P = \pi$. The converse is false — a stationary distribution need not satisfy detailed balance.

**Significance:** Detailed balance is easier to verify than stationarity (it's a pairwise condition rather than a global one), and it characterises **reversible** chains. All Metropolis-Hastings algorithms are constructed to satisfy detailed balance by design.

### 5.2 Reversible Markov Chains

**Definition (Reversible Chain).** A Markov chain is **reversible** (or **time-reversible**) with respect to distribution $\pi$ if it satisfies detailed balance. Equivalently, the chain running forwards in time has the same law as the chain running backwards.

**The reversed chain:** Given a stationary Markov chain with transition matrix $P$, define the **time-reversed chain** by $\hat{P}_{ij} = \pi_j P_{ji} / \pi_i$. The reversed chain is also a Markov chain, with the same stationary distribution. The original chain is reversible iff $\hat{P} = P$.

**Symmetric Markov chains are reversible:** If $P_{ij} = P_{ji}$ for all $i,j$ (symmetric transition matrix), then detailed balance holds with the uniform distribution $\pi_i = 1/N$. The symmetric random walk on a graph is reversible.

**Non-reversible example:** The chain $1 \to 2 \to 3 \to 1$ (cyclic with $P_{12}=P_{23}=P_{31}=1$) has uniform stationary distribution but is NOT reversible: it has a preferred direction of rotation.

**For AI:** Reversibility is a sufficient condition for correctness of MCMC algorithms. A Metropolis-Hastings chain with target $\pi$ satisfies detailed balance — it's reversible — and therefore has $\pi$ as stationary distribution. Non-reversible MCMC (e.g., lifting methods, HMC) can mix faster but is harder to analyse.

### 5.3 Birth-Death Chains

A **birth-death chain** is a Markov chain on $\{0,1,2,\ldots\}$ where only nearest-neighbour transitions are allowed:
$$P_{i,i+1} = p_i, \quad P_{i,i-1} = q_i = 1-p_i, \quad P_{ij} = 0 \text{ for } |i-j| > 1$$

Birth-death chains are automatically reversible, and their stationary distribution can be computed explicitly. The detailed balance equations $\pi_i p_i = \pi_{i+1} q_{i+1}$ give:
$$\pi_n = \pi_0 \prod_{k=0}^{n-1} \frac{p_k}{q_{k+1}}$$

Setting $\pi_0 = 1/Z$ (normalisation) gives the stationary distribution.

**Queue models:** An $M/M/1$ queue with arrival rate $\lambda < 1$ (service rate) is a birth-death chain. Stationary distribution: $\pi_n = (1-\rho)\rho^n$ where $\rho = \lambda$. The chain is stable (stationary distribution exists) iff $\lambda < 1$.

**For AI:** The queue length at an LLM serving system (number of pending requests) is approximately a birth-death chain. The stationary distribution under load $\rho = \lambda/\mu$ determines average latency: $\mathbb{E}[\text{queue length}] = \rho/(1-\rho)$.

### 5.4 Spectral Decomposition of Reversible Chains

For a reversible chain, the transition matrix $P$ has real eigenvalues $1 = \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_N \geq -1$ and an orthonormal basis of eigenvectors (with respect to the $\pi$-weighted inner product).

**Spectral decomposition:**
$$P^n_{ij} = \pi_j \sum_{k=1}^N \lambda_k^n \phi_k(i) \phi_k(j)$$

where $\phi_k$ are the eigenfunctions (normalised in $L^2(\pi)$). As $n \to \infty$, only the $k=1$ term survives (since $|\lambda_k| < 1$ for $k \geq 2$), giving $P^n_{ij} \to \pi_j$. The rate of convergence is $|\lambda_2|^n$.

**The spectral gap:** $\text{gap} = 1 - \lambda_2$ measures how fast the chain forgets its initial state. A chain with spectral gap $\delta$ mixes in $O(1/\delta)$ steps. Bounding the spectral gap from below is the main technical challenge in proving MCMC convergence rates.

**Dirichlet form:** For reversible chains, the spectral gap equals:
$$\text{gap} = \min_{f \perp \mathbf{1}} \frac{\mathcal{E}(f,f)}{\text{Var}_\pi(f)}$$

where $\mathcal{E}(f,f) = \frac{1}{2}\sum_{i,j} \pi_i P_{ij}(f(j)-f(i))^2$ is the Dirichlet form. This variational characterisation is the basis for Poincaré inequality methods.

---

## 6. Mixing Times and Convergence Rates

### 6.1 Total Variation Distance

**Definition.** The **total variation (TV) distance** between two probability distributions $\mu$ and $\nu$ on $\mathcal{S}$ is:
$$\|\mu - \nu\|_{\text{TV}} = \sup_{A \subseteq \mathcal{S}} |\mu(A) - \nu(A)| = \frac{1}{2}\sum_i |\mu_i - \nu_i|$$

TV distance measures how distinguishable two distributions are: it equals the maximum probability that any event $A$ has different probability under $\mu$ vs. $\nu$. The TV distance is 0 iff $\mu = \nu$, and 1 iff the distributions have disjoint support.

**Coupling characterisation:** $\|\mu - \nu\|_{\text{TV}} = \inf_{(X,Y)} P(X \neq Y)$ where the infimum is over all couplings $(X,Y)$ with marginals $\mu$ and $\nu$. This is the **coupling inequality**: any coupling gives an upper bound on TV distance.

**For AI:** TV distance is used to measure how far an MCMC chain is from stationarity, and to certify when the chain has "mixed" sufficiently to produce approximately independent samples from the posterior.

### 6.2 Mixing Time

**Definition.** The **mixing time** of a Markov chain with stationary distribution $\pi$ is:
$$t_{\text{mix}}(\varepsilon) = \min\left\{t \geq 0 : \max_{x \in \mathcal{S}} \|P^t(x,\cdot) - \pi\|_{\text{TV}} \leq \varepsilon\right\}$$

The standard mixing time is $t_{\text{mix}} = t_{\text{mix}}(1/4)$ (the default $\varepsilon = 1/4$ is conventional). Mixing time measures how long the chain must run before its distribution is $\varepsilon$-close to stationarity from the worst-case starting state.

**Why the worst case?** In applications, we often don't control the starting state. The worst-case mixing time certifies that regardless of initialisation, the chain has converged after $t_{\text{mix}}$ steps.

**Submultiplicativity:** $t_{\text{mix}}(2^{-k}/4) \leq k \cdot t_{\text{mix}}$. So to achieve $\varepsilon = 2^{-k}/4$ precision, it suffices to run for $k \cdot t_{\text{mix}}$ steps.

### 6.3 Spectral Gap

For a reversible ergodic chain, the spectral gap $\text{gap} = 1 - \lambda_2$ provides a bound on mixing time:

**Theorem (Spectral Gap Bound):**
$$t_{\text{mix}}(\varepsilon) \leq \left\lceil \frac{\log(1/(\varepsilon \pi_{\min}))}{\text{gap}} \right\rceil$$

where $\pi_{\min} = \min_i \pi_i$.

**Proof sketch:** From the spectral decomposition, $\|P^n(x,\cdot) - \pi\|_{\text{TV}} \leq \frac{1}{2\sqrt{\pi_x}} (1-\text{gap})^n$. Setting this $\leq \varepsilon$ and solving for $n$ gives the bound.

**Lower bound:** There is also a lower bound: $t_{\text{mix}} \geq \frac{1}{2\,\text{gap}}$. So the mixing time is $\Theta(1/\text{gap})$ for reversible chains.

**Computing the spectral gap:** For small matrices, compute eigenvalues of $P$ directly. For large chains, use the Cheeger inequality: $\text{gap}/2 \leq h \leq \sqrt{2\,\text{gap}}$ where $h$ is the Cheeger constant (conductance of the chain).

### 6.4 Coupling Arguments

**Definition (Coupling).** A **coupling** of two Markov chains with the same transition matrix $P$ but different initial distributions $\mu, \nu$ is a process $(X_t, Y_t)$ such that each marginal is a valid Markov chain with transition $P$, and once $X_t = Y_t$ they remain together (the **coalescence** or **meeting** time).

**Coupling inequality:** $\|P^t(\mu,\cdot) - P^t(\nu,\cdot)\|_{\text{TV}} \leq P(X_t \neq Y_t) \leq P(\tau_{\text{meet}} > t)$

where $\tau_{\text{meet}} = \min\{t : X_t = Y_t\}$ is the coupling time.

**Grand coupling:** For a finite ergodic chain, one can construct a single coupling $\{(X_t^x, X_t^y) : x, y \in \mathcal{S}\}$ of chains starting from all possible pairs $(x,y)$ simultaneously, such that $\tau_{\text{meet}}$ is the same for all pairs. The mixing time is then $t_{\text{mix}} = O(\mathbb{E}[\tau_{\text{meet}}])$.

**Example: Random walk on hypercube.** For a lazy random walk on $\{0,1\}^d$, couple two chains by updating the same coordinate at each step. When the coordinate is chosen, the two chains agree on that bit after the update. All $d$ bits agree after $O(d \log d)$ steps (coupon collector), giving $t_{\text{mix}} = O(d \log d)$.

### 6.5 AI Relevance: Fast Mixing and MCMC Warm Starts

**Fast mixing in practice:** For Bayesian inference in LLM fine-tuning (e.g., RLHF reward models), the posterior over model weights is a distribution on $\mathbb{R}^d$ with $d \sim 10^9$. Direct MCMC is intractable; approximate methods (Langevin dynamics, HMC) exploit gradient information to achieve better mixing.

**Spectral gap and dimensionality:** For a Gaussian target $\mathcal{N}(0, \Sigma)$, the Langevin SDE mixing time scales as $O(\kappa)$ where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number of $\Sigma$. Preconditioning (e.g., SGLD with adaptive learning rates, Adam-Langevin) improves the effective spectral gap.

**Warm starts:** If the initial distribution $\mu^{(0)}$ is already close to $\pi$ (e.g., from a previous MCMC run), the chain needs fewer steps to mix. In sequential Bayesian updating (fine-tuning an LLM on new data), the previous posterior serves as a warm start.

**Temperature scheduling:** Simulated annealing starts with a high-temperature chain (wide, fast-mixing distribution) and gradually cools toward the target. This exploits fast mixing at high temperature and accurate targeting at low temperature.

---

## 7. Continuous-Time Markov Chains

### 7.1 Generator Matrix

A **continuous-time Markov chain (CTMC)** $\{X_t\}_{t \geq 0}$ on a finite state space $\mathcal{S}$ satisfies the Markov property for continuous time:
$$P(X_{t+s}=j \mid X_u : u \leq s) = P(X_{t+s}=j \mid X_s)$$

The chain holds in each state for an exponentially distributed sojourn time, then jumps according to a discrete-time chain. The rate of leaving state $i$ is $q_i = \sum_{j \neq i} q_{ij}$ where $q_{ij} \geq 0$ are the **jump rates** from $i$ to $j$.

**Definition (Generator Matrix / Q-Matrix).** The **generator** (or **Q-matrix**) of a CTMC is the matrix $Q$ with:
$$Q_{ij} = \begin{cases} q_{ij} \geq 0 & i \neq j \quad \text{(jump rate from } i \text{ to } j\text{)} \\ -q_i = -\sum_{j \neq i} q_{ij} & i = j \quad \text{(holding rate)} \end{cases}$$

Key property: rows of $Q$ sum to zero ($Q \mathbf{1} = \mathbf{0}$). The diagonal entries are negative; off-diagonal entries are non-negative.

**Transition matrix:** The probability of being in state $j$ at time $t$ given state $i$ at time 0 is:
$$P(t) = e^{Qt} = \sum_{n=0}^\infty \frac{(Qt)^n}{n!}$$

This is the **matrix exponential**. Note $P(0) = I$ and $\lim_{t\to\infty} P(t) = \mathbf{1}\pi$ (convergence to stationarity for ergodic chains).

**Embedded chain:** The CTMC can be decomposed into (1) an embedded discrete-time Markov chain governing the sequence of states visited, and (2) exponential holding times in each state. Jump probabilities: $\tilde{P}_{ij} = q_{ij}/q_i$ for $i \neq j$.

### 7.2 Kolmogorov Equations

**Forward Equation (Fokker-Planck):**
$$\frac{d}{dt} P(t) = P(t) Q \quad \Rightarrow \quad P(t) = e^{Qt}$$

This equation describes how the distribution evolves forward in time: $\frac{d}{dt}\mu(t) = \mu(t) Q$ where $\mu(t) = \mu(0) e^{Qt}$.

**Backward Equation:**
$$\frac{d}{dt} P(t) = Q\, P(t)$$

The forward equation applies to the time-marginal distribution; the backward equation to the transition probabilities viewed as functions of the initial state.

**Stationary distribution:** $\pi$ is stationary if $\frac{d}{dt}\mu(t) = 0$ at $\mu = \pi$, i.e.:
$$\pi Q = \mathbf{0} \quad \text{(row vector)} \quad \Leftrightarrow \quad \sum_i \pi_i Q_{ij} = 0 \text{ for all } j$$

For an ergodic CTMC, the stationary distribution is unique.

### 7.3 Stationary Distribution of CTMCs

**Detailed balance for CTMCs:** A distribution $\pi$ satisfies detailed balance if $\pi_i Q_{ij} = \pi_j Q_{ji}$ for all $i \neq j$. Detailed balance implies stationarity ($\pi Q = 0$).

**Computing the stationary distribution:** Solve $\pi Q = 0$ with $\sum_i \pi_i = 1$. This is equivalent to finding the null space of $Q^T$ (right eigenvector with eigenvalue 0).

**Birth-death CTMC:** With birth rate $\lambda_n$ from state $n$ and death rate $\mu_n$, detailed balance gives $\pi_n \lambda_n = \pi_{n+1}\mu_{n+1}$, yielding:
$$\pi_n = \pi_0 \prod_{k=0}^{n-1} \frac{\lambda_k}{\mu_{k+1}}$$

### 7.4 Poisson Process as CTMC

> **Recall from [§06 Stochastic Processes](../06-Stochastic-Processes/notes.md#5-poisson-processes):** The Poisson process $N_t$ counts arrivals up to time $t$ with independent Poisson increments. Inter-arrival times are $\text{Exp}(\lambda)$.

The Poisson process is the simplest CTMC: states $\{0,1,2,\ldots\}$, jump rates $q_{n,n+1} = \lambda$ (arrivals only), generator:
$$Q = \begin{pmatrix} -\lambda & \lambda & & \\ & -\lambda & \lambda & \\ & & \ddots & \ddots \end{pmatrix}$$

The Poisson process is **transient** — it drifts to $+\infty$ with no stationary distribution. An $M/M/1$ queue (arrivals at rate $\lambda$, service at rate $\mu > \lambda$) modifies the Poisson process by adding downward jumps (service completions), yielding a birth-death CTMC with geometric stationary distribution $\pi_n = (1-\rho)\rho^n$ where $\rho = \lambda/\mu$.

**For AI:** The Poisson process models token arrivals at an LLM serving endpoint. At high load ($\rho \to 1$), the queue length distribution becomes heavy-tailed, leading to high-percentile latency spikes. The M/M/1 formula $\mathbb{E}[\text{queue}] = \rho/(1-\rho)$ quantifies this: 90% utilisation means 9× the base queue length.

---

## 8. MCMC: Markov Chain Monte Carlo

### 8.1 The Sampling Problem

**The challenge:** In Bayesian inference, the posterior $\pi(\theta \mid \mathcal{D}) \propto p(\mathcal{D}\mid\theta)\,p(\theta)$ is analytically intractable for most models. Computing $\mathbb{E}_\pi[f(\theta)]$ requires integrating over a high-dimensional distribution we cannot normalise.

**MCMC solution:** Construct a Markov chain $\{\theta_t\}$ on the parameter space such that:
1. The chain is ergodic (irreducible, aperiodic, positive recurrent)
2. The stationary distribution equals the target $\pi$

By the ergodic theorem, time-averaging gives posterior expectations:
$$\frac{1}{T}\sum_{t=1}^T f(\theta_t) \xrightarrow{T\to\infty} \mathbb{E}_\pi[f(\theta)]$$

The key insight: we only need to know $\pi(\theta)$ up to a normalising constant. MCMC constructs the chain using only unnormalised evaluations $\tilde{\pi}(\theta) \propto \pi(\theta)$.

### 8.2 Metropolis-Hastings

**Algorithm (Metropolis-Hastings).** Given current state $\theta$, target $\pi$, proposal $q(\cdot \mid \theta)$:

1. Sample candidate $\theta' \sim q(\cdot \mid \theta)$
2. Compute acceptance ratio: $a = \min\left(1,\ \frac{\pi(\theta')\, q(\theta \mid \theta')}{\pi(\theta)\, q(\theta' \mid \theta)}\right)$
3. With probability $a$: accept $\theta_{t+1} = \theta'$; otherwise: $\theta_{t+1} = \theta$ (reject and stay)

**Correctness:** The Metropolis-Hastings chain satisfies detailed balance with respect to $\pi$:
$$\pi(\theta) q(\theta' \mid \theta) a(\theta, \theta') = \pi(\theta') q(\theta \mid \theta') a(\theta', \theta)$$

*Proof:* Without loss of generality, assume $\pi(\theta')q(\theta\mid\theta') \leq \pi(\theta)q(\theta'\mid\theta)$. Then $a(\theta,\theta')=\frac{\pi(\theta')q(\theta|\theta')}{\pi(\theta)q(\theta'|\theta)}$ and $a(\theta',\theta)=1$, so both sides equal $\pi(\theta')q(\theta|\theta')$.

**Special cases:**
- **Metropolis algorithm:** Symmetric proposal $q(\theta'\mid\theta) = q(\theta\mid\theta')$ simplifies acceptance to $a = \min(1, \pi(\theta')/\pi(\theta))$
- **Independence sampler:** Proposal $q(\theta'\mid\theta) = q(\theta')$ independent of current state; acceptance $a = \min(1, \pi(\theta')q(\theta)/(\pi(\theta)q(\theta')))$

**Proposal tuning:** For a Gaussian random walk proposal $q(\theta'\mid\theta) = \mathcal{N}(\theta, \sigma^2 I)$, the optimal $\sigma$ achieves acceptance rate ~23.4% in high dimensions (Roberts-Gelman-Gilks 1997). Too small $\sigma$: high acceptance but slow exploration; too large: low acceptance, inefficient.

**For AI:** MCMC is used for posterior inference in Bayesian neural networks, uncertainty quantification in LLM outputs, and sampling from energy-based models. In practice, Langevin dynamics (gradient-based MCMC) replaces vanilla MH for high-dimensional targets.

### 8.3 Gibbs Sampling

**Algorithm (Gibbs Sampling).** For target $\pi(\theta_1,\ldots,\theta_d)$, cycle through coordinates:
1. Sample $\theta_1^{(t+1)} \sim \pi(\theta_1 \mid \theta_2^{(t)}, \ldots, \theta_d^{(t)})$
2. Sample $\theta_2^{(t+1)} \sim \pi(\theta_2 \mid \theta_1^{(t+1)}, \theta_3^{(t)}, \ldots, \theta_d^{(t)})$
3. $\vdots$
4. Sample $\theta_d^{(t+1)} \sim \pi(\theta_d \mid \theta_1^{(t+1)}, \ldots, \theta_{d-1}^{(t+1)})$

**Correctness:** Gibbs sampling is a special case of Metropolis-Hastings where each coordinate update is a MH step with proposal equal to the full conditional. The acceptance ratio is always 1 — every proposal is accepted. This requires the ability to sample from full conditionals $\pi(\theta_k \mid \theta_{-k})$.

**Block Gibbs:** Instead of updating one coordinate at a time, update a block of coordinates jointly. More efficient when coordinates within a block are highly correlated.

**For AI:** Gibbs sampling is used in latent Dirichlet allocation (LDA) for topic modelling — the full conditionals for topic assignments are available in closed form. In RL, it can be used to sample from the Q-function distribution for exploration.

### 8.4 Convergence Diagnostics

Running an MCMC chain for a finite number of steps gives an approximate (not exact) sample from $\pi$. Practical convergence diagnostics:

**Trace plots:** Plot $\theta_t$ vs. $t$. A well-mixed chain looks like stationary noise. Trends, drifts, or stuck values indicate non-convergence.

**$\hat{R}$ statistic (Gelman-Rubin):** Run $M \geq 4$ chains from diverse starting points. Compute $\hat{R} = \sqrt{V_{\text{between}}/V_{\text{within}}}$. If $\hat{R} \leq 1.01$, chains have converged to the same distribution.

**Effective sample size (ESS):** Due to autocorrelation within the chain, $T$ MCMC samples give fewer independent samples than $T$ iid samples. $\text{ESS} = T / (1 + 2\sum_{k=1}^\infty \rho_k)$ where $\rho_k$ is the lag-$k$ autocorrelation. $\text{ESS}/T \ll 1$ indicates poor mixing.

**Burn-in:** The first $B$ samples (before the chain has mixed) are discarded. Typical burn-in is $10-50\%$ of total samples.

### 8.5 Hamiltonian Monte Carlo

**Motivation:** For distributions on $\mathbb{R}^d$, the random walk Metropolis-Hastings has step size $O(d^{-1/2})$ to achieve acceptable acceptance rates, giving diffusive mixing: $O(d)$ steps to traverse the distribution. Hamiltonian Monte Carlo (HMC) exploits gradient information to take large, high-acceptance steps.

**Algorithm:** Augment $\theta \in \mathbb{R}^d$ with momentum $p \sim \mathcal{N}(0, M)$. The Hamiltonian is $H(\theta,p) = -\log\pi(\theta) + \frac{1}{2}p^T M^{-1} p$. Propose a new state by simulating Hamiltonian dynamics using the leapfrog integrator for $L$ steps with step size $\varepsilon$, then accept/reject with MH ratio $e^{-\Delta H}$.

**Why it works:** Hamiltonian dynamics preserves the joint distribution $e^{-H(\theta,p)}$ on $(\theta,p)$. Marginalising out $p$ recovers $\pi(\theta)$. The leapfrog integrator introduces discretisation error, corrected by the MH accept/reject step.

**Mixing time:** HMC achieves $O(d^{1/4})$ gradient evaluations per independent sample (vs. $O(d)$ for random walk), making it scalable to high dimensions. NUTS (No-U-Turn Sampler) automates the choice of $L$ and $\varepsilon$.

**For AI:** HMC is the backbone of probabilistic programming systems like Stan and PyMC. In Bayesian deep learning, SGLD (stochastic gradient Langevin dynamics) adds Gaussian noise to SGD steps, yielding approximate Bayesian inference at scale.

---

## 9. ML Deep Dive

### 9.1 Language Model Generation

Every autoregressive language model — GPT, LLaMA, PaLM, Gemini — generates text as a Markov chain on the vocabulary. The state at step $n$ is the entire generated sequence $(w_1,\ldots,w_n)$ (treated as a single token in the KV cache), and the transition is:
$$P(w_{n+1} = v \mid w_1,\ldots,w_n) = \text{softmax}(\text{LM}(w_1,\ldots,w_n))_v$$

**Temperature and the stationary distribution:** At temperature $\tau$:
$$P_\tau(w_{n+1} = v \mid \text{context}) = \frac{\exp(\text{logit}_v / \tau)}{\sum_{v'} \exp(\text{logit}_{v'}/\tau)}$$

- $\tau \to 0$: greedy decoding (deterministic); $\tau = 1$: standard sampling; $\tau \to \infty$: uniform
- The generated sequence is a Markov chain; its stationary distribution (if the chain were infinite) would represent the model's "preferred" discourse

**Mixing time of language models:** For a well-trained LLM, the Markov chain mixes quickly — the model "forgets" its starting state within a few hundred tokens for most topics. This reflects the inductive bias toward coherent text: the chain has a large spectral gap.

**Top-$k$ and nucleus sampling:** Top-$k$ sampling restricts transitions to the $k$ highest-probability tokens; nucleus (top-$p$) sampling restricts to the minimal set of tokens whose cumulative probability exceeds $p$. Both modify the transition matrix to improve mixing (reduce probability of stuck low-quality loops).

**Repetition and periodicity:** Without repetition penalties, language model chains can enter periodic loops (mode collapse in generation: "the the the ..."). Adding a repetition penalty introduces a state-dependent transition modification that destroys the periodic structure.

### 9.2 PageRank

**Setup:** Represent the web as a directed graph with $N$ pages. The random surfer model: a user follows a random link at each step with probability $\alpha$ (damping factor), or teleports to a random page with probability $1-\alpha$.

**Transition matrix:**
$$P_{ij} = \alpha \cdot \frac{A_{ij}}{d_i} + (1-\alpha) \cdot \frac{1}{N}$$

where $A_{ij} = 1$ if page $i$ links to page $j$, and $d_i = \sum_j A_{ij}$ is the out-degree of page $i$.

**Dangling nodes:** Pages with no outbound links ($d_i = 0$) create a stochastic matrix problem. Fix: replace the row for dangling node $i$ with the uniform distribution $1/N$. The resulting matrix is stochastic.

**PageRank computation:** The PageRank vector $\pi$ satisfies $\pi P = \pi$. Computed by power iteration:
$$\pi^{(k+1)} = \pi^{(k)} P$$

Convergence guaranteed by Perron-Frobenius (the damped matrix is primitive). Convergence rate: $\alpha^k$ geometric (since the second eigenvalue is $\leq \alpha$). With $\alpha = 0.85$, convergence to $10^{-8}$ precision takes $\approx 100$ iterations.

**Interpretation:** $\pi_i$ is the long-run fraction of time the random surfer spends on page $i$. Pages with high in-degree from other high-PageRank pages get high scores.

**For AI:** Influence in a social network, citations in academic papers, and token importance in attention patterns can all be modelled as PageRank variants. The attention mechanism in transformers can be viewed as a differentiable, query-dependent version of PageRank.

### 9.3 Markov Decision Processes

A **Markov Decision Process (MDP)** extends Markov chains by adding actions:
$$(\mathcal{S}, \mathcal{A}, P, r, \gamma)$$

- $\mathcal{S}$: state space; $\mathcal{A}$: action space
- $P(s' \mid s, a)$: transition probabilities given action $a$
- $r(s, a)$: reward function
- $\gamma \in [0,1)$: discount factor

A **policy** $\pi(a \mid s)$ induces a Markov chain on $\mathcal{S}$ with transition $P^\pi(s'\mid s) = \sum_a \pi(a\mid s) P(s'\mid s,a)$.

**Bellman equation:** The value function $V^\pi(s) = \mathbb{E}^\pi[\sum_{t=0}^\infty \gamma^t r(s_t,a_t) \mid s_0=s]$ satisfies:
$$V^\pi(s) = \sum_a \pi(a\mid s)\left[r(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V^\pi(s')\right]$$

In matrix form: $\mathbf{V}^\pi = \mathbf{r}^\pi + \gamma P^\pi \mathbf{V}^\pi$, giving $\mathbf{V}^\pi = (I - \gamma P^\pi)^{-1}\mathbf{r}^\pi$.

**Policy evaluation = linear system:** Given a fixed policy $\pi$, computing $V^\pi$ requires inverting $(I - \gamma P^\pi)$. This is guaranteed well-conditioned since $\gamma < 1$ implies all eigenvalues of $\gamma P^\pi$ have magnitude $< 1$.

**RLHF as MDP:** Reinforcement learning from human feedback trains a language model (policy $\pi_\theta$) to maximise a reward model $r_\phi$ — an MDP with the KV cache as state, vocabulary as actions, and the human preference model as reward.

### 9.4 Hidden Markov Models

A **Hidden Markov Model (HMM)** is a Markov chain $\{Z_t\}$ (hidden states) with observations $\{X_t\}$ emitted from the current hidden state:
$$Z_0 \sim \mu_0, \quad Z_t \mid Z_{t-1} \sim P_{Z_{t-1}}, \quad X_t \mid Z_t \sim E_{Z_t}$$

where $P$ is the transition matrix and $E$ is the emission distribution.

**Forward algorithm:** Compute $\alpha_t(i) = P(X_1,\ldots,X_t, Z_t=i)$ recursively:
$$\alpha_t(j) = E_j(X_t) \sum_i \alpha_{t-1}(i) P_{ij}, \quad \alpha_0(j) = \mu_0(j) E_j(X_1)$$

Time complexity: $O(T \cdot |\mathcal{S}|^2)$. Used for likelihood computation and decoding.

**Viterbi algorithm:** Find the most likely hidden state sequence $\hat{Z}_{1:T} = \arg\max_{z_{1:T}} P(z_{1:T} \mid X_{1:T})$ via dynamic programming:
$$\delta_t(j) = E_j(X_t) \max_i \delta_{t-1}(i) P_{ij}$$

Time complexity: same as forward algorithm.

**Baum-Welch (EM for HMMs):** Learn $P, E, \mu_0$ from unlabelled observations. E-step: compute forward-backward probabilities. M-step: update parameters using expected sufficient statistics. Converges to a local maximum of the likelihood.

**For AI:** HMMs are classical sequence models (speech recognition, gene finding). Modern LLMs subsume HMMs: the hidden state (KV cache) is a rich continuous representation of the context, and emission (next-token distribution) is the softmax output.

### 9.5 Diffusion Models as CTMCs

The DDPM forward process can be viewed as a discrete-time Markov chain with Gaussian transitions:
$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I)$$

The marginal $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$ with $\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$.

> **Recall from [§06 Stochastic Processes](../06-Stochastic-Processes/notes.md#6-brownian-motion):** In the SDE formulation (Song et al., 2021), the continuous-time version is a CTMC/SDE: $dx = f(x,t)\,dt + g(t)\,dB_t$.

**The reverse process:** By time-reversal of Markov chains (Anderson, 1982), the reverse of an ergodic CTMC is also a CTMC. The reverse diffusion has:
$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(t))$$

The neural network learns to parameterise the reverse transitions — equivalently, to approximate the score function $\nabla_x \log q_t(x)$.

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Confusing stationarity with convergence | Stationarity ($\pi P = \pi$) is a property of a distribution; convergence ($\mu^{(n)} \to \pi$) is about the chain's trajectory. The chain may start at $\pi$ and be stationary without "converging" (it's already there). | Distinguish: stationarity is the destination; convergence is the journey. |
| 2 | Assuming irreducibility implies unique stationary distribution | An irreducible chain on an infinite state space may be null-recurrent (no normalisable stationary distribution), e.g., SRW on $\mathbb{Z}$. | Unique stationary distribution requires irreducibility + positive recurrence. |
| 3 | Forgetting aperiodicity for convergence | An irreducible positive-recurrent but periodic chain has a unique stationary distribution but $P^{(n)}_{ij}$ does NOT converge — it oscillates. | Convergence to stationarity requires ergodicity = irreducible + positive recurrent + aperiodic. |
| 4 | Treating detailed balance as necessary for stationarity | Detailed balance ($\pi_i P_{ij} = \pi_j P_{ji}$) is sufficient but not necessary for $\pi P = \pi$. Many chains (e.g., cyclic chains) have stationary distributions without satisfying detailed balance. | Detailed balance $\Rightarrow$ stationarity, but stationarity $\not\Rightarrow$ detailed balance. |
| 5 | Confusing the transition matrix orientation | Some books write $P_{ij}$ as probability from $j$ to $i$ (column stochastic); others write it as probability from $i$ to $j$ (row stochastic). This flips $\pi P = \pi$ vs. $P\pi = \pi$. | Fix a convention: row stochastic means rows sum to 1 and $\pi P = \pi$ (left eigenvector). |
| 6 | Ignoring burn-in in MCMC | The first $B$ MCMC samples come from a distribution close to $\mu^{(0)}$ (initial distribution), not $\pi$. Including them biases posterior estimates. | Always discard a burn-in period. Use $\hat{R}$ and ESS to assess convergence. |
| 7 | Confusing mixing time with burn-in | Mixing time $t_{\text{mix}}$ is a property of the chain (how long until the worst-case distribution is close to $\pi$). Burn-in is a practical heuristic. They are related but not equal. | Burn-in should be at least $t_{\text{mix}}$; more if starting far from $\pi$. |
| 8 | Assuming high acceptance rate = well-mixed MCMC | A chain that always accepts (tiny step size in MH) explores very slowly; acceptance rate ~1 with step size → 0 gives high acceptance but poor mixing. | Target ~23% acceptance in high dimensions (optimal for Gaussian targets). |
| 9 | Using $P^n$ to compute stationary distribution for large $n$ | Matrix exponentiation $P^n$ converges to the rank-1 matrix $\mathbf{1}\pi$, but floating-point errors accumulate. | Use power iteration on the distribution vector: $\pi^{(k+1)} = \pi^{(k)} P$, which is numerically stable. |
| 10 | Misidentifying the state space for LLM generation | A bigram LM has states = vocabulary (size $V$). A full autoregressive LM has states = entire context = exponential in context length. | The Markov chain for a transformer is on the context window, not the vocabulary alone. |
| 11 | Confusing transient and absorbing | A transient state will eventually be left for good; an absorbing state is one from which the chain never leaves. Every absorbing state is recurrent (trivially), not transient. | Transient: $P(\text{return}) < 1$; absorbing: $P_{ii}=1$. Absorbing states are recurrent. |
| 12 | Expecting MCMC samples to be independent | MCMC samples are correlated (autocorrelated) — consecutive samples are not independent. ESS < T measures effective independence. | Report ESS, not raw sample count. Use thinning to reduce autocorrelation if needed. |

---

## 11. Exercises

**Exercise 1 ★ — Transition Matrix and Distribution Evolution**

A weather model has two states: Sunny (1) and Rainy (2), with transition matrix $P = \begin{pmatrix}0.8 & 0.2 \\ 0.4 & 0.6\end{pmatrix}$.

(a) Starting from $\mu^{(0)} = (1, 0)$ (certainly sunny), compute $\mu^{(1)}, \mu^{(5)}, \mu^{(20)}$.

(b) Compute the 3-step transition matrix $P^3$. What is $P(X_3=2 \mid X_0=1)$?

(c) Find the stationary distribution $\pi$ by solving $\pi P = \pi$ with $\pi_1 + \pi_2 = 1$.

(d) Verify that $\mu^{(n)} \to \pi$ as $n \to \infty$ by computing $\|\mu^{(n)} - \pi\|_{\text{TV}}$ for $n = 1, 5, 10, 20$.

---

**Exercise 2 ★ — State Classification**

Consider the Markov chain on $\{1,2,3,4,5\}$ with transition matrix:
$$P = \begin{pmatrix}
0 & 1 & 0 & 0 & 0 \\
0.5 & 0 & 0.5 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0
\end{pmatrix}$$

(a) Draw the transition diagram. Identify all communicating classes.

(b) Classify each state as recurrent or transient. Justify using the return probability criterion.

(c) For the recurrent states, determine their period. Are any states aperiodic?

(d) Starting from state 1, what is the probability of eventually reaching state 4?

---

**Exercise 3 ★ — Computing Stationary Distributions**

(a) Implement `stationary_power_iteration(P, n_iter=1000)` that starts from a uniform distribution and returns $\pi^{(n)}$ after $n$ iterations. Apply to the $3\times3$ chain with $P_{ij} = 1/3$ for all $i,j$ and a $4\times4$ random row-stochastic matrix.

(b) Verify your result satisfies $\pi P = \pi$ to tolerance $10^{-6}$.

(c) For a doubly-stochastic matrix (rows AND columns sum to 1), prove the stationary distribution is uniform without solving any equations. Verify numerically.

(d) For the birth-death chain with $p_i = 0.6$, $q_i = 0.4$ on $\{0,1,2,3,4\}$ with reflecting barriers, compute $\pi$ analytically and verify against power iteration.

---

**Exercise 4 ★★ — Detailed Balance and Birth-Death Chains**

(a) Verify that the two-state chain $P = \begin{pmatrix}1-p&p\\q&1-q\end{pmatrix}$ satisfies detailed balance with $\pi = (q/(p+q), p/(p+q))$.

(b) Implement a $M/M/1$ queue birth-death chain with $\lambda=0.6, \mu=1.0$ on states $\{0,1,\ldots,20\}$ with an absorbing boundary at 20. Compute the stationary distribution and verify $\pi_n \approx (1-\rho)\rho^n$ with $\rho = 0.6$.

(c) For the cyclic chain $1\to2\to3\to1$ with $P_{12}=P_{23}=P_{31}=1$, find the stationary distribution. Does it satisfy detailed balance? Explain why not.

(d) Show that if $P$ is doubly stochastic, then detailed balance holds with the uniform distribution iff $P$ is also symmetric.

---

**Exercise 5 ★★ — Mixing Time and Spectral Gap**

(a) For the two-state chain in Exercise 1, compute the eigenvalues of $P$. What is the spectral gap?

(b) Using the spectral gap bound, upper-bound the mixing time $t_{\text{mix}}(0.01)$.

(c) Compute the TV distance $\|P^t(1,\cdot) - \pi\|_{\text{TV}}$ for $t = 1, 2, 5, 10, 20$ and plot the convergence. At what $t$ does it drop below $0.01$?

(d) For the lazy version $P' = (I+P)/2$, redo parts (a)-(c). How does the spectral gap change? How does the mixing time change?

---

**Exercise 6 ★★ — Metropolis-Hastings**

(a) Implement Metropolis-Hastings with a Gaussian random walk proposal $q(\theta'|\theta) = \mathcal{N}(\theta, \sigma^2)$ for target $\pi(\theta) \propto \exp(-\theta^4/4 + \theta^2/2)$ (double-well potential). Run for $T=50000$ steps with $\sigma=0.5$.

(b) Verify your sampler satisfies detailed balance: for two states $\theta_1, \theta_2$, check $\pi(\theta_1)\,K(\theta_1,\theta_2) \approx \pi(\theta_2)\,K(\theta_2,\theta_1)$ numerically where $K$ is the MH transition kernel.

(c) Compute the acceptance rate and effective sample size (ESS). How does ESS change as $\sigma$ varies over $\{0.1, 0.5, 1.0, 2.0, 5.0\}$?

(d) Estimate $\mathbb{E}_\pi[\theta^2]$ from your samples. Compare to the true value (computed numerically).

---

**Exercise 7 ★★★ — PageRank**

(a) Implement PageRank power iteration for the 5-node graph with adjacency matrix:
$$A = \begin{pmatrix}0&1&1&0&0\\0&0&1&1&0\\0&0&0&1&1\\1&0&0&0&1\\0&1&0&0&0\end{pmatrix}$$
with damping factor $\alpha=0.85$. Handle dangling nodes.

(b) Verify that your PageRank vector satisfies $\pi P = \pi$ to tolerance $10^{-8}$.

(c) Add a new node 6 that links to all existing nodes, and all existing nodes link to node 6. How does the PageRank of node 1 change? Why?

(d) Implement personalised PageRank where the teleportation vector is non-uniform: $(0.4, 0.2, 0.1, 0.1, 0.1, 0.1)$. Compare to standard PageRank.

---

**Exercise 8 ★★★ — HMM Forward Algorithm and Viterbi**

An HMM models DNA sequences with 2 hidden states (CpG island: $H$, non-CpG: $L$) and 4 observations (A, C, G, T).

(a) Given transition matrix $T_{HH}=0.5, T_{HL}=0.5, T_{LH}=0.4, T_{LL}=0.6$ and emission matrices (HMM notebook provides values), implement the forward algorithm and compute $P(X_{1:5} = \text{ACGTT})$ under equal initial probabilities.

(b) Implement the Viterbi algorithm to find the most likely hidden state sequence for the same observation.

(c) Verify the forward probabilities satisfy $\sum_i \alpha_t(i) = P(X_{1:t})$ at each $t$.

(d) Compare Viterbi decoding to the most-probable-state-per-position (posterior decoding using the backward algorithm). When do they differ?

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI/ML Application | Impact |
| --- | --- | --- |
| Transition matrix / stationary dist. | PageRank web graph; LLM token distribution | Foundation for all sequential reasoning |
| Ergodic theorem | MCMC posterior estimation; time-averaging in RL | Justifies replacing expectation with time average |
| Perron-Frobenius | PageRank convergence proof; convergence of power iteration | Guarantees unique ranking vectors exist |
| Detailed balance / reversibility | Correctness of MH, Gibbs, SGLD | All Bayesian MCMC algorithms rely on this |
| Spectral gap / mixing time | MCMC convergence rate; warm-start certification | Explains why MCMC works (or fails) in practice |
| Metropolis-Hastings | Bayesian neural network posteriors; sampling EBMs | Foundation of probabilistic inference in ML |
| Gibbs sampling | LDA topic models; Boltzmann machine inference | Tractable posterior sampling via conditionals |
| HMC / NUTS | Stan, PyMC, posterior sampling in Bayesian DL | Gold standard for Bayesian inference on $\mathbb{R}^d$ |
| MDP / Bellman equation | RL algorithms (PPO, SAC, DQN, RLHF) | Every RL agent solves a Markov chain problem |
| HMM forward-backward | Speech recognition, gene prediction, legacy NLP | Efficient inference on latent sequence models |
| CTMC / generator matrix | LLM serving queues; continuous-time RL | Latency modelling and event-driven simulation |
| Diffusion as CTMC | DDPM/DDIM forward process; score-matching | Connects generative modelling to Markov chain theory |
| Power iteration | PageRank; dominant eigenvector computation | $O(N^2)$ algorithm that scales to trillion-node graphs |
| Lazy chains / warm starts | MCMC initialisation strategy | Reduces burn-in and improves sample efficiency |

---

## 13. Conceptual Bridge

Markov chains occupy the central position in the probability curriculum: they are the first class of stochastic processes with nontrivial structure — neither completely independent (iid) nor completely dependent. The Markov property gives just enough memory to model real sequential phenomena while remaining mathematically tractable.

**Looking backward:** Markov chains build on everything in Chapters 1–6. The transition matrix is a row-stochastic matrix — linear algebra from Chapters 2–3. The stationary distribution is an eigenvector problem solved by Perron-Frobenius. Computing stationary distributions uses the conditional probability machinery from §03. The ergodic theorem is a strengthening of the law of large numbers (§06). The Metropolis-Hastings acceptance ratio uses Bayes' theorem (§03). Mixing times use concentration inequalities (§05) through the spectral gap. The stochastic process framework of filtrations and stopping times (§06) provides the rigorous foundation.

**Looking forward:** Markov chains are the gateway to Chapter 7 (Statistics), where MCMC enables Bayesian inference with intractable posteriors; Chapter 8 (Optimisation), where Markov chain mixing theory explains why SGD finds flat minima; and Chapter 9 (Information Theory), where the entropy of a stationary distribution connects to compression. Markov Decision Processes — the reinforcement learning setting — are a direct extension requiring optimisation over policies.

**The deep connection:** The central theorem of Markov chain theory — ergodicity implies $\mu^{(n)} \to \pi$ — is the probabilistic analogue of the contraction mapping theorem from analysis: repeated application of the transition operator contracts all distributions toward the unique fixed point $\pi$. The spectral gap quantifies the contraction rate, just as the Lipschitz constant does in the deterministic setting. This structural parallel runs through SGD convergence theory, where the gradient operator acts as a stochastic contraction.

```
MARKOV CHAINS IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  §01 Random Variables ─────────────────────────────────────┐
  §02 Distributions ──────────────────────────────────────┐  │
  §03 Joint Distributions  ───────────────────────────┐   │  │
  §04 Expectation & Moments ──────────────────────┐   │   │  │
  §05 Concentration Inequalities ─────────────┐   │   │   │  │
  §06 Stochastic Processes ───────────────┐   │   │   │   │  │
                                           ↓   ↓   ↓   ↓  ↓  ↓
                              ┌─────────────────────────────────┐
                              │   §07  MARKOV  CHAINS           │
                              │                                 │
                              │  Markov property                │
                              │  Transition matrices            │
                              │  Stationary distributions       │
                              │  Mixing times, spectral gap     │
                              │  MCMC (MH, Gibbs, HMC)         │
                              │  PageRank, MDP, HMM             │
                              └──────────────┬──────────────────┘
                                             │
               ┌─────────────────────────────┼────────────────────┐
               ↓                             ↓                    ↓
    Ch7: Statistics               Ch8: Optimisation        Ch9: Info Theory
    (MCMC for Bayes,              (SGD mixing theory,      (Entropy of π,
     posterior inference)          Langevin dynamics)       KL from stationarity)
               ↓                             ↓
    RL/RLHF: MDP                  Diffusion models
    (policy eval,                 (DDPM as CTMC,
     value functions)              reverse Markov)

════════════════════════════════════════════════════════════════════════
```


---

## Appendix A: Harmonic Functions and the Dirichlet Problem

A function $h : \mathcal{S} \to \mathbb{R}$ is **harmonic** for a Markov chain at state $i$ if:
$$h(i) = \sum_j P_{ij} h(j) = (Ph)(i)$$

That is, $h(i)$ equals the expected value of $h$ one step later. Harmonic functions are the "conserved quantities" of Markov chains — if $h(X_0)$ is the starting value, then $h(X_n)$ forms a martingale.

**Maximum principle:** For an irreducible chain on a finite state space, the only bounded harmonic functions are constants. This is the probabilistic analogue of the maximum principle for harmonic functions in PDE theory.

**Dirichlet problem:** On a chain with absorbing boundary $\partial \mathcal{S}$ and interior $\mathcal{S}^\circ$, the function $h(i) = E_i[f(X_{\tau_\partial})]$ (expected boundary value at hitting time) is the unique solution to:
$$h(i) = (Ph)(i) \text{ for } i \in \mathcal{S}^\circ, \quad h(i) = f(i) \text{ for } i \in \partial\mathcal{S}$$

**Application — gambler's ruin:** Let $h(i) = P(\text{reach } N \mid X_0=i)$. Then $h$ is harmonic on $\{1,\ldots,N-1\}$ with boundary conditions $h(0)=0$, $h(N)=1$. For the simple random walk: $h(i) = i/N$.

**For AI:** Value functions in reinforcement learning are harmonic: $V^\pi(s) = \sum_a \pi(a|s)[r(s,a) + \gamma \sum_{s'}P(s'|s,a)V^\pi(s')]$ is a Bellman equation — a discrete-time analogue of the Dirichlet problem with discount factor $\gamma$.

---

## Appendix B: Spectral Theory for Non-Reversible Chains

For non-reversible chains, eigenvalues of $P$ can be complex (though all have $|\lambda| \leq 1$). The Jordan form replaces the diagonal eigendecomposition.

**Pseudo-spectrum:** For nearly-reversible chains, the pseudo-spectrum can be much larger than the spectrum, causing transient amplification before eventual convergence.

**Non-reversible speedup:** Some non-reversible chains mix faster than any reversible chain with the same stationary distribution. The "lifted" or "skewed" chains in the literature achieve mixing time $O(1/h^2)$ vs. $O(1/h)$ for reversible chains, where $h$ is the Cheeger constant.

**Lifting:** A common technique to construct non-reversible fast-mixing chains: add a "direction" variable $d \in \{+1,-1\}$ to the state, creating a Markov chain on $\mathcal{S} \times \{+,-\}$ that preferentially moves in one direction while still having $\pi$ as marginal stationary distribution.

---

## Appendix C: Advanced MCMC Methods

### C.1 Stochastic Gradient Langevin Dynamics (SGLD)

For Bayesian inference on large datasets, standard MH requires evaluating $\log\pi(\theta)$ at the full dataset — too expensive. SGLD combines SGD with Gaussian noise:
$$\theta_{t+1} = \theta_t - \frac{\eta_t}{2}\nabla \tilde{L}(\theta_t) + \sqrt{\eta_t}\,\xi_t, \quad \xi_t \sim \mathcal{N}(0,I)$$

where $\tilde{L}(\theta) = -\frac{N}{n}\sum_{i \in \text{minibatch}} \log p(x_i|\theta) - \log p(\theta)$ is the mini-batch stochastic gradient.

For step sizes satisfying $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ (Robbins-Monro conditions), the chain converges to samples from the posterior. At small but fixed step size $\eta$, the chain samples from an approximate posterior within $O(\eta)$ of the true posterior.

**For AI:** SGLD underlies Bayesian deep learning at scale. It explains why SGD with learning rate decay has regularisation properties: the decaying noise variance means the chain converges to a tighter distribution around the posterior mode.

### C.2 Parallel Tempering (Replica Exchange)

For multimodal targets, single chains can get stuck in one mode. **Parallel tempering** runs $K$ chains at different temperatures $\beta_1 < \beta_2 < \cdots < \beta_K$ (with $\beta_K = 1$ being the target). Periodically, adjacent chains swap states with MH acceptance probability:
$$a = \min\left(1,\, \frac{\pi_{\beta_i}(x_j)\pi_{\beta_j}(x_i)}{\pi_{\beta_i}(x_i)\pi_{\beta_j}(x_j)}\right)$$

High-temperature chains mix fast (flat distribution) and pass samples to low-temperature chains through swaps. This enables exploration of multiple modes.

### C.3 Slice Sampling

Slice sampling introduces a uniform auxiliary variable $u \sim \text{Uniform}(0, \pi(\theta))$, creating a joint distribution on $(\theta, u)$ that is uniform on the "slice" $\{(\theta, u) : u < \pi(\theta)\}$. Marginalising out $u$ recovers $\pi(\theta)$.

The slice sampler alternates between updating $u$ (trivial given $\theta$) and updating $\theta$ (uniform on the slice level set). No tuning parameter; acceptance rate = 1; effective for univariate distributions.

---

## Appendix D: Worked Examples

### D.1 Google PageRank Computation (Mini Example)

Consider a 4-page web: page 1 links to 2,3; page 2 links to 4; page 3 links to 2; page 4 links to 1,3.

Transition matrix (with no damping for clarity):
$$P = \begin{pmatrix}0&1/2&1/2&0\\0&0&0&1\\0&1&0&0\\1/2&0&1/2&0\end{pmatrix}$$

Stationary distribution (solve $\pi P = \pi$, $\sum\pi_i=1$):
$\pi = (4/13, 3/13, 4/13, 2/13)$ — verified by direct computation.

With damping $\alpha=0.85$: $P' = 0.85 P + 0.15 \cdot J/4$ where $J$ is the all-ones matrix. Power iteration converges in ~30 iterations.

### D.2 MCMC on a Bimodal Distribution

Target: $\pi(\theta) \propto e^{-(\theta-3)^2/2} + e^{-(\theta+3)^2/2}$ (mixture of two Gaussians at $\pm 3$).

With Gaussian random walk MH ($\sigma=0.5$): chain mixes within one mode; inter-modal jumps are rare because the barrier between modes is $O(10)$ nats. Acceptance rate ~60%, but ESS/T ~0.02 (poor mixing).

With $\sigma=3$: chain jumps between modes; acceptance rate ~15%, ESS/T ~0.10 (better mixing).

This illustrates the exploration-exploitation tradeoff in MCMC proposal design.

### D.3 HMM Example: Weather from Ice Core

Two hidden states: Ice Age (I), Warm Period (W). Observations: high ($h$) or low ($l$) dust in ice core.

Transition: $P_{II}=0.7, P_{IW}=0.3, P_{WI}=0.4, P_{WW}=0.6$.
Emission: $P(h|I)=0.8, P(l|I)=0.2, P(h|W)=0.2, P(l|W)=0.8$.

For observation sequence $h, l, h, h, l$: the Viterbi algorithm gives hidden sequence $I, W, I, I, W$ (high dust = ice age, low dust = warm period).

Forward algorithm gives likelihood $P(\text{obs sequence}) \approx 0.0035$.

---

## Appendix E: Key Theorems and Proofs

### E.1 Proof: Stationary Distribution is Unique for Finite Irreducible Chains

**Theorem.** A finite irreducible Markov chain has a unique stationary distribution $\pi$ with $\pi_i > 0$ for all $i$.

**Proof.** Consider the $N$-dimensional simplex $\Delta = \{\mu : \mu_i \geq 0, \sum\mu_i = 1\}$. The map $T: \mu \mapsto \mu P$ is a continuous map from $\Delta$ to $\Delta$. By Brouwer's fixed point theorem, $T$ has at least one fixed point $\pi$.

For uniqueness: suppose $\pi$ and $\nu$ are two stationary distributions. Since the chain is irreducible and finite, all states are positive recurrent, so $\pi_i = 1/\mu_i > 0$ where $\mu_i$ is the mean return time. The mean return time is unique (it equals $1/\pi_i$ and also $1/\nu_i$ if $\nu$ is stationary), so $\pi_i = \nu_i$ for all $i$.

For $\pi_i > 0$: by irreducibility, from any state $j$ there exists a path to $i$. The stationary mass at each state in this path is $> 0$ (by positivity of transitions and stationarity), propagating to give $\pi_i > 0$.

### E.2 Proof: Detailed Balance Implies Stationarity

**Theorem.** If $\pi_i P_{ij} = \pi_j P_{ji}$ for all $i,j$, then $\pi P = \pi$.

**Proof.** For each $j$:
$$(\pi P)_j = \sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} = \pi_j \sum_i P_{ji} = \pi_j \cdot 1 = \pi_j$$

where we used detailed balance in the second equality and the row-sum property of stochastic matrices in the fourth.

### E.3 Proof: Convergence via Coupling (Sketch)

**Theorem.** Let $P$ be ergodic. For any $\mu, \nu$ and coupling $(X_t, Y_t)$ with $X_0 \sim \mu$, $Y_0 \sim \nu$, define $\tau = \min\{t : X_t = Y_t\}$. Then:
$$\|\mu P^t - \nu P^t\|_{\text{TV}} \leq P(\tau > t)$$

**Proof.** For any event $A$:
$$\mu P^t(A) - \nu P^t(A) = P(X_t \in A) - P(Y_t \in A) = P(X_t \in A, \tau \leq t) + P(X_t \in A, \tau > t) - P(Y_t \in A, \tau \leq t) - P(Y_t \in A, \tau > t)$$

On $\{\tau \leq t\}$: $X_t = Y_t$ (coalesce), so $P(X_t \in A, \tau \leq t) = P(Y_t \in A, \tau \leq t)$. Therefore:
$$|\mu P^t(A) - \nu P^t(A)| = |P(X_t \in A, \tau > t) - P(Y_t \in A, \tau > t)| \leq P(\tau > t)$$

Taking supremum over $A$ gives the result.

---

## Appendix F: Python Recipes for Markov Chains

```python
import numpy as np

# ── 1. Power iteration for stationary distribution
def stationary(P, tol=1e-12, max_iter=10000):
    pi = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(max_iter):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new
    return pi

# ── 2. Simulate a Markov chain
def simulate_chain(P, x0, n_steps):
    N = P.shape[0]
    x = x0
    trajectory = [x]
    for _ in range(n_steps):
        x = np.random.choice(N, p=P[x])
        trajectory.append(x)
    return np.array(trajectory)

# ── 3. Metropolis-Hastings for 1D target
def metropolis_hastings(log_pi, n_steps, sigma=0.5, x0=0.0):
    x = x0
    samples = []
    for _ in range(n_steps):
        x_prime = x + sigma * np.random.randn()
        log_a = log_pi(x_prime) - log_pi(x)
        if np.log(np.random.rand()) < log_a:
            x = x_prime
        samples.append(x)
    return np.array(samples)

# ── 4. PageRank
def pagerank(A, alpha=0.85, tol=1e-8):
    N = A.shape[0]
    out_degree = A.sum(axis=1, keepdims=True)
    # Fix dangling nodes
    dangling = (out_degree == 0).flatten()
    out_degree[dangling] = 1
    P = A / out_degree
    P[dangling] = 1 / N
    teleport = np.ones((N, N)) / N
    P_hat = alpha * P + (1 - alpha) * teleport
    pi = np.ones(N) / N
    for _ in range(1000):
        pi_new = pi @ P_hat
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new
    return pi

# ── 5. HMM forward algorithm
def hmm_forward(obs, T, E, pi0):
    """obs: list of observations, T: N×N transition, E: N×M emission, pi0: initial."""
    N = T.shape[0]
    alpha = np.zeros((len(obs), N))
    alpha[0] = pi0 * E[:, obs[0]]
    for t in range(1, len(obs)):
        alpha[t] = (alpha[t-1] @ T) * E[:, obs[t]]
    return alpha, alpha[-1].sum()  # alpha matrix and likelihood
```

---

## Appendix G: Connections to Other Areas

**Spectral graph theory:** For a random walk on an undirected graph $G$, the transition matrix $P = D^{-1}A$ (where $D$ is the degree matrix and $A$ is the adjacency matrix) has eigenvalues in $[-1,1]$. The spectral gap of the Laplacian $L = I - D^{-1/2}AD^{-1/2}$ equals the spectral gap of the walk. Well-connected (expander) graphs have large spectral gaps and fast-mixing random walks.

**Information theory:** The relative entropy (KL divergence) between the chain's distribution and stationarity decreases monotonically: $D_{\text{KL}}(\mu^{(n)} \| \pi) \geq D_{\text{KL}}(\mu^{(n+1)} \| \pi)$. This is the **data processing inequality** applied to the Markov transition. The rate of KL decrease is related to the spectral gap.

**Ergodic theory:** The Markov chain ergodic theorem is a special case of Birkhoff's ergodic theorem in abstract measure-preserving systems. The stationarity condition $\pi P = \pi$ says $\pi$ is an invariant measure for the measure-preserving map $T : \Omega \to \Omega$ on the path space.

**Linear programming:** Computing the stationary distribution satisfying $\pi P = \pi$, $\pi \geq 0$, $\|\pi\|_1 = 1$ is a linear system. The LP relaxation of combinatorial problems can sometimes be solved by finding stationary distributions of associated Markov chains.

**Quantum computing:** Quantum walks are the quantum analogue of random walks on graphs, with amplitudes instead of probabilities. They can achieve quadratic speedups over classical random walks for certain search and sampling problems.

---

## Appendix H: Self-Assessment Checklist

Before moving to Chapter 7 (Statistics), verify you can:

- [ ] Write the transition matrix for any small Markov chain from a verbal description
- [ ] Compute $\mu^{(n)} = \mu^{(0)}P^n$ for $n = 1, 2, 5$ by hand or code
- [ ] Classify all states of a small chain as communicating/recurrent/transient/absorbing
- [ ] Solve $\pi P = \pi$ for a $2\times2$ or $3\times3$ stochastic matrix
- [ ] State Perron-Frobenius and explain why the stationary distribution is unique
- [ ] Verify detailed balance for a given $(\pi, P)$ pair
- [ ] Compute the spectral gap from eigenvalues and give a mixing time bound
- [ ] Implement Metropolis-Hastings for a 1D target and compute acceptance rate
- [ ] Implement power iteration for PageRank with dangling node handling
- [ ] Implement the HMM forward algorithm and compute the likelihood of an observation sequence
- [ ] Explain why MCMC sample autocorrelation reduces the effective sample size
- [ ] State the ergodic theorem and explain why it justifies MCMC estimation


---

## Appendix I: Detailed Proofs — Perron-Frobenius and Mixing

### I.1 Perron-Frobenius Theorem (Complete Proof)

We prove the Perron-Frobenius theorem for primitive stochastic matrices.

**Step 1: Eigenvalue 1 exists.** For any stochastic matrix $P$, $P\mathbf{1} = \mathbf{1}$ (column vector of ones). So $\mathbf{1}^T P = \mathbf{1}^T$ iff $P$ is doubly stochastic. More relevantly: by the Brouwer fixed-point theorem on $\Delta^{N-1}$, the map $\pi \mapsto \pi P$ has a fixed point — the stationary distribution. This fixed point is a left eigenvector with eigenvalue 1.

**Step 2: $|\lambda| \leq 1$ for all eigenvalues.** Let $\lambda$ be an eigenvalue with (right) eigenvector $v$ ($Pv = \lambda v$). Pick $v_{\max} = \max_i |v_i|$. Then $|\lambda||v_{\max}| = |\lambda v_{i^*}| = |(Pv)_{i^*}| = |\sum_j P_{i^*j}v_j| \leq \sum_j P_{i^*j}|v_j| \leq v_{\max}$. So $|\lambda| \leq 1$.

**Step 3: $|\lambda| = 1$ implies $\lambda = 1$ for primitive $P$.** For a primitive matrix, $P^m > 0$ (all entries strictly positive) for some $m$. Suppose $|\lambda| = 1$, $\lambda \neq 1$, with $Pv = \lambda v$ and $v_{\max} = 1$. The equality case in Step 2 requires $|v_j| = 1$ for all $j$ and all rows of $P$ to "align" the phases of $v_j$. But for $P^m > 0$, all rows of $P^m$ are strictly positive, and the alignment condition forces $\lambda^m = 1$ and all $v_j$ equal — meaning $v = c\mathbf{1}$ and $\lambda = 1$. Contradiction.

**Step 4: Convergence.** Since $\lambda = 1$ is the unique eigenvalue on the unit circle for primitive $P$, all other eigenvalues $\lambda_k$ satisfy $|\lambda_k| \leq 1 - \delta$ for some $\delta > 0$. The spectral decomposition gives $P^n = \mathbf{1}\pi + \sum_{k \geq 2} \lambda_k^n \phi_k \psi_k^T$ where $|\lambda_k|^n \to 0$ at rate $(1-\delta)^n$.

### I.2 Coupling Proof of Convergence (Complete)

**Theorem.** Let $P$ be ergodic with stationary distribution $\pi$. For any $x \in \mathcal{S}$:
$$\|P^n(x,\cdot) - \pi\|_{\text{TV}} \leq P(\tau > n)$$

where $\tau$ is the coalescence time of the Markov coupling (two chains started at $x$ and $\sim\pi$).

**Proof.** Let $(X_n, Y_n)$ be a coupling with $X_0 = x$ and $Y_0 \sim \pi$, where both chains use the same transition $P$ and coalesce when they meet. Since $Y_0 \sim \pi$ and $\pi$ is stationary, $Y_n \sim \pi$ for all $n$.

For any event $A$:
$$P^n(x, A) - \pi(A) = P(X_n \in A) - P(Y_n \in A)$$

On $\{\tau \leq n\}$: $X_n = Y_n$ (they have coalesced), so their indicators for $A$ are equal.

$$P^n(x, A) - \pi(A) = E[\mathbf{1}_{X_n \in A} - \mathbf{1}_{Y_n \in A}]$$
$$= E[(\mathbf{1}_{X_n \in A} - \mathbf{1}_{Y_n \in A})\mathbf{1}_{\tau > n}]$$
$$\leq E[\mathbf{1}_{\tau > n}] = P(\tau > n)$$

Since this holds for any $A$ and for $A^c$ by symmetry, taking the supremum gives the TV bound.

### I.3 Spectral Gap Lower Bound via Conductance (Cheeger Inequality)

**Definition.** The **conductance** (Cheeger constant) of a chain is:
$$h = \min_{S \subseteq \mathcal{S}, \pi(S) \leq 1/2} \frac{\sum_{i \in S, j \notin S} \pi_i P_{ij}}{\pi(S)}$$

Conductance measures the minimum probability flux crossing any "bottleneck" cut in the chain.

**Cheeger Inequality:** For a reversible chain:
$$\frac{h^2}{2} \leq \text{gap} \leq 2h$$

The lower bound is the harder direction and gives: $\text{gap} \geq h^2/2$. In words: a large bottleneck (small $h$) implies a small spectral gap (slow mixing). Conversely, if $h$ is bounded below, the chain mixes in $O(1/h^2)$ steps (vs. $O(1/h)$ for the tighter bound $\text{gap} \leq 2h$).

**For AI:** The Cheeger inequality provides the tightest practical bounds for MCMC convergence. For neural network posteriors, the conductance can be estimated using gradient information — a well-conditioned posterior (no sharp barriers) has high conductance.

---

## Appendix J: Markov Chains and Entropy

### J.1 Entropy Production

The **relative entropy** (KL divergence) from the chain's distribution to stationarity is:
$$H(\mu^{(n)} \| \pi) = \sum_i \mu^{(n)}_i \log \frac{\mu^{(n)}_i}{\pi_i}$$

**Data processing inequality (for Markov chains):** $H(\mu^{(n+1)} \| \pi) \leq H(\mu^{(n)} \| \pi)$

This follows from the data processing inequality: applying the stochastic map $P$ cannot increase KL divergence. Equality holds iff $\mu^{(n)} = \pi$.

**For reversible chains:** The relative entropy decays at rate governed by the log-Sobolev constant. Chains with larger log-Sobolev constants (tighter functional inequalities) decay faster.

### J.2 Entropy of the Stationary Distribution

The Shannon entropy of $\pi$ is $H(\pi) = -\sum_i \pi_i \log \pi_i$. For a uniform distribution (doubly stochastic $P$), $H(\pi) = \log N$ is maximised. For PageRank, the entropy of $\pi$ measures how "concentrated" web traffic is — low entropy means a few pages dominate.

**Maximum entropy interpretation:** The stationary distribution of a reversible chain with detailed balance can be interpreted as the maximum entropy distribution subject to the constraint that the detailed balance equations hold. This connects Markov chains to the exponential family (§02).

### J.3 Kullback-Leibler and MCMC Acceptance

The Metropolis-Hastings acceptance ratio has an information-theoretic interpretation:
$$a(\theta, \theta') = \min\left(1, \frac{\pi(\theta')q(\theta|\theta')}{\pi(\theta)q(\theta'|\theta)}\right) = \min\left(1, e^{\log\pi(\theta')-\log\pi(\theta)-\log q(\theta'|\theta)+\log q(\theta|\theta')}\right)$$

The exponent is the log-ratio of probability densities — related to the KL divergence between the proposal and target. When the proposal $q$ matches $\pi$ well, most proposals are accepted.

---

## Appendix K: Continuous-Time Chains and Generators

### K.1 Semigroup Theory

The family of transition matrices $\{P(t)\}_{t \geq 0}$ forms a **Markov semigroup**: $P(0) = I$, $P(s+t) = P(s)P(t)$, and $t \mapsto P(t)$ is continuous in $t$. The generator $Q = \frac{d}{dt}P(t)\big|_{t=0}$ characterises the infinitesimal behaviour.

**Exponential formula:** $P(t) = e^{Qt}$ where the matrix exponential can be computed via eigendecomposition of $Q$. If $Q$ has eigenvalues $0 = \mu_1 \geq \mu_2 \geq \cdots \geq \mu_N$ (all real and non-positive for reversible chains), then:
$$P(t) = \sum_k e^{\mu_k t} \phi_k \psi_k^T$$

Convergence to stationarity is at rate $e^{|\mu_2|t}$ where $|\mu_2| = -\mu_2$ is the spectral gap of $Q$.

### K.2 Reversible CTMCs

A CTMC is reversible with respect to $\pi$ if $\pi_i Q_{ij} = \pi_j Q_{ji}$ for all $i \neq j$. The spectral theory for reversible CTMCs is identical to that for reversible DTMCs: real eigenvalues, orthonormal eigenfunctions, exponential convergence.

**Detailed balance for CTMCs:**
$$\pi_i q_{ij} = \pi_j q_{ji} \text{ for all } i \neq j$$

This is the continuous-time analogue of the discrete detailed balance. Metropolis-Hastings in continuous time (the Metropolis algorithm for particle simulations) satisfies this by construction.

---

## Appendix L: Theory Problems

**Problem L.1.** Prove that for a doubly stochastic matrix $P$ (rows and columns both sum to 1), the uniform distribution is stationary. Give an example showing the converse fails.

**Problem L.2.** Let $P$ be irreducible with stationary distribution $\pi$. Prove that $\pi_i > 0$ for all $i$ by constructing a path from any $j$ with $\pi_j > 0$ to $i$.

**Problem L.3.** (Coupling inequality) Let $P$ be ergodic. Construct a coupling of two copies of the chain, one started at $x$ and one at the stationary distribution $\pi$. Use the coupling inequality to prove $\|P^n(x,\cdot) - \pi\|_{\text{TV}} \leq P(\tau_{\text{meet}} > n)$.

**Problem L.4.** For the symmetric random walk on $\{0,1,\ldots,N\}$ with reflecting barriers, show the spectral gap is $1 - \cos(\pi/N) \approx \pi^2/(2N^2)$ for large $N$. What is the mixing time?

**Problem L.5.** (Metropolis-Hastings correctness) Let $K(x,y)$ be the MH transition kernel with proposal $q$ and target $\pi$. Show that $\pi_x K(x,y) = \pi_y K(y,x)$ (detailed balance), and conclude $\pi$ is stationary.

**Problem L.6.** For a birth-death chain on $\{0,1,\ldots,N\}$ with rates $p_i = p$ and $q_i = q$ (constant), find the stationary distribution. For what values of $p$ does the chain have a proper stationary distribution on $\{0,1,2,\ldots\}$ (infinite chain)?

**Problem L.7.** The **lazy walk** on a graph $G$ sets $P_{ii} = 1/2$ (stay with prob 1/2) and $P_{ij} = 1/(2d_i)$ for neighbours $j$ (where $d_i$ is the degree). Show the lazy walk has all non-negative eigenvalues and therefore converges monotonically to stationarity in TV distance.

**Problem L.8.** (PageRank convergence rate) For the damped PageRank matrix $P = \alpha P_0 + (1-\alpha)J/N$ with damping $\alpha$, show all eigenvalues except the dominant one have magnitude $\leq \alpha$. Conclude the PageRank power iteration converges in $O(\log(1/\varepsilon)/\log(1/\alpha))$ iterations.

---

## Appendix M: Notation Summary

| Symbol | Meaning |
| --- | --- |
| $\mathcal{S}$ | State space |
| $P_{ij}$ | Transition probability from state $i$ to state $j$ |
| $P^{(n)}_{ij}$ | $n$-step transition probability ($(P^n)_{ij}$) |
| $\mu^{(n)}$ | Distribution at time $n$; row vector |
| $\pi$ | Stationary distribution; row vector with $\pi P = \pi$ |
| $f_i$ | Return probability to state $i$: $P(T_i < \infty \mid X_0=i)$ |
| $\mu_i$ | Mean return time to state $i$; $\pi_i = 1/\mu_i$ |
| $d(i)$ | Period of state $i$: $\gcd\{n \geq 1 : P^{(n)}_{ii} > 0\}$ |
| $\|\mu - \nu\|_{\text{TV}}$ | Total variation distance between $\mu$ and $\nu$ |
| $t_{\text{mix}}(\varepsilon)$ | Mixing time to $\varepsilon$-accuracy |
| $\text{gap}(P)$ | Spectral gap: $1 - |\lambda_2|$ for reversible $P$ |
| $h$ | Cheeger constant (conductance) |
| $Q$ | Generator matrix of a CTMC; $Q_{ij} = q_{ij}$, $Q_{ii} = -q_i$ |
| $P(t) = e^{Qt}$ | CTMC transition matrix at time $t$ |
| $a(\theta,\theta')$ | MH acceptance probability |
| $\tau_{\text{meet}}$ | Coupling time (coalescence time of two chains) |
| $\alpha_t(i)$ | HMM forward variable: $P(X_1,\ldots,X_t, Z_t=i)$ |
| $\delta_t(j)$ | Viterbi variable: $\max_{z_{1:t-1}} P(X_1,\ldots,X_t, Z_t=j, z_{1:t-1})$ |


---

## Appendix N: Advanced ML Applications

### N.1 RLHF as a Constrained MDP

Reinforcement learning from human feedback (RLHF) trains a language model policy $\pi_\theta$ to maximise:
$$\mathbb{E}_{\pi_\theta}[r_\phi(x, y)] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

where $r_\phi$ is a learned reward model, $\pi_{\text{ref}}$ is the reference (SFT) policy, and $\beta$ controls the KL penalty.

This is a **constrained Markov chain optimisation problem**: the state space is all possible prefixes (contexts), the action space is the vocabulary, and the KL divergence penalty ensures the learned chain $\pi_\theta$ doesn't deviate too far from the reference chain $\pi_{\text{ref}}$ in transition structure.

The optimal policy has an explicit closed form:
$$\pi^*(y|x) = \pi_{\text{ref}}(y|x) \cdot \frac{e^{r_\phi(x,y)/\beta}}{Z(x)}$$

where $Z(x)$ is a normalising constant. This is a **Gibbs distribution** with the reward as energy function — exactly the target distribution for MCMC.

**Implication:** The RLHF optimal policy can be sampled by constructing a Markov chain with the above stationary distribution. Recent work uses Langevin-type MCMC to directly sample from $\pi^*$ without RL.

### N.2 Diffusion Sampling as Time-Reversed CTMC

The score-based generative modelling framework (Song et al., 2021) writes the forward noising process as a CTMC/SDE and generates by running the reversed process. The key identity is **Anderson's reverse-time SDE**:

$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar{B}_t$$

where $\bar{B}_t$ is a Brownian motion running backwards in time. The **score function** $\nabla_x \log p_t(x)$ is the generator of the reversed Markov chain. Learning this score function via denoising score matching allows sampling by running the reverse process.

**Discrete diffusion:** For categorical distributions (text), the VP-SDE is replaced by a CTMC on the vocabulary with generator $Q(t)$. MDLM and other discrete diffusion models use the forward chain $q(x_t|x_0)$ = masked tokens, Uniform transitions, or absorbing-state corruption, then learn the reverse chain.

### N.3 MCMC for LLM Sampling

Standard autoregressive LLM decoding (top-$k$, nucleus) can be viewed as Markov chain sampling from a specific transition matrix. Recent **speculative decoding** methods (Chen et al., 2023; Leviathan et al., 2023) use a smaller "draft" model as a proposal and the large model as the acceptance criterion — this is precisely **independence Metropolis-Hastings**:

- Proposal: $q(x_{n+1}|\text{context}) = \text{small\_LM}(x_{n+1}|\text{context})$
- Acceptance: $a = \min(1, \text{large\_LM}/\text{small\_LM})$
- Guarantees: the accepted tokens have the same distribution as large-LM-only sampling

This is MCMC applied to language model decoding — Markov chain theory directly yields correctness of speculative decoding.

### N.4 Graph Neural Networks and Random Walks

Graph Neural Networks (GNNs) can be understood through the lens of Markov chains. A single GNN layer performs:
$$h_v^{(k+1)} = f\left(h_v^{(k)},\, \text{AGGREGATE}\left(\{h_u^{(k)} : u \in \mathcal{N}(v)\}\right)\right)$$

For the simple mean-aggregation GNN: $h_v^{(k+1)} = \sigma(W h_v^{(k)} + W'\frac{1}{d_v}\sum_{u \sim v} h_u^{(k)})$. The aggregation step is applying the random walk transition matrix $D^{-1}A$ to the feature matrix. $k$ layers of GNN = applying the random walk $k$ times — the receptive field grows as the $k$-step neighbourhood.

**Over-smoothing:** After many layers, all node features converge to the stationary distribution of the random walk (uniform for regular graphs). This is **over-smoothing** — the GNN loses discriminative power because the Markov chain mixes. Mixing time bounds from §6 directly bound the number of useful GNN layers.

### N.5 Token Position Encoding as Markov Chain

The self-attention mechanism with rotary position encoding (RoPE) can be viewed as computing a position-dependent Markov chain over tokens. Each attention head defines a stochastic matrix over token positions (the attention weights), and information flows along this chain.

**Attention as ergodic averaging:** In multi-head attention, each head computes $\text{softmax}(QK^T/\sqrt{d})V$. The softmax weights form a row-stochastic matrix — a one-step Markov transition. The output is the expected value of $V$ under this distribution. Stacking attention layers is like composing Markov kernels.

**Information bottleneck:** The rank of the attention matrix limits the "state space" of the Markov chain. Low-rank attention (as in linear attention / Performer) corresponds to a Markov chain with restricted state space — mixing may be faster but with less expressivity.

---

## Appendix O: Review Problems

**Review Problem 1.** Let $P$ be a $3\times3$ stochastic matrix with $P_{11}=0.5, P_{12}=0.3, P_{13}=0.2, P_{21}=0.1, P_{22}=0.7, P_{23}=0.2, P_{31}=0.3, P_{32}=0.3, P_{33}=0.4$. (a) Is this chain irreducible? (b) Find the stationary distribution. (c) Starting from state 1, compute the distribution after 10 steps. (d) Compare to stationarity.

**Review Problem 2.** Consider a random walk on a cycle of length $N$ (states $\{0,1,\ldots,N-1\}$, move clockwise or counterclockwise with equal probability $1/2$). (a) What is the stationary distribution? (b) What is the period? (c) Compute the mixing time using the spectral gap (eigenvalues: $\cos(2\pi k/N)$ for $k=0,\ldots,N-1$). (d) How does mixing time scale with $N$?

**Review Problem 3.** (MCMC) You want to sample from $\pi(x) \propto x^{a-1}e^{-bx}$ (a Gamma distribution) using MH with log-normal proposal $q(x'|x) = \text{LogNormal}(\log x, \sigma^2)$. (a) Write down the acceptance ratio. (b) Is $q$ symmetric? (c) Will this chain have $\pi$ as stationary distribution?

**Review Problem 4.** (PageRank) The web has 3 pages: 1 links to 2 and 3; 2 links to 1; 3 links to 1. With damping $\alpha=0.85$, write the PageRank transition matrix and find $\pi$ by solving a linear system. Which page has the highest PageRank?

**Review Problem 5.** (HMM) For an HMM with 2 hidden states, 2 observations, transition $T = \begin{pmatrix}0.6&0.4\\0.3&0.7\end{pmatrix}$, emission $E = \begin{pmatrix}0.9&0.1\\0.2&0.8\end{pmatrix}$, initial $\pi_0 = (0.5, 0.5)$: (a) Compute $P(\text{obs}=(0,1,0))$ using the forward algorithm. (b) Find the Viterbi path.

---

## Appendix P: Common Distributions in MCMC Targets

When $\pi(\theta) \propto \exp(-f(\theta))$, the gradient $\nabla f$ drives Langevin dynamics. Key shapes:

**Gaussian:** $f(\theta) = \frac{1}{2}\theta^T\Sigma^{-1}\theta$. Gradient: $\Sigma^{-1}\theta$. Langevin mixes in $O(\kappa(\Sigma))$ steps. HMC mixes in $O(\kappa(\Sigma)^{1/2})$ steps (advantage for ill-conditioned targets).

**Logistic posterior:** $f(\theta) = -\sum_i y_i x_i^T\theta + \sum_i \log(1+e^{x_i^T\theta}) + \frac{1}{2\sigma^2}\|\theta\|^2$. Strongly log-concave; SGLD mixes in polynomial time.

**Mixture of Gaussians:** $\pi = \sum_k w_k \mathcal{N}(\mu_k, \Sigma_k)$. Multi-modal; standard MCMC can get stuck between modes. Parallel tempering or simulated annealing needed for good mixing.

**Heavy-tailed:** $f(\theta) = \nu\log(1 + \|\theta\|^2/\nu)$ (Student-t). Sub-quadratic growth means the tails are heavy. HMC can struggle; NUTS with dual averaging handles this.

**Boltzmann (energy-based models):** $f(\theta) \propto -E_\phi(x)$ for neural network energy $E_\phi$. Sampling requires running MCMC; contrastive divergence (CD-$k$) uses short MCMC chains as an approximation.

---

## Appendix Q: Further Reading

1. **Levin, Peres, Wilmer** — *Markov Chains and Mixing Times* (AMS, 2nd ed., 2017) — the definitive reference for mixing times and coupling; freely available online

2. **Norris, J.R.** — *Markov Chains* (Cambridge, 1997) — rigorous treatment of discrete and continuous-time chains with clean proofs

3. **Brooks, Gelman, Jones, Meng** — *Handbook of Markov Chain Monte Carlo* (CRC Press, 2011) — comprehensive MCMC reference; includes HMC, NUTS, diagnostics

4. **Betancourt, M.** — "A Conceptual Introduction to Hamiltonian Monte Carlo" (arXiv, 2017) — outstanding intuitive treatment of HMC geometry

5. **Page et al.** — "The PageRank Citation Ranking: Bringing Order to the Web" (Stanford Technical Report, 1999) — original PageRank paper

6. **Rabiner, L.R.** — "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition" (Proc. IEEE, 1989) — classic HMM reference

7. **Song et al.** — "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021) — connects diffusion models to CTMC theory

8. **Welling & Teh** — "Bayesian Learning via Stochastic Gradient Langevin Dynamics" (ICML 2011) — introduces SGLD for large-scale Bayesian inference

9. **Sutton & Barto** — *Reinforcement Learning: An Introduction* (MIT Press, 2nd ed., 2018) — MDPs, Bellman equations, TD-learning; free online


---

## Appendix R: Spectral Methods for Large-Scale Markov Chains

### R.1 Lanczos Algorithm for Large Transition Matrices

For transition matrices arising in PageRank ($N \sim 10^{10}$ web pages), direct eigendecomposition is infeasible. The **power iteration** and **Lanczos algorithm** enable computing the top eigenvectors using only matrix-vector products.

**Power iteration for stationary distribution:**
- Memory: $O(N)$ for the distribution vector
- Per-iteration cost: $O(\text{nnz})$ (number of non-zeros in $P$) for sparse matrices
- Convergence in $O(\log(1/\varepsilon) / \log(1/|\lambda_2|))$ iterations

For the web graph, $|\lambda_2| \leq \alpha = 0.85$ (damping), so ~70 iterations give $\varepsilon = 10^{-8}$ precision.

**Randomised SVD:** For low-rank approximation of $P^n$ (needed for long-range transition probability estimation), randomised methods can approximate the top-$r$ singular vectors in $O(N r \log r)$ time — much cheaper than full SVD.

### R.2 Multi-Scale Markov Chains

For chains with multiple timescales (fast-mixing within clusters, slow-mixing between clusters), **aggregation-disaggregation** methods provide efficient algorithms:

1. **Aggregation:** Lump states within each cluster into a single meta-state. Compute the inter-cluster transition matrix $\bar{P}$.
2. **Disaggregation:** Compute the within-cluster stationary distributions $\pi^{(c)}$ for each cluster $c$.
3. **Combine:** $\pi_i = \bar{\pi}_c \cdot \pi^{(c)}_i$ for state $i$ in cluster $c$.

This decomposition is exact when the chain is "nearly lumpable" (within-cluster transitions are much faster than between-cluster). It's the foundation of multi-scale MCMC methods.

**For AI:** LLM attention patterns have multi-scale structure: attention heads attend to nearby tokens (fast mixing) and distant context (slow mixing). This multi-scale structure can be exploited for efficient long-context modelling.

### R.3 Reversibilisation

Any Markov chain can be made reversible by considering the **Doob $h$-transform** or by **multiplicative reversibilisation**:
$$\tilde{P}_{ij} = \frac{1}{2}(P_{ij} + \pi_j P_{ji} / \pi_i)$$

The resulting $\tilde{P}$ is reversible with the same stationary distribution $\pi$. Reversibilisation can speed up or slow down convergence depending on the chain — it destroys the directional bias that non-reversible chains use for faster mixing.

---

## Appendix S: Continuous-State Markov Chains

### S.1 Harris Chains

For Markov chains on continuous state spaces (like MCMC on $\mathbb{R}^d$), the discretestate ergodic theory extends with modifications. A Markov chain with transition kernel $K(x, \cdot)$ is:

- **$\phi$-irreducible** if there exists a $\sigma$-finite measure $\phi$ such that for any Borel set $A$ with $\phi(A) > 0$, we have $\sum_{n=1}^\infty K^n(x, A) > 0$ for all $x$.
- **Harris recurrent** if it is $\phi$-irreducible and returns to any set $A$ with $\phi(A) > 0$ with probability 1.
- **Positive Harris recurrent** if the chain is Harris recurrent and has a unique invariant probability measure $\pi$.

A positive Harris recurrent, aperiodic chain is called a **Harris ergodic** chain, and the ergodic theorem holds: time averages converge to $\pi$-expectations.

**For MCMC:** The Metropolis-Hastings chain on $\mathbb{R}^d$ with Gaussian proposal and a proper target density $\pi$ is Harris ergodic under mild regularity conditions. This justifies MCMC estimators for continuous posteriors.

### S.2 Geometric Ergodicity

A Markov chain is **geometrically ergodic** if there exist constants $C < \infty$ and $\rho < 1$ such that:
$$\|K^n(x, \cdot) - \pi\|_{\text{TV}} \leq C(x)\,\rho^n \quad \text{for all } x$$

Geometric ergodicity implies a central limit theorem for MCMC estimators:
$$\sqrt{n}\left(\frac{1}{n}\sum_{t=1}^n f(X_t) - \mathbb{E}_\pi[f]\right) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)$$

where $\sigma_f^2 = \text{Var}_\pi(f) + 2\sum_{k=1}^\infty \text{Cov}_\pi(f(X_0), f(X_k))$ is the asymptotic variance. This is the **Markov chain CLT** — it justifies asymptotic confidence intervals for MCMC estimates.

**For AI:** Geometric ergodicity of the Langevin algorithm for log-concave targets gives polynomial bounds on the number of gradient evaluations needed for $\varepsilon$-accurate posterior estimates. This is the theoretical basis for Bayesian deep learning via SGLD.

---

## Appendix T: Markov Chain Games and Nash Equilibria

### T.1 Stochastic Games

A **stochastic game** (Shapley, 1953) is an MDP with multiple agents. Two-player zero-sum stochastic games have:
- State space $\mathcal{S}$; action spaces $\mathcal{A}_1, \mathcal{A}_2$
- Transition $P(s'|s, a_1, a_2)$
- Reward $r(s, a_1, a_2)$ (player 1 maximises, player 2 minimises)

At a Nash equilibrium, each player's policy is a best response to the other's. The value function satisfies a minimax Bellman equation:
$$V^*(s) = \min_{a_2} \max_{a_1} \left[r(s,a_1,a_2) + \gamma \sum_{s'} P(s'|s,a_1,a_2) V^*(s')\right]$$

**For AI:** Multi-agent RL (MARL) with competing agents (e.g., AlphaGo, Libratus for poker) uses stochastic game theory. RLHF with adversarial reward models is a stochastic game between the policy and the worst-case reward model.

### T.2 Markov Perfect Equilibrium

In dynamic games, a **Markov perfect equilibrium** (MPE) is a Nash equilibrium where strategies depend only on the current state (Markov property on the strategy). MPE is the game-theoretic analogue of the optimal policy in MDPs.

Computing MPE requires solving a system of coupled Bellman equations — one per player. For two-player zero-sum games, this reduces to linear programming (minimax theorem). For general-sum games, this is PPAD-complete.

---

## Appendix U: Temporal Difference Learning and Martingales

Connecting back to [§06 Stochastic Processes §3.7](../06-Stochastic-Processes/notes.md), the **TD(0) algorithm** is:
$$V(s_t) \leftarrow V(s_t) + \alpha_t [r_t + \gamma V(s_{t+1}) - V(s_t)]$$

The update direction $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the **TD error**. Under the true value function $V^*$, $\mathbb{E}[\delta_t | s_t] = 0$ (martingale difference). TD learning is a **stochastic approximation** algorithm for finding the fixed point of the Bellman operator $T^\pi V = r^\pi + \gamma P^\pi V$.

**Convergence proof via martingale theory:** Write $V_t = V^* + \varepsilon_t$ (error from true value). Then:
$$\varepsilon_{t+1}(s_t) = (1 - \alpha_t)\varepsilon_t(s_t) + \alpha_t[\gamma P^\pi \varepsilon_t(s_t) + M_{t+1}]$$

where $M_{t+1} = r_t + \gamma V^*(s_{t+1}) - V^*(s_t)$ is a martingale difference noise term. Under the Robbins-Monro conditions $\sum\alpha_t = \infty$, $\sum\alpha_t^2 < \infty$, the ODE method (Borkar, 2008) shows $\varepsilon_t \to 0$ a.s.

---

## Appendix V: Practical MCMC Checklist for AI/ML

**Before running MCMC:**
- [ ] Identify the target: is $\pi(\theta | \mathcal{D})$ proper? Is $\log\pi$ evaluable efficiently?
- [ ] Choose algorithm: log-concave → Langevin/HMC; multimodal → parallel tempering; discrete → Gibbs
- [ ] Set proposal scale: tune to ~23% acceptance (MH) or ~65% (HMC)
- [ ] Run multiple chains ($\geq 4$) from diverse starting points

**During MCMC:**
- [ ] Monitor trace plots for signs of non-stationarity (trends, sticking)
- [ ] Track acceptance rate: should be in target range
- [ ] Compute $\hat{R}$ after every 1000 iterations; stop when $\hat{R} \leq 1.01$

**After MCMC:**
- [ ] Discard burn-in (first 50% as rule of thumb)
- [ ] Compute ESS per parameter; target ESS $\geq 400$ for reliable estimates
- [ ] Check posterior predictive checks: do simulated data match observed data?
- [ ] Report: posterior mean, posterior std, 95% credible interval, ESS, $\hat{R}$

**Common pathologies:**
- **Divergent transitions (HMC):** posterior geometry has sharp ridges; reparameterise
- **Low ESS:** chain mixes slowly; increase step size, use HMC, or apply preconditioning
- **$\hat{R} > 1.1$:** chains haven't agreed; run longer, use better initialisation
- **Bimodal trace plots:** chain stuck between modes; use parallel tempering


---

## Appendix W: Worked Computation — Two-State Chain

### Full Analysis

Let $P = \begin{pmatrix}0.7 & 0.3 \\ 0.5 & 0.5\end{pmatrix}$, starting from $\mu^{(0)} = (1, 0)$.

**Step 1: Eigenvalues.** Characteristic polynomial: $\det(P - \lambda I) = (0.7-\lambda)(0.5-\lambda) - 0.3 \times 0.5 = \lambda^2 - 1.2\lambda + 0.20 = (\lambda-1)(\lambda-0.2) = 0$. Eigenvalues: $\lambda_1=1, \lambda_2=0.2$.

**Step 2: Stationary distribution.** From $\pi P = \pi$: $0.7\pi_1 + 0.5\pi_2 = \pi_1 \Rightarrow \pi_2 = 0.6\pi_1$. With $\pi_1+\pi_2=1$: $\pi = (5/8, 3/8)$.

**Step 3: $n$-step distribution.** Using the spectral decomposition:
$$P^n = \pi + (0.2)^n \cdot \text{(correction term)}$$

Explicitly: $P^n = \begin{pmatrix}5/8 & 3/8 \\ 5/8 & 3/8\end{pmatrix} + (0.2)^n\begin{pmatrix}3/8 & -3/8 \\ -5/8 & 5/8\end{pmatrix}$

So $\mu^{(n)} = \mu^{(0)}P^n = (5/8 + 3/8 \cdot (0.2)^n,\ 3/8 - 3/8 \cdot (0.2)^n)$.

**Step 4: TV distance.** $\|\mu^{(n)} - \pi\|_{\text{TV}} = \frac{1}{2}|{3/8 \cdot (0.2)^n}| + \frac{1}{2}|{3/8 \cdot (0.2)^n}| = 3/8 \cdot (0.2)^n$.

At $n=5$: $3/8 \times 0.00032 = 0.00012$ — essentially converged. Spectral gap $= 1 - 0.2 = 0.8$, so mixing is fast.

**Step 5: Verify detailed balance.**
$$\pi_1 P_{12} = 5/8 \times 0.3 = 0.1875, \quad \pi_2 P_{21} = 3/8 \times 0.5 = 0.1875 \checkmark$$

The chain is reversible with this stationary distribution.

---

## Appendix X: Advanced Topics

### X.1 Geometric Random Walks for Volume Computation

The **ball walk** and **Gaussian walk** are Markov chains used to estimate the volume of convex bodies. Starting from a point $x$ in a convex body $K$, the chain moves to a uniformly random point within a ball of radius $r$ centred at $x$ (rejecting if outside $K$). The stationary distribution is uniform on $K$.

Mixing time analysis: $t_{\text{mix}} = O(n^2 \text{poly}(1/\varepsilon))$ for the ball walk, where $n$ is the dimension. This gives a polynomial-time randomised algorithm for volume computation (Dyer-Frieze-Kannan theorem).

**For AI:** High-dimensional sampling problems (Bayesian inference with $d \sim 10^9$ parameters) require understanding how mixing time scales with dimension. Geometric random walks provide the fundamental tools.

### X.2 Markov Chain Tree Theorem

For a strongly connected directed graph with transition matrix $P$, the stationary distribution can be computed via spanning trees:
$$\pi_i = \frac{\tau_i}{\sum_j \tau_j}$$

where $\tau_i$ is the sum of weights of all spanning trees directed toward node $i$ (arborescences rooted at $i$). This is the **Matrix-Tree theorem** (Kirchhoff 1847) extended to directed graphs.

**Implication:** The stationary distribution has a combinatorial formula in terms of the graph structure. This provides an alternative to eigendecomposition that can be faster for sparse graphs.

### X.3 Lumping and Aggregation

A partition $\{C_1,\ldots,C_m\}$ of $\mathcal{S}$ is **lumpable** with respect to $P$ if for any two states $i, j$ in the same class $C_k$, the aggregated transitions $\sum_{\ell \in C_m} P_{i\ell} = \sum_{\ell \in C_m} P_{j\ell}$ for all classes $C_m$. In this case, the quotient chain $\bar{P}$ on $\{C_1,\ldots,C_m\}$ is a well-defined Markov chain.

Lumping enables dimensionality reduction: replace a large chain with a smaller aggregated chain. The stationary distribution of the lumped chain aggregates that of the original.

**For AI:** Grouping similar states (e.g., tokens with similar embeddings) can define a lumped chain that captures the high-level dynamics of the LLM's token distribution, enabling efficient analysis of LLM behaviour at scale.

---

## Appendix Y: Connections to Physics

### Y.1 Statistical Mechanics and Gibbs Measures

The **Gibbs distribution** $\pi(x) \propto e^{-H(x)/(k_BT)}$ (Boltzmann distribution) is the equilibrium distribution of a physical system with Hamiltonian $H$ at temperature $T$. This is the canonical target distribution for MCMC in physics.

The Metropolis algorithm was originally designed to sample from Gibbs distributions of particle systems (Metropolis et al., 1953). MCMC methods in ML directly descended from statistical mechanics — including the energy-based models (EBMs) that predate modern deep learning.

**Temperature annealing:** Starting at high $T$ (flat, easy-to-sample distribution) and gradually cooling to $T=0$ (greedy optimisation) is **simulated annealing** — a probabilistic optimisation algorithm. The connection between MCMC sampling and optimisation at $T \to 0$ is the bridge between Bayesian inference and MAP estimation.

### Y.2 Detailed Balance and Thermodynamic Equilibrium

Detailed balance in physics is called **microscopic reversibility** or **time-reversal symmetry**. A physical system is at thermodynamic equilibrium iff its microscopic dynamics satisfy detailed balance. Systems out of equilibrium (driven by energy flows) violate detailed balance and maintain directed probability currents.

**Non-equilibrium steady states:** Some biological systems (motor proteins, gene regulatory networks) maintain non-reversible Markov chains in steady state — driven by ATP hydrolysis. These are "out-of-equilibrium" in the thermodynamic sense. Non-reversible MCMC is inspired by this physics.

### Y.3 Renormalization Group and Multi-Scale Chains

The **renormalization group** (Wilson, 1971) is a technique in physics for coarse-graining: replacing fine-grained microscopic degrees of freedom with effective coarse-grained ones. This is mathematically analogous to the lumping/aggregation of Markov chains.

In ML, **knowledge distillation** (training a small model to mimic a large model) is a form of renormalization: the small model's transition matrix approximates the "effective" dynamics of the large model at a coarser scale.


---

## Appendix Z: Glossary

**Absorbing state.** A state $i$ with $P_{ii}=1$; the chain stays forever once it arrives. Every absorbing state is recurrent.

**Aperiodic.** A state $i$ with period $d(i)=1$. A chain is aperiodic if all states are aperiodic. Aperiodicity + irreducibility + positive recurrence = ergodicity.

**Birth-death chain.** A Markov chain on $\mathbb{Z}_{\geq 0}$ with only nearest-neighbour transitions ($|i-j| \leq 1$). Always reversible; stationary distribution computed via detailed balance.

**Chapman-Kolmogorov equations.** $P^{(m+n)} = P^m P^n$: the $n+m$ step transition matrix factors as a product of $m$- and $n$-step matrices.

**Communicating class.** A maximal set of states $C$ where $i \leftrightarrow j$ for all $i,j \in C$. Partitions the state space into equivalence classes.

**Conductance (Cheeger constant).** $h = \min_{S: \pi(S) \leq 1/2} Q(S, S^c)/\pi(S)$ where $Q(S,S^c)$ is the probability flux from $S$ to $S^c$ at stationarity. Bounds: $h^2/2 \leq \text{gap} \leq 2h$.

**Coupling.** A joint process $(X_t, Y_t)$ where each marginal is a valid Markov chain with transition $P$. Used to bound TV distance via $\|\mu P^n - \nu P^n\|_{\text{TV}} \leq P(\tau > n)$ where $\tau$ is the coalescence time.

**CTMC.** Continuous-time Markov chain. Holds in each state for an exponential time, then jumps. Specified by a generator matrix $Q$ with $P(t) = e^{Qt}$.

**Detailed balance.** $\pi_i P_{ij} = \pi_j P_{ji}$: probability flux is equal in both directions at stationarity. Sufficient (not necessary) for $\pi$ to be stationary.

**Doubly stochastic matrix.** A matrix where rows AND columns all sum to 1. Stationary distribution is uniform.

**Embedded chain.** The discrete-time chain of states visited by a CTMC (ignoring holding times). Jump probabilities $\tilde{P}_{ij} = q_{ij}/q_i$.

**Ergodic chain.** Irreducible + positive recurrent + aperiodic. Convergence to unique stationary distribution is guaranteed.

**Ergodic theorem.** For an ergodic chain: $\frac{1}{n}\sum_{k=0}^{n-1} f(X_k) \to \mathbb{E}_\pi[f]$ a.s. Justifies MCMC estimation.

**Fundamental matrix.** $N = (I-Q)^{-1}$ for absorbing chains, where $Q$ is the sub-matrix of transitions between transient states. $N_{ij}$ = expected visits to state $j$ starting from $i$ before absorption.

**Generator matrix (Q-matrix).** $Q_{ij} = q_{ij} \geq 0$ for $i \neq j$ (jump rates), $Q_{ii} = -\sum_{j \neq i} q_{ij}$. Rows sum to zero. CTMC transition: $P(t) = e^{Qt}$.

**Gibbs sampling.** MCMC algorithm that updates one coordinate at a time from its full conditional distribution. Special case of MH with acceptance ratio 1.

**Harmonic function.** $h(i) = (Ph)(i)$: equals its own expected value one step ahead. The martingale $h(X_n)$ is constant in expectation.

**HMC (Hamiltonian Monte Carlo).** MCMC using gradient information to make long-range moves. Augments the state with momentum; uses leapfrog integration to simulate Hamiltonian dynamics.

**HMM (Hidden Markov Model).** A Markov chain $(Z_t)$ of hidden states with observations $(X_t)$ emitted from the current state. Algorithms: forward, backward, Viterbi, Baum-Welch.

**Irreducible chain.** Every state is accessible from every other: $i \to j$ for all $i,j$. Ensures no trapping in subsets.

**Lazy chain.** $P' = (I+P)/2$: stays put with prob 1/2, otherwise transitions. Aperiodic by construction; eigenvalues non-negative.

**Lumpable partition.** A partition where all states in the same class have identical transition probabilities to every other class. Enables dimensionality reduction.

**Markov chain.** A sequence of random variables where the future depends on the past only through the present: $P(X_{n+1}|X_0,\ldots,X_n) = P(X_{n+1}|X_n)$.

**Markov property.** $X_{n+1} \perp (X_0,\ldots,X_{n-1}) | X_n$: conditional independence of future from past given present.

**MDP (Markov Decision Process).** Extension of Markov chain with actions; the policy induces a Markov chain; the Bellman equation characterises the value function.

**Metropolis-Hastings.** MCMC algorithm: propose $\theta' \sim q(\cdot|\theta)$, accept with probability $\min(1, \pi(\theta')q(\theta|\theta')/(\pi(\theta)q(\theta'|\theta)))$. Correct by detailed balance.

**Mixing time.** $t_{\text{mix}}(\varepsilon) = \min\{t: \max_x \|P^t(x,\cdot)-\pi\|_{\text{TV}} \leq \varepsilon\}$. Bounded by $O(\log(\pi_{\min}^{-1})/\text{gap})$.

**Null recurrent.** Recurrent state with infinite mean return time ($\mu_i = \infty$). Common in countable chains (1D SRW); no normalisable stationary distribution.

**PageRank.** Stationary distribution of the random surfer Markov chain on the web graph. Computed by power iteration.

**Period.** $d(i) = \gcd\{n \geq 1: P^{(n)}_{ii} > 0\}$. Period 1 = aperiodic.

**Perron-Frobenius theorem.** For primitive stochastic $P$: eigenvalue 1 is unique and dominant; unique stationary distribution $\pi > 0$; $P^n \to \mathbf{1}\pi^T$ geometrically.

**Positive recurrent.** Recurrent state with finite mean return time ($\mu_i < \infty$). In a finite irreducible chain, all states are positive recurrent.

**Reversible chain.** Satisfies detailed balance; the time-reversed chain equals the forward chain. Equivalent to: $P$ is self-adjoint in $L^2(\pi)$.

**SGLD.** Stochastic gradient Langevin dynamics: SGD + Gaussian noise. Approximates Bayesian inference at scale.

**Spectral gap.** $\text{gap} = 1 - |\lambda_2|$ for reversible chains. Determines mixing rate: $t_{\text{mix}} = \Theta(1/\text{gap})$.

**Stationary distribution.** $\pi$ with $\pi P = \pi$, $\pi_i \geq 0$, $\sum\pi_i = 1$. The equilibrium distribution; unique for ergodic chains.

**Stochastic matrix.** $P_{ij} \geq 0$ and $\sum_j P_{ij} = 1$ for all $i$. The transition matrix of a DTMC.

**Transient state.** State $i$ with $f_i = P(\text{return to } i) < 1$. The chain leaves and never returns with positive probability; $\sum_n P^{(n)}_{ii} < \infty$.

**Total variation (TV) distance.** $\|\mu-\nu\|_{\text{TV}} = \sup_A|\mu(A)-\nu(A)| = \frac{1}{2}\sum_i|\mu_i-\nu_i|$. The mixing distance for Markov chains.

**Viterbi algorithm.** Dynamic programming for the most likely hidden state sequence in an HMM. Complexity $O(T|\mathcal{S}|^2)$.


---

## Appendix AA: Practice Problems with Solutions

### AA.1 Absorption Probability Calculation

**Problem.** A gambler starts with $\$3$ and plays at a casino. Each round they win $\$1$ (prob 0.45) or lose $\$1$ (prob 0.55). The game ends when they reach $\$0$ (ruin) or $\$6$ (goal). Find: (a) probability of reaching $\$6$ from $\$3$, (b) expected duration of the game.

**Solution.** Let $p = 0.45$, $q = 0.55$, $r = q/p = 0.55/0.45 = 11/9$. Using the absorption formula for biased walks with absorbing boundaries at 0 and $N=6$:

(a) $P(\text{reach 6} | \text{start at 3}) = \frac{1 - r^3}{1 - r^6} = \frac{1 - (11/9)^3}{1 - (11/9)^6} \approx \frac{1-1.521}{1-5.354} \approx \frac{-0.521}{-4.354} \approx 0.120$

Only 12% chance of reaching the goal from the midpoint — the bias significantly hurts the gambler.

(b) Expected duration: $\mathbb{E}[\tau] = \frac{1}{q-p}\left[6 \cdot \frac{1 - r^3}{1-r^6} - 3\right] \approx \frac{1}{0.1}[6 \times 0.120 - 3] = 10 \times (-2.28) = ...$

Actually for biased walks: $\mathbb{E}[\tau|X_0=a] = \frac{a}{q-p} - \frac{N}{q-p} \cdot P(\text{win})$. With $q-p=0.1$: $\mathbb{E}[\tau] = 3/0.1 - 6/0.1 \times 0.120 = 30 - 7.2 = 22.8$ rounds.

### AA.2 PageRank Mini-Example with Computation

**Problem.** Three web pages: 1 links to {2,3}, 2 links to {3}, 3 links to {1}. Find PageRank with $\alpha=0.8$.

**Adjacency matrix:**
$$A = \begin{pmatrix}0&1&1\\0&0&1\\1&0&0\end{pmatrix}$$

Out-degrees: $d_1=2, d_2=1, d_3=1$.

**Transition matrix without damping:**
$$P_0 = \begin{pmatrix}0&1/2&1/2\\0&0&1\\1&0&0\end{pmatrix}$$

**Damped matrix ($\alpha=0.8$):**
$$P = 0.8 P_0 + 0.2 \times \frac{1}{3}J = \begin{pmatrix}1/15&2/5+1/15&2/5+1/15\\1/15&1/15&4/5+1/15\\4/5+1/15&1/15&1/15\end{pmatrix}$$

Solving $\pi P = \pi$ with $\pi_1+\pi_2+\pi_3=1$: by symmetry of the network (1 has out-degree 2, others 1) and solving the $3\times3$ system: $\pi \approx (0.390, 0.211, 0.399)$.

Page 3 has highest PageRank — it's the target of all paths through the cycle $1\to2\to3\to1$, plus direct link from 1.

### AA.3 HMM Forward Algorithm Computation

**Problem.** HMM with 2 states H,L and 2 observations A,B.
- $\pi_0 = (0.5, 0.5)$, $T = \begin{pmatrix}0.7&0.3\\0.4&0.6\end{pmatrix}$
- Emission: $E_H = (0.9, 0.1)$, $E_L = (0.2, 0.8)$ (A and B respectively)
- Observation: A, B

**Forward algorithm:**

$t=1$, obs=A: $\alpha_1(H) = 0.5 \times 0.9 = 0.45$, $\alpha_1(L) = 0.5 \times 0.2 = 0.10$

$t=2$, obs=B:
$$\alpha_2(H) = 0.1 \times [0.45 \times 0.7 + 0.10 \times 0.4] = 0.1 \times [0.315 + 0.04] = 0.0355$$
$$\alpha_2(L) = 0.8 \times [0.45 \times 0.3 + 0.10 \times 0.6] = 0.8 \times [0.135 + 0.06] = 0.156$$

Likelihood: $P(\text{obs}=(A,B)) = \alpha_2(H) + \alpha_2(L) = 0.0355 + 0.156 = 0.1915$.

Posterior: $P(Z_2=H|\text{obs}) = 0.0355/0.1915 \approx 18.5\%$; $P(Z_2=L|\text{obs}) \approx 81.5\%$.

Interpretation: Observing (A,B) — first high-emission then low-emission — suggests state L at time 2.

---

## Appendix BB: Index of Key Theorems

| Theorem / Result | Where Proved | Key Implication |
| --- | --- | --- |
| Perron-Frobenius | §4.2, App. I.1 | Unique stationary distribution for ergodic chains |
| Ergodic theorem | §3.5, App. E | Time averages converge to $\mathbb{E}_\pi[f]$; justifies MCMC |
| Chapman-Kolmogorov | §2.3 | $n$-step transitions: $P^{m+n} = P^m P^n$ |
| Convergence theorem | §4.3 | $\|\mu^{(n)} - \pi\|_{\text{TV}} \to 0$ for ergodic chains |
| Detailed balance $\Rightarrow$ stationarity | §5.1, App. E.2 | Sufficient condition; easier to verify |
| Spectral gap bound | §6.3 | $t_{\text{mix}} \leq \log(\pi_{\min}^{-1}/\varepsilon)/\text{gap}$ |
| Cheeger inequality | §6.3, App. I.3 | $h^2/2 \leq \text{gap} \leq 2h$ |
| Coupling inequality | §6.1, App. I.2 | $\|P^n(x,\cdot)-\pi\|_{\text{TV}} \leq P(\tau > n)$ |
| MH correctness | §8.2 | Detailed balance implies $\pi$ is stationary |
| MH acceptance optimality | §8.2 | ~23% acceptance optimal in high dimensions |
| Markov chain CLT | App. S.2 | $\sqrt{n}(\text{MCMC avg} - \mathbb{E}_\pi[f]) \to \mathcal{N}(0,\sigma_f^2)$ |
| Dirichlet problem / OST | App. A | Value function = expected boundary value |
| Mean return time formula | §4.1 | $\pi_i = 1/\mu_i$; occupancy = reciprocal return time |
| Absorption formula | §3.4 | $B = (I-Q)^{-1}R$; absorption probabilities from fundamental matrix |
| Viterbi optimality | §9.4 | DP gives exact most likely path in $O(T|\mathcal{S}|^2)$ |


---

## Appendix CC: Mixing Times for Specific Chains

A reference table of known mixing time results for important chains:

| Chain | State Space | Mixing Time $t_{\text{mix}}$ | Technique |
| --- | --- | --- | --- |
| Two-state chain | $\{0,1\}$ | $O(1/(p+q))$ | Eigenvalue $1-(p+q)$ |
| Random walk on path $P_n$ | $\{0,\ldots,n\}$ | $\Theta(n^2)$ | Eigenvalues $\cos(k\pi/n)$ |
| Random walk on cycle $C_n$ | $\mathbb{Z}/n\mathbb{Z}$ | $\Theta(n^2)$ | Fourier analysis |
| Random walk on hypercube $\{0,1\}^d$ | $\{0,1\}^d$ | $\Theta(d\log d)$ | Coupling (coupon collector) |
| Random walk on complete graph $K_n$ | $\{1,\ldots,n\}$ | $\Theta(1)$ | Spectral gap $= 1-1/(n-1) \approx 1$ |
| Random walk on expander | $\{1,\ldots,n\}$ | $\Theta(\log n)$ | Spectral gap $= \Omega(1)$ |
| Glauber dynamics (Ising, high $T$) | $\{-1,+1\}^n$ | $O(n\log n)$ | Coupling or spectral gap |
| Glauber dynamics (Ising, $T < T_c$) | $\{-1,+1\}^n$ | $\exp(\Omega(n))$ | Phase transition barrier |
| MH on $\mathcal{N}(0, I_d)$ (Gaussian RW) | $\mathbb{R}^d$ | $\Theta(d)$ | Optimal scaling |
| HMC on $\mathcal{N}(0, I_d)$ | $\mathbb{R}^d$ | $O(d^{1/4})$ | Leapfrog integrator analysis |
| PageRank (damping $\alpha$) | Web pages | $O(\log(N)/\log(1/\alpha))$ | Second eigenvalue $\leq \alpha$ |

Key observations:
- **Dimension scaling:** Random walk MH scales as $O(d)$ (diffusive), HMC as $O(d^{1/4})$ (much better)
- **Phase transitions:** Near critical temperature, MCMC for Ising model has exponential mixing time — a computational hard phase
- **Expander graphs:** $\Omega(1)$ spectral gap means $O(\log n)$ mixing — the best possible for non-trivial graphs

---

## Appendix DD: Software Ecosystem

The following Python libraries implement the algorithms in this section:

**MCMC and Probabilistic Programming:**
- `PyMC` — full-featured Bayesian modelling with NUTS/HMC sampler
- `Stan` (via `cmdstanpy`/`pystan`) — the gold standard for Bayesian modelling; generates C++ code for fast sampling
- `NumPyro` — JAX-based; supports GPU-accelerated HMC and NUTS
- `blackjax` — modular MCMC library (JAX); easy to extend with custom kernels

**Markov Chain Simulation:**
- `networkx` — graph operations; random walks on graphs
- `numpy` — transition matrix operations; power iteration; eigendecomposition

**Hidden Markov Models:**
- `hmmlearn` — sklearn-compatible HMM; Baum-Welch training, Viterbi decoding
- `pomegranate` — probabilistic models including HMMs with GPU support

**Reinforcement Learning (MDP):**
- `gymnasium` (formerly OpenAI Gym) — RL environments as MDPs
- `stable-baselines3` — RL algorithms (PPO, SAC, TD3) implementing policy optimisation

**Diffusion Models:**
- `diffusers` (HuggingFace) — implements DDPM, DDIM, and score-based models as CTMCs

---

## Appendix EE: Common Markov Chain Constructions

**Ehrenfest Model:** $N$ balls in two urns, uniform random ball moved at each step. State = number of balls in urn 1. Birth-death chain; stationary distribution: Binomial$(N, 1/2)$. Models diffusion/thermodynamic equilibrium.

**Polya Urn:** Start with $a$ red and $b$ blue balls; each step add a ball of the drawn colour. Path-dependent; NOT Markov without extended state. But $X_n = \text{fraction red}$ satisfies $\mathbb{E}[X_{n+1}|X_n] = X_n$ — a martingale.

**Bernoulli-Laplace Diffusion:** $N$ balls total, $k$ red and $N-k$ blue, split between two urns of size $N/2$. State = number of red in urn 1. Symmetric birth-death chain; used to model genetics and combinatorics.

**Random Transposition Walk:** On $S_n$ (permutation group of $n$ elements), pick two positions uniformly and swap. State space: $S_n$ (size $n!$). Mixing time: $\Theta(n\log n)$. Used to model card shuffling (Bayer-Diaconis theorem).

**Metropolis on Graphs:** For any graph $G$ and target $\pi$, the Metropolis chain with uniform proposal on neighbours satisfies detailed balance with respect to $\pi$. Useful for sampling graph colorings, independent sets, etc.

**Skip-free chains:** Transition matrices with $P_{ij} = 0$ for $j < i-1$ (can only go up by 1 or down arbitrarily). Examples: GI/M/1 queues. Efficient algorithms via matrix-analytic methods.


---

## Appendix FF: Spectral Graph Theory Connection

### FF.1 Normalized Laplacian and Random Walks

For an undirected graph $G=(V,E)$ with degree matrix $D$ and adjacency matrix $A$, the random walk transition matrix is $P = D^{-1}A$. The **normalized Laplacian** is:
$$\mathcal{L} = I - D^{-1/2}AD^{-1/2} = I - D^{-1/2}PD^{1/2}$$

The eigenvalues of $\mathcal{L}$ are $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_N \leq 2$. The spectral gap of the random walk is $\lambda_2(\mathcal{L})$.

**Cheeger inequality for graphs:** $\lambda_2/2 \leq h(G) \leq \sqrt{2\lambda_2}$ where $h(G) = \min_{S:|S|\leq|V|/2} |E(S,\bar{S})|/(d_{\min}|S|)$ is the edge expansion of the graph.

**Expander graphs:** A $d$-regular expander has $\lambda_2(\mathcal{L}) = \Omega(1)$ (bounded below independently of $N$). Examples: Ramanujan graphs achieve $\lambda_2 = 1 - 2\sqrt{d-1}/d$ (optimal). Random regular graphs are expanders with high probability.

### FF.2 Graph-Based ML and Markov Chains

**Graph attention networks:** GAT attention coefficients $e_{ij} = a(\mathbf{W}h_i, \mathbf{W}h_j)$ (learnable) create a data-dependent transition matrix over the graph. The spectral properties of this attention-weighted adjacency matrix determine the GNN's ability to propagate information.

**Label propagation:** Semi-supervised learning via label propagation: $F \leftarrow \alpha PF + (1-\alpha)Y$ where $P=D^{-1}A$ is the random walk matrix, $F$ are labels, $Y$ are known labels, and $\alpha$ is a damping parameter. This is exactly the PageRank recursion applied to labels — convergence follows from the same Perron-Frobenius analysis.

**Spectral clustering:** The $k$ smallest eigenvectors of $\mathcal{L}$ embed the graph into $\mathbb{R}^k$ such that well-connected clusters map to nearby points. This exploits the fact that slow-mixing chains (small spectral gap) correspond to loosely connected clusters.

---

## Appendix GG: Long-Range Dependencies and Markov Approximations

### GG.1 Finite-Order Markov Chains

A **$k$th-order Markov chain** satisfies:
$$P(X_n | X_0,\ldots,X_{n-1}) = P(X_n | X_{n-k},\ldots,X_{n-1})$$

This can always be reduced to a first-order chain on the augmented state $(X_{n-k+1},\ldots,X_n)$ with state space size $|\mathcal{S}|^k$.

**Trade-off:** Higher-order captures more dependencies but exponentially increases state space. N-gram language models are $k$th-order Markov chains on the vocabulary — trigram models ($k=2$) are common in classical NLP.

**For transformers:** A transformer with context length $L$ has an effective $L$th-order Markov chain for token generation. The KV cache stores the sufficient statistics (the last $L$ tokens) — first-order Markov in the augmented state space.

### GG.2 Variable-Length Markov Chains (VLMC)

**VLMCs** use a context tree to specify the order dynamically: some contexts require deep history, others only shallow. The transition probability $P(X_n | X_{n-1},\ldots) = P(X_n | \text{suffix}(X_{n-1},\ldots,X_{n-k}))$ where $k$ depends on the context.

VLMCs provide compact representations of processes with heterogeneous memory. Related to suffix trees and compressed suffix arrays — efficient data structures for sequential data.

### GG.3 Hidden Non-Markovian Processes

Many real processes are non-Markovian but become Markovian on an augmented state space. Examples:
- **AR($p$)** process: $X_n = \sum_{k=1}^p \phi_k X_{n-k} + \varepsilon_n$. First-order Markov on $(X_n,\ldots,X_{n-p+1})$.
- **ARMA($p,q$)**: Moving average component creates non-Markovian behaviour; state space includes latent noise terms.
- **LLM with KV cache**: First-order Markov on the KV cache state; non-Markovian on just the last token.

---

## Appendix HH: Excursion Theory and Renewal Processes

### HH.1 Renewal Theory

A **renewal process** is a sequence of times $T_1, T_2, \ldots$ where $T_k - T_{k-1}$ are iid positive random variables (inter-renewal times). The process $N_t = \max\{k : T_k \leq t\}$ counts renewals up to time $t$.

**Renewal theorem:** $\lim_{t\to\infty} N_t/t = 1/\mu$ where $\mu = \mathbb{E}[T_2-T_1]$ is the mean inter-renewal time. This is the law of large numbers for renewal processes.

**Connection to Markov chains:** The times of visits to a fixed recurrent state $i$ form a renewal process with inter-renewal distribution = return time distribution. The renewal theorem gives $\pi_i = 1/\mu_i$ (mean return time formula).

### HH.2 Excursions from a State

An **excursion** from state $i$ is the portion of the trajectory between two consecutive visits to $i$. Excursions are iid by the strong Markov property. The distribution of the excursion — its length, states visited, and path structure — encodes rich information about the chain.

**For AI:** In RLHF, the "excursions" of the LM's generation from topic to topic form a renewal-like structure. Understanding the statistics of these excursions (how long the LM stays on-topic, how it transitions between topics) is important for reward model design.


---

## Appendix II: Summary Diagrams

```
MARKOV CHAIN THEORY SUMMARY
════════════════════════════════════════════════════════════════════════

  CHAIN PROPERTIES                   IMPLICATIONS
  ────────────────                   ────────────
  Irreducible ─────────────────────→ Every state reachable from every other
  + Positive Recurrent ─────────────→ Stationary distribution π exists
  + Aperiodic ──────────────────────→ π unique; P^n → 1·π^T (convergence)
  (= Ergodic)

  Reversible ──────────────────────→ Detailed balance π_i P_ij = π_j P_ji
                                     Real eigenvalues; symmetric in L²(π)
                                     MH, Gibbs, HMC all reversible

  MIXING SPEED HIERARCHY
  ───────────────────────
  Spectral gap large ──────────────→ Fast mixing (t_mix = O(1/gap))
  Cheeger constant large ──────────→ Large spectral gap (Cheeger inequality)
  Non-reversible + directed ────────→ Potentially faster than any reversible
  HMC gradient steps ──────────────→ O(d^{1/4}) vs O(d) for random walk

  COMPUTING π
  ───────────
  Small N: solve π P = π, ‖π‖₁ = 1 (linear system)
  Large N: power iteration π^{k+1} = π^{k} P (PageRank)
  Continuous space: MCMC (Metropolis-Hastings, HMC)
  Birth-death: explicit formula via detailed balance

  ML APPLICATIONS
  ───────────────
  Language generation ─────────────→ Markov chain on vocabulary (or context)
  PageRank ────────────────────────→ Stationary dist of web graph walk
  MCMC ────────────────────────────→ Bayesian posterior sampling
  RL (MDP) ────────────────────────→ Policy induces Markov chain; Bellman eqn
  HMM ─────────────────────────────→ Latent Markov chain with observations
  Diffusion models ────────────────→ Forward = CTMC; reverse = learned CTMC
  GNNs ────────────────────────────→ Message passing = transition matrix power

════════════════════════════════════════════════════════════════════════
```

