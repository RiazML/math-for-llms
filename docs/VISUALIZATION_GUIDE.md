# 🎨 Visualization Guide

> How to visualize mathematical concepts for better understanding.

---

## Table of Contents

1. [Visualization Tools](#visualization-tools)
2. [Linear Algebra Visualizations](#linear-algebra-visualizations)
3. [Calculus Visualizations](#calculus-visualizations)
4. [Probability Visualizations](#probability-visualizations)
5. [Optimization Visualizations](#optimization-visualizations)
6. [Information Theory Visualizations](#information-theory-visualizations)
7. [Graph Theory Visualizations](#graph-theory-visualizations)
8. [Advanced ML Visualizations](#advanced-ml-visualizations)
9. [Code Templates](#code-templates)

---

## Visualization Tools

### Recommended Libraries

```python
# Essential imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# For interactive visualizations
import plotly.graph_objects as go
import plotly.express as px

# For animations
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
```

### Setup for Clean Plots

```python
# Matplotlib style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# For LaTeX-style labels
plt.rcParams['text.usetex'] = False  # Set True if LaTeX installed
plt.rcParams['mathtext.fontset'] = 'stix'
```

---

## Linear Algebra Visualizations

### 1. Vectors in 2D

```python
def plot_vectors_2d(vectors, labels=None, colors=None, origin=True):
    """
    Plot 2D vectors from origin.

    Parameters
    ----------
    vectors : list of tuples or np.ndarray
        List of (x, y) vectors
    labels : list of str, optional
        Labels for each vector
    colors : list of str, optional
        Colors for each vector
    origin : bool
        Whether to show origin
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(vectors)))

    for i, v in enumerate(vectors):
        label = labels[i] if labels else f'v{i+1}'
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color=colors[i], label=label, width=0.015)

    # Set equal aspect ratio and grid
    all_coords = np.array(vectors)
    max_val = np.abs(all_coords).max() * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return fig, ax

# Example usage
vectors = [(3, 2), (1, 4), (-2, 1)]
labels = ['a', 'b', 'c']
plot_vectors_2d(vectors, labels)
plt.title('2D Vectors')
plt.show()
```

### 2. Linear Transformations

```python
def visualize_transformation(A, n_points=20):
    """
    Visualize how a matrix transforms the unit square.

    Parameters
    ----------
    A : np.ndarray
        2x2 transformation matrix
    n_points : int
        Number of points per side of grid
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create unit square grid
    t = np.linspace(0, 1, n_points)

    # Grid points
    grid_x = []
    grid_y = []

    # Horizontal lines
    for y_val in np.linspace(0, 1, 5):
        grid_x.extend(t)
        grid_y.extend([y_val] * n_points)

    # Vertical lines
    for x_val in np.linspace(0, 1, 5):
        grid_x.extend([x_val] * n_points)
        grid_y.extend(t)

    grid = np.array([grid_x, grid_y])

    # Transform grid
    transformed = A @ grid

    # Plot original
    axes[0].scatter(grid[0], grid[1], c='blue', s=1, alpha=0.5)
    axes[0].set_xlim(-0.5, 2)
    axes[0].set_ylim(-0.5, 2)
    axes[0].set_aspect('equal')
    axes[0].set_title('Original Grid')
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    axes[0].axvline(x=0, color='k', linewidth=0.5)

    # Plot transformed
    axes[1].scatter(transformed[0], transformed[1], c='red', s=1, alpha=0.5)

    # Add eigenvectors if real
    eigenvalues, eigenvectors = np.linalg.eig(A)
    if np.isreal(eigenvalues).all():
        for i in range(2):
            ev = eigenvectors[:, i].real * eigenvalues[i].real
            axes[1].quiver(0, 0, ev[0], ev[1], angles='xy', scale_units='xy',
                          scale=1, color='green', width=0.02,
                          label=f'λ={eigenvalues[i]:.2f}')

    axes[1].set_xlim(-0.5, 2)
    axes[1].set_ylim(-0.5, 2)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Transformed Grid\ndet(A) = {np.linalg.det(A):.2f}')
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linewidth=0.5)
    axes[1].legend()

    plt.tight_layout()
    return fig, axes

# Example: rotation + scaling
theta = np.pi/4
A = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]]) * 1.5
visualize_transformation(A)
plt.show()
```

### 3. Eigenvalue Visualization

```python
def visualize_eigenvectors(A):
    """
    Visualize eigenvectors and their transformation.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')

    colors = ['blue', 'red']

    for i in range(len(eigenvalues)):
        if np.isreal(eigenvalues[i]):
            ev = eigenvectors[:, i].real
            lam = eigenvalues[i].real

            # Original eigenvector
            ax.quiver(0, 0, ev[0], ev[1], angles='xy', scale_units='xy', scale=1,
                     color=colors[i], width=0.015,
                     label=f'v{i+1}: eigenvector')

            # Transformed eigenvector (scaled by eigenvalue)
            ax.quiver(0, 0, lam*ev[0], lam*ev[1], angles='xy', scale_units='xy',
                     scale=1, color=colors[i], width=0.015, alpha=0.5,
                     label=f'Av{i+1} = {lam:.2f}v{i+1}')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title(f'Eigenvectors of A\nEigenvalues: {eigenvalues}')
    ax.grid(True, alpha=0.3)

    return fig, ax

# Example
A = np.array([[2, 1], [1, 2]])
visualize_eigenvectors(A)
plt.show()
```

### 4. SVD Visualization

```python
def visualize_svd(A):
    """
    Visualize the SVD decomposition A = UΣV^T.
    """
    U, S, Vt = np.linalg.svd(A)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original matrix
    sns.heatmap(A, annot=True, fmt='.2f', ax=axes[0], cmap='RdBu_r', center=0)
    axes[0].set_title('A (Original)')

    # U matrix
    sns.heatmap(U, annot=True, fmt='.2f', ax=axes[1], cmap='RdBu_r', center=0)
    axes[1].set_title('U (Left singular vectors)')

    # Sigma matrix
    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, S)
    sns.heatmap(Sigma, annot=True, fmt='.2f', ax=axes[2], cmap='YlOrRd')
    axes[2].set_title('Σ (Singular values)')

    # V^T matrix
    sns.heatmap(Vt, annot=True, fmt='.2f', ax=axes[3], cmap='RdBu_r', center=0)
    axes[3].set_title('V^T (Right singular vectors)')

    plt.tight_layout()
    return fig, axes

# Example
A = np.array([[3, 2, 2], [2, 3, -2]])
visualize_svd(A)
plt.show()
```

---

## Calculus Visualizations

### 1. Derivative as Tangent Line

```python
def visualize_derivative(f, f_prime, x0, x_range=(-3, 3)):
    """
    Visualize a function and its derivative at a point.
    """
    x = np.linspace(x_range[0], x_range[1], 200)
    y = f(x)

    # Tangent line at x0
    slope = f_prime(x0)
    y0 = f(x0)
    tangent = y0 + slope * (x - x0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    ax.plot(x, tangent, 'r--', linewidth=2, label=f'Tangent at x={x0}')
    ax.scatter([x0], [y0], color='red', s=100, zorder=5)

    ax.set_xlim(x_range)
    ax.set_ylim(min(y) - 1, max(y) + 1)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title(f"Derivative Visualization\nSlope at x={x0}: f'({x0}) = {slope:.2f}")
    ax.grid(True, alpha=0.3)

    return fig, ax

# Example: f(x) = x^2
f = lambda x: x**2
f_prime = lambda x: 2*x
visualize_derivative(f, f_prime, x0=1.5)
plt.show()
```

### 2. Gradient Descent Visualization

```python
def visualize_gradient_descent(f, grad_f, x0, learning_rate=0.1, n_iterations=50):
    """
    Visualize gradient descent on a 1D function.
    """
    x = np.linspace(-3, 3, 200)
    y = f(x)

    # Perform gradient descent
    trajectory = [x0]
    current_x = x0

    for _ in range(n_iterations):
        current_x = current_x - learning_rate * grad_f(current_x)
        trajectory.append(current_x)

    trajectory = np.array(trajectory)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    ax.scatter(trajectory, f(trajectory), c=range(len(trajectory)),
               cmap='Reds', s=50, zorder=5)
    ax.plot(trajectory, f(trajectory), 'r--', alpha=0.5)

    # Mark start and end
    ax.scatter([trajectory[0]], [f(trajectory[0])], color='green', s=100,
               zorder=6, label='Start')
    ax.scatter([trajectory[-1]], [f(trajectory[-1])], color='red', s=100,
               zorder=6, label='End')

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.set_title(f'Gradient Descent (lr={learning_rate}, iterations={n_iterations})')
    ax.grid(True, alpha=0.3)

    return fig, ax, trajectory

# Example
f = lambda x: x**2 + 0.5*np.sin(3*x)
grad_f = lambda x: 2*x + 1.5*np.cos(3*x)
visualize_gradient_descent(f, grad_f, x0=2.5, learning_rate=0.1)
plt.show()
```

### 3. 3D Surface and Gradient

```python
def visualize_3d_gradient(f, grad_f, x_range=(-2, 2), y_range=(-2, 2)):
    """
    Visualize a 2D function and its gradient field.
    """
    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(14, 5))

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('3D Surface')

    # Contour with gradient field
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)

    # Gradient field (subsample for clarity)
    x_grad = np.linspace(x_range[0], x_range[1], 15)
    y_grad = np.linspace(y_range[0], y_range[1], 15)
    X_grad, Y_grad = np.meshgrid(x_grad, y_grad)
    U, V = grad_f(X_grad, Y_grad)

    ax2.quiver(X_grad, Y_grad, -U, -V, alpha=0.7, color='red')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour with Gradient Field')
    ax2.set_aspect('equal')

    plt.tight_layout()
    return fig

# Example: f(x,y) = x^2 + y^2
f = lambda x, y: x**2 + y**2
grad_f = lambda x, y: (2*x, 2*y)
visualize_3d_gradient(f, grad_f)
plt.show()
```

---

## Probability Visualizations

### 1. Distribution Comparison

```python
def visualize_distributions():
    """
    Visualize common probability distributions.
    """
    x = np.linspace(-5, 5, 200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Normal distribution
    from scipy import stats

    for mu, sigma in [(0, 1), (0, 2), (1, 1)]:
        y = stats.norm.pdf(x, mu, sigma)
        axes[0, 0].plot(x, y, label=f'μ={mu}, σ={sigma}')
    axes[0, 0].set_title('Normal Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('PDF')

    # Uniform distribution
    for a, b in [(-1, 1), (0, 2), (-2, 3)]:
        x_unif = np.linspace(a-0.5, b+0.5, 200)
        y = stats.uniform.pdf(x_unif, a, b-a)
        axes[0, 1].plot(x_unif, y, label=f'[{a}, {b}]')
    axes[0, 1].set_title('Uniform Distribution')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('PDF')

    # Exponential distribution
    x_exp = np.linspace(0, 5, 200)
    for lam in [0.5, 1, 2]:
        y = stats.expon.pdf(x_exp, scale=1/lam)
        axes[1, 0].plot(x_exp, y, label=f'λ={lam}')
    axes[1, 0].set_title('Exponential Distribution')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('PDF')

    # Poisson distribution
    k = np.arange(0, 15)
    for lam in [1, 4, 8]:
        y = stats.poisson.pmf(k, lam)
        axes[1, 1].bar(k + 0.2*(lam//2-1), y, width=0.2, label=f'λ={lam}', alpha=0.7)
    axes[1, 1].set_title('Poisson Distribution')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('k')
    axes[1, 1].set_ylabel('PMF')

    plt.tight_layout()
    return fig

visualize_distributions()
plt.show()
```

### 2. Central Limit Theorem

```python
def visualize_clt(distribution='uniform', n_samples=1000, sample_sizes=[1, 2, 5, 30]):
    """
    Visualize the Central Limit Theorem.
    """
    from scipy import stats

    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(16, 4))

    for idx, n in enumerate(sample_sizes):
        # Generate sample means
        if distribution == 'uniform':
            samples = np.random.uniform(0, 1, (n_samples, n))
        elif distribution == 'exponential':
            samples = np.random.exponential(1, (n_samples, n))
        else:
            samples = np.random.uniform(0, 1, (n_samples, n))

        sample_means = samples.mean(axis=1)

        # Plot histogram
        axes[idx].hist(sample_means, bins=30, density=True, alpha=0.7, edgecolor='black')

        # Overlay normal distribution
        mu = sample_means.mean()
        sigma = sample_means.std()
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        axes[idx].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)

        axes[idx].set_title(f'n = {n}')
        axes[idx].set_xlabel('Sample Mean')

    axes[0].set_ylabel('Density')
    fig.suptitle(f'Central Limit Theorem ({distribution.capitalize()} Distribution)', y=1.02)
    plt.tight_layout()

    return fig

visualize_clt('exponential')
plt.show()
```

### 3. Bayes' Theorem Visualization

```python
def visualize_bayes(prior, likelihood_positive, likelihood_negative):
    """
    Visualize Bayes' theorem for binary classification.

    Parameters
    ----------
    prior : float
        P(Disease)
    likelihood_positive : float
        P(Positive | Disease)
    likelihood_negative : float
        P(Positive | No Disease)
    """
    # Calculate posterior
    p_positive = prior * likelihood_positive + (1 - prior) * likelihood_negative
    posterior = (likelihood_positive * prior) / p_positive

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Prior
    axes[0].bar(['Disease', 'No Disease'], [prior, 1-prior], color=['red', 'blue'])
    axes[0].set_title('Prior P(Disease)')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Probability')

    # Likelihood
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width/2, [likelihood_positive, 1-likelihood_positive],
                width, label='Disease', color='red', alpha=0.7)
    axes[1].bar(x + width/2, [likelihood_negative, 1-likelihood_negative],
                width, label='No Disease', color='blue', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Positive', 'Negative'])
    axes[1].set_title('Likelihood P(Test | State)')
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    # Posterior
    axes[2].bar(['Disease', 'No Disease'], [posterior, 1-posterior], color=['red', 'blue'])
    axes[2].set_title(f'Posterior P(Disease | Positive)\n= {posterior:.3f}')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    return fig, posterior

# Example: Medical test
visualize_bayes(prior=0.01, likelihood_positive=0.99, likelihood_negative=0.05)
plt.show()
```

---

## Optimization Visualizations

### 1. Learning Rate Comparison

```python
def visualize_learning_rates(f, grad_f, x0, learning_rates, n_iterations=100):
    """
    Compare different learning rates.
    """
    x = np.linspace(-3, 3, 200)
    y = f(x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot function and trajectories
    axes[0].plot(x, y, 'b-', linewidth=2, label='f(x)')

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(learning_rates)))

    histories = []
    for lr, color in zip(learning_rates, colors):
        trajectory = [x0]
        current_x = x0

        for _ in range(n_iterations):
            current_x = current_x - lr * grad_f(current_x)
            trajectory.append(current_x)
            if abs(current_x) > 10:  # Divergence check
                break

        trajectory = np.array(trajectory[:min(len(trajectory), n_iterations)])
        histories.append(f(trajectory))

        axes[0].plot(trajectory, f(trajectory), '--', color=color, alpha=0.7)
        axes[0].scatter(trajectory[-1], f(trajectory[-1]), color=color, s=100,
                       label=f'lr={lr}')

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].legend()
    axes[0].set_title('Gradient Descent Trajectories')
    axes[0].set_ylim(-1, 10)

    # Loss curves
    for lr, color, history in zip(learning_rates, colors, histories):
        axes[1].plot(history, color=color, label=f'lr={lr}')

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].set_title('Loss Over Time')
    axes[1].set_yscale('log')

    plt.tight_layout()
    return fig

# Example
f = lambda x: x**2
grad_f = lambda x: 2*x
visualize_learning_rates(f, grad_f, x0=2.5, learning_rates=[0.01, 0.1, 0.5, 0.99])
plt.show()
```

### 2. 2D Optimization Visualization

```python
def visualize_2d_optimization(f, grad_f, x0, learning_rate=0.1, n_iterations=50,
                               optimizer='gd'):
    """
    Visualize optimization on a 2D surface.
    """
    # Create grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Run optimization
    trajectory = [np.array(x0)]
    current = np.array(x0, dtype=float)

    # For momentum
    velocity = np.zeros(2)

    for _ in range(n_iterations):
        grad = np.array(grad_f(current[0], current[1]))

        if optimizer == 'gd':
            current = current - learning_rate * grad
        elif optimizer == 'momentum':
            velocity = 0.9 * velocity + learning_rate * grad
            current = current - velocity

        trajectory.append(current.copy())

    trajectory = np.array(trajectory)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contour(X, Y, Z, levels=30, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)

    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=4,
            linewidth=1, label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100,
               zorder=5, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100,
               zorder=5, label='End')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title(f'2D Optimization ({optimizer}, lr={learning_rate})')
    ax.set_aspect('equal')

    return fig, trajectory

# Example: Rosenbrock function (banana function)
f = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2
grad_f = lambda x, y: (-2*(1-x) - 400*x*(y-x**2), 200*(y-x**2))

visualize_2d_optimization(f, grad_f, x0=[-1.5, 1.5], learning_rate=0.001,
                          n_iterations=1000, optimizer='gd')
plt.show()
```

---

## Information Theory Visualizations

### 1. Entropy Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_binary_entropy():
    """Visualize binary entropy function H(p) = -p*log(p) - (1-p)*log(1-p)"""
    p = np.linspace(0.001, 0.999, 1000)
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p, H, 'b-', linewidth=2)
    ax.set_xlabel('Probability p')
    ax.set_ylabel('Entropy H(p) bits')
    ax.set_title('Binary Entropy Function')
    ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Maximum entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### 2. KL Divergence Visualization

```python
def plot_kl_divergence():
    """Visualize KL divergence between two Gaussians"""
    x = np.linspace(-5, 8, 1000)
    
    # P ~ N(0, 1), Q ~ N(mu, sigma)
    from scipy.stats import norm
    P = norm.pdf(x, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (mu, sigma) in zip(axes, [(0, 1), (1, 1), (2, 2)]):
        Q = norm.pdf(x, mu, sigma)
        kl = np.sum(P * np.log(P / (Q + 1e-10)) * (x[1] - x[0]))
        
        ax.fill_between(x, P, alpha=0.3, label='P ~ N(0,1)')
        ax.fill_between(x, Q, alpha=0.3, label=f'Q ~ N({mu},{sigma})')
        ax.set_title(f'KL(P||Q) = {kl:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('KL Divergence: Asymmetric Distance', fontsize=14)
    plt.tight_layout()
    plt.show()
```

### 3. Cross-Entropy Loss Landscape

```python
def plot_cross_entropy_loss():
    """Visualize cross-entropy loss for binary classification"""
    p_pred = np.linspace(0.01, 0.99, 200)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    loss_y1 = -np.log(p_pred)          # When true label = 1
    loss_y0 = -np.log(1 - p_pred)      # When true label = 0
    
    ax.plot(p_pred, loss_y1, 'b-', linewidth=2, label='y=1: -log(p)')
    ax.plot(p_pred, loss_y0, 'r-', linewidth=2, label='y=0: -log(1-p)')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Loss')
    ax.set_title('Binary Cross-Entropy Loss')
    ax.set_ylim(0, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

---

## Graph Theory Visualizations

### 1. Graph Drawing with NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_graph_types():
    """Visualize different graph types side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Random graph
    G1 = nx.erdos_renyi_graph(10, 0.3, seed=42)
    nx.draw(G1, ax=axes[0], with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, edge_color='gray')
    axes[0].set_title('Random Graph (Erdős-Rényi)')
    
    # Small-world graph
    G2 = nx.watts_strogatz_graph(10, 4, 0.3, seed=42)
    nx.draw_circular(G2, ax=axes[1], with_labels=True, node_color='lightgreen',
                     node_size=500, font_size=10, edge_color='gray')
    axes[1].set_title('Small-World Graph')
    
    # Scale-free graph
    G3 = nx.barabasi_albert_graph(15, 2, seed=42)
    nx.draw_spring(G3, ax=axes[2], with_labels=True, node_color='lightyellow',
                   node_size=500, font_size=10, edge_color='gray')
    axes[2].set_title('Scale-Free Graph')
    
    plt.suptitle('Common Graph Types in ML', fontsize=14)
    plt.tight_layout()
    plt.show()
```

### 2. Adjacency Matrix Heatmap

```python
def plot_adjacency_heatmap(G):
    """Visualize adjacency matrix as a heatmap"""
    A = nx.adjacency_matrix(G).toarray()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graph drawing
    nx.draw_spring(G, ax=axes[0], with_labels=True, node_color='lightblue',
                   node_size=600, font_weight='bold')
    axes[0].set_title('Graph')
    
    # Adjacency matrix
    im = axes[1].imshow(A, cmap='Blues')
    axes[1].set_title('Adjacency Matrix')
    axes[1].set_xlabel('Node')
    axes[1].set_ylabel('Node')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
```

### 3. Spectral Graph Analysis

```python
def plot_graph_spectrum(G):
    """Visualize graph Laplacian eigenvalues"""
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Eigenvalue spectrum
    axes[0].bar(range(len(eigenvalues)), eigenvalues, color='steelblue')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Laplacian Spectrum')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Fiedler vector (2nd smallest eigenvector) for spectral clustering
    _, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    colors = ['red' if v < 0 else 'blue' for v in fiedler]
    
    nx.draw_spring(G, ax=axes[1], node_color=colors, with_labels=True,
                   node_size=600, font_weight='bold')
    axes[1].set_title('Spectral Clustering (Fiedler Vector)')
    
    plt.tight_layout()
    plt.show()
```

---

## Advanced ML Visualizations

### 1. Training Dynamics

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(history):
    """Visualize model training progress"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[2].plot(history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### 2. Loss Landscape Visualization (2D Slice)

```python
def plot_loss_landscape_2d():
    """Visualize 2D loss landscape with gradient descent path"""
    from matplotlib import cm
    
    # Create a sample loss landscape (Rosenbrock-like)
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2  # Rosenbrock
    Z = np.log(Z + 1)  # Log scale for visibility
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Contour plot
    cs = axes[0].contour(X, Y, Z, levels=30, cmap='viridis')
    axes[0].clabel(cs, inline=True, fontsize=8)
    axes[0].set_xlabel('w₁')
    axes[0].set_ylabel('w₂')
    axes[0].set_title('Loss Landscape (Contour)')
    axes[0].plot(1, 1, 'r*', markersize=15, label='Minimum')
    axes[0].legend()
    
    # 3D surface
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    ax3d.set_xlabel('w₁')
    ax3d.set_ylabel('w₂')
    ax3d.set_zlabel('Loss')
    ax3d.set_title('Loss Landscape (3D)')
    
    plt.tight_layout()
    plt.show()
```

### 3. Attention Heatmap

```python
def plot_attention_heatmap(attention_weights, tokens_x, tokens_y):
    """Visualize transformer attention weights"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(tokens_x)))
    ax.set_yticks(range(len(tokens_y)))
    ax.set_xticklabels(tokens_x, rotation=45, ha='right')
    ax.set_yticklabels(tokens_y)
    ax.set_xlabel('Key tokens')
    ax.set_ylabel('Query tokens')
    ax.set_title('Attention Weights')
    
    # Add value annotations
    for i in range(len(tokens_y)):
        for j in range(len(tokens_x)):
            ax.text(j, i, f'{attention_weights[i,j]:.2f}',
                    ha='center', va='center',
                    color='white' if attention_weights[i,j] > 0.5 else 'black',
                    fontsize=8)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
```

### 4. Embedding Visualization (t-SNE / UMAP)

```python
def plot_embeddings_2d(embeddings, labels, method='tsne'):
    """Visualize high-dimensional embeddings in 2D"""
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
    
    coords = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10',
                        alpha=0.6, s=20)
    ax.set_title(f'{method.upper()} Embedding Visualization')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()
    plt.show()
```

---

## Code Templates

### Quick Plotting Template

```python
def quick_plot(x, y, title='', xlabel='x', ylabel='y', figsize=(10, 6)):
    """Quick and clean plotting template."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax
```

### Subplot Template

```python
def create_subplots(n_rows, n_cols, figsize=None):
    """Create a clean subplot grid."""
    if figsize is None:
        figsize = (5*n_cols, 4*n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.tight_layout()
    return fig, axes
```

### Save Figure Template

```python
def save_figure(fig, filename, dpi=300, formats=['png', 'pdf']):
    """Save figure in multiple formats."""
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filename}")
```

---

## Best Practices

### Color Choices

- Use colorblind-friendly palettes: `plt.cm.viridis`, `plt.cm.plasma`
- Keep it simple: max 5-6 colors per plot
- Use consistent colors across related plots

### Labels and Annotations

- Always label axes
- Add titles that explain the visualization
- Use legends when multiple series
- Add annotations for key points

### Figure Sizing

- Default: 10x6 for single plots
- For subplots: 5*n_cols x 4*n_rows
- For presentations: increase font size

### Export Settings

- DPI 300 for print
- PNG for web
- PDF for publications
- SVG for scalable graphics

---

_"A picture is worth a thousand equations!"_ 🎨
