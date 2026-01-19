"""
Optimization Landscape Analysis - Examples
==========================================
Analyzing loss surface geometry and critical points.
"""

import numpy as np
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')


def example_critical_points():
    """Identify and classify critical points."""
    print("=" * 60)
    print("EXAMPLE 1: Critical Point Classification")
    print("=" * 60)
    
    print("f(x,y) = x³ - 3xy²")
    print("\n∇f = [3x² - 3y², -6xy]")
    print("Setting ∇f = 0:")
    print("  x² = y² and xy = 0")
    print("  → (0, 0) is the only critical point")
    
    print("\nHessian:")
    print("  H = [[6x, -6y], [-6y, -6x]]")
    print("  H(0,0) = [[0, 0], [0, 0]]")
    print("\nDegenerate case! Need higher-order analysis.")
    
    # Another example
    print("\n" + "-" * 40)
    print("f(x,y) = x² - y² (saddle)")
    
    H = np.array([[2, 0], [0, -2]])
    eigvals = np.linalg.eigvalsh(H)
    
    print(f"Hessian: {H.tolist()}")
    print(f"Eigenvalues: {eigvals}")
    print(f"Classification: Saddle point (mixed signs)")
    
    print("\n" + "-" * 40)
    print("f(x,y) = x² + y² (minimum)")
    
    H = np.array([[2, 0], [0, 2]])
    eigvals = np.linalg.eigvalsh(H)
    
    print(f"Hessian: {H.tolist()}")
    print(f"Eigenvalues: {eigvals}")
    print(f"Classification: Local minimum (all positive)")


def example_condition_number():
    """Effect of condition number on optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Condition Number Effects")
    print("=" * 60)
    
    def optimize_quadratic(A, b, eta, n_steps=100):
        """Gradient descent on f(x) = 0.5 x'Ax - b'x."""
        x = np.zeros(len(b))
        trajectory = [x.copy()]
        
        for _ in range(n_steps):
            grad = A @ x - b
            x = x - eta * grad
            trajectory.append(x.copy())
        
        return np.array(trajectory)
    
    b = np.array([1, 1])
    
    # Well-conditioned
    A_good = np.array([[2, 0], [0, 1]])
    kappa_good = np.linalg.cond(A_good)
    x_opt_good = np.linalg.solve(A_good, b)
    
    # Ill-conditioned
    A_bad = np.array([[100, 0], [0, 1]])
    kappa_bad = np.linalg.cond(A_bad)
    x_opt_bad = np.linalg.solve(A_bad, b)
    
    eta_good = 2 / (2 + 1)  # Optimal for well-conditioned
    eta_bad = 2 / (100 + 1)  # Safe for ill-conditioned
    
    traj_good = optimize_quadratic(A_good, b, eta_good, 20)
    traj_bad = optimize_quadratic(A_bad, b, eta_bad, 100)
    
    print(f"Well-conditioned: κ = {kappa_good:.1f}")
    print(f"  Optimal x = {x_opt_good}")
    print(f"  After 20 steps: {np.round(traj_good[-1], 6)}")
    print(f"  Error: {np.linalg.norm(traj_good[-1] - x_opt_good):.6e}")
    
    print(f"\nIll-conditioned: κ = {kappa_bad:.1f}")
    print(f"  Optimal x = {x_opt_bad}")
    print(f"  After 100 steps: {np.round(traj_bad[-1], 6)}")
    print(f"  Error: {np.linalg.norm(traj_bad[-1] - x_opt_bad):.6e}")
    
    print("\nHigh condition number → slower convergence")


def example_saddle_point_escape():
    """Escaping from saddle points."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Saddle Point Escape")
    print("=" * 60)
    
    # f(x,y) = x² - y² has saddle at origin
    def f(x):
        return x[0]**2 - x[1]**2
    
    def grad(x):
        return np.array([2*x[0], -2*x[1]])
    
    # Start near saddle
    x_init = np.array([0.01, 0.01])
    
    # Pure gradient descent
    x_gd = x_init.copy()
    eta = 0.1
    gd_trajectory = [x_gd.copy()]
    
    for _ in range(50):
        x_gd = x_gd - eta * grad(x_gd)
        gd_trajectory.append(x_gd.copy())
    
    # GD with noise (simulating SGD)
    np.random.seed(42)
    x_noisy = x_init.copy()
    noisy_trajectory = [x_noisy.copy()]
    
    for _ in range(50):
        noise = 0.05 * np.random.randn(2)
        x_noisy = x_noisy - eta * (grad(x_noisy) + noise)
        noisy_trajectory.append(x_noisy.copy())
    
    gd_trajectory = np.array(gd_trajectory)
    noisy_trajectory = np.array(noisy_trajectory)
    
    print(f"Saddle at origin, starting at {x_init}")
    
    print(f"\nPure GD trajectory (selected steps):")
    print(f"{'Step':>4} {'x':>12} {'y':>12} {'f(x,y)':>12}")
    print("-" * 45)
    for i in [0, 10, 25, 50]:
        x = gd_trajectory[i]
        print(f"{i:>4} {x[0]:>12.6f} {x[1]:>12.6f} {f(x):>12.6f}")
    
    print(f"\nGD with noise trajectory:")
    print(f"{'Step':>4} {'x':>12} {'y':>12} {'f(x,y)':>12}")
    print("-" * 45)
    for i in [0, 10, 25, 50]:
        x = noisy_trajectory[i]
        print(f"{i:>4} {x[0]:>12.6f} {x[1]:>12.6f} {f(x):>12.6f}")
    
    print("\nNoise helps escape saddle along negative curvature direction (y)")


def example_hessian_spectrum():
    """Analyze Hessian eigenvalue spectrum."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Hessian Spectrum Analysis")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate a neural network-like Hessian
    # Bulk: many small eigenvalues
    # Outliers: few large eigenvalues
    d = 100
    
    # Create Hessian with this structure
    bulk_eigvals = 0.1 * np.abs(np.random.randn(d - 5))
    outlier_eigvals = np.array([5.0, 3.0, 2.0, 1.5, 1.0])
    all_eigvals = np.concatenate([bulk_eigvals, outlier_eigvals])
    
    # Random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(d, d))
    H = Q @ np.diag(all_eigvals) @ Q.T
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(H)
    eigvals = np.sort(eigvals)[::-1]
    
    print(f"Simulated Hessian (d = {d})")
    print(f"\nTop 10 eigenvalues: {np.round(eigvals[:10], 4)}")
    print(f"Bottom 10 eigenvalues: {np.round(eigvals[-10:], 4)}")
    
    print(f"\nStatistics:")
    print(f"  Max eigenvalue: {eigvals[0]:.4f}")
    print(f"  Min eigenvalue: {eigvals[-1]:.4f}")
    print(f"  Condition number: {eigvals[0]/eigvals[-1]:.1f}")
    print(f"  Fraction < 0.5: {np.mean(eigvals < 0.5):.1%}")
    print(f"  Fraction > 1.0: {np.mean(eigvals > 1.0):.1%}")
    
    print("\nTypical NN Hessian: bulk of small eigenvalues + few outliers")


def example_flatness_sharpness():
    """Compare flat vs sharp minima."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Flat vs Sharp Minima")
    print("=" * 60)
    
    # Two minima with same loss but different curvature
    def f_sharp(x):
        return 100 * x**2
    
    def f_flat(x):
        return x**2
    
    # Both have minimum at x=0 with f(0)=0
    print("Two minima at x=0 with f(0)=0:")
    print("  Sharp: f(x) = 100x², Hessian = 200")
    print("  Flat:  f(x) = x²,    Hessian = 2")
    
    # Perturbation sensitivity
    print("\nSensitivity to perturbation:")
    print(f"{'Perturbation ε':>15} {'Sharp f(ε)':>15} {'Flat f(ε)':>15}")
    print("-" * 50)
    
    for eps in [0.01, 0.05, 0.1, 0.2]:
        print(f"{eps:>15.2f} {f_sharp(eps):>15.6f} {f_flat(eps):>15.6f}")
    
    # Generalization interpretation
    print("\nGeneralization interpretation:")
    print("If train/test distributions differ by ε:")
    print("  Sharp minimum: large generalization gap")
    print("  Flat minimum: small generalization gap")
    
    # Expected loss under Gaussian perturbation
    print("\nExpected loss under ε ~ N(0, σ²):")
    for sigma in [0.01, 0.05, 0.1]:
        # E[x²] = σ² for x ~ N(0, σ²)
        expected_sharp = 100 * sigma**2
        expected_flat = sigma**2
        print(f"  σ = {sigma}: Sharp = {expected_sharp:.4f}, Flat = {expected_flat:.4f}")


def example_mode_connectivity():
    """Explore connectivity between minima."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Mode Connectivity")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simple non-convex function with multiple minima
    def f(x):
        return np.sin(3*x[0])**2 + np.sin(3*x[1])**2 + 0.1*(x[0]**2 + x[1]**2)
    
    # Two local minima (approximately)
    w1 = np.array([0.0, 0.0])
    w2 = np.array([1.05, 1.05])  # Near another minimum
    
    print(f"Two points: w₁ = {w1}, w₂ = {np.round(w2, 2)}")
    print(f"f(w₁) = {f(w1):.4f}, f(w₂) = {f(w2):.4f}")
    
    # Linear interpolation
    print("\nLinear path: w(t) = (1-t)w₁ + tw₂")
    print(f"{'t':>6} {'f(w(t))':>12}")
    print("-" * 20)
    
    linear_losses = []
    ts = np.linspace(0, 1, 11)
    for t in ts:
        w = (1-t) * w1 + t * w2
        loss = f(w)
        linear_losses.append(loss)
        if t in [0, 0.2, 0.5, 0.8, 1.0]:
            print(f"{t:>6.1f} {loss:>12.4f}")
    
    print(f"\nMax loss on linear path: {max(linear_losses):.4f}")
    print(f"Barrier height: {max(linear_losses) - max(f(w1), f(w2)):.4f}")
    
    # Quadratic Bezier curve (lower loss path)
    print("\nQuadratic Bezier path (curved):")
    control = np.array([0.5, 0.5])  # Control point
    
    bezier_losses = []
    for t in ts:
        # Quadratic Bezier: (1-t)²w₁ + 2(1-t)t·control + t²w₂
        w = (1-t)**2 * w1 + 2*(1-t)*t * control + t**2 * w2
        loss = f(w)
        bezier_losses.append(loss)
        if t in [0, 0.2, 0.5, 0.8, 1.0]:
            print(f"{t:>6.1f} {loss:>12.4f}")
    
    print(f"\nMax loss on Bezier path: {max(bezier_losses):.4f}")
    print("Curved path can have lower barrier!")


def example_gradient_flow():
    """Gradient flow dynamics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Gradient Flow Dynamics")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Linear regression: min ||Xw - y||²
    n, d = 20, 10
    X = np.random.randn(n, d)
    w_true = np.zeros(d)
    w_true[:3] = [1, -1, 0.5]  # Sparse
    y = X @ w_true
    
    def loss(w):
        return 0.5 * np.linalg.norm(X @ w - y)**2
    
    def gradient(w):
        return X.T @ (X @ w - y)
    
    # Gradient descent from zeros (implicit regularization)
    w = np.zeros(d)
    eta = 0.01
    trajectory = [w.copy()]
    
    for _ in range(1000):
        w = w - eta * gradient(w)
        trajectory.append(w.copy())
    
    trajectory = np.array(trajectory)
    
    print("Linear regression with d > n (underdetermined)")
    print(f"n = {n}, d = {d}")
    print(f"\nTrue w (sparse): {w_true}")
    print(f"GD solution:     {np.round(trajectory[-1], 4)}")
    
    # Minimum norm solution
    w_min_norm = X.T @ np.linalg.solve(X @ X.T, y)
    print(f"Min-norm sol:    {np.round(w_min_norm, 4)}")
    
    print(f"\n||w_GD||₂ = {np.linalg.norm(trajectory[-1]):.4f}")
    print(f"||w_min_norm||₂ = {np.linalg.norm(w_min_norm):.4f}")
    print("\nGD finds minimum norm solution (implicit bias)!")


def example_loss_surface_slice():
    """Visualize loss surface along directions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Loss Surface 1D Slice")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simple neural network-like loss
    n, d = 50, 10
    X = np.random.randn(n, d)
    y = np.sign(np.random.randn(n))
    
    def logistic_loss(w):
        z = X @ w
        return np.mean(np.log(1 + np.exp(-y * z)))
    
    def gradient(w):
        z = X @ w
        p = 1 / (1 + np.exp(-y * z))
        return -X.T @ (y * (1 - p)) / n
    
    # Train to find a minimum
    w_star = np.zeros(d)
    for _ in range(500):
        w_star = w_star - 0.5 * gradient(w_star)
    
    print(f"Found minimum with loss = {logistic_loss(w_star):.6f}")
    
    # Random direction
    direction = np.random.randn(d)
    direction = direction / np.linalg.norm(direction)
    
    # Loss along direction
    print("\nLoss along random direction from minimum:")
    print(f"{'α':>8} {'f(w* + αd)':>15}")
    print("-" * 25)
    
    alphas = np.linspace(-2, 2, 9)
    for alpha in alphas:
        w = w_star + alpha * direction
        print(f"{alpha:>8.2f} {logistic_loss(w):>15.6f}")
    
    # Gradient direction
    g = gradient(w_star)
    g_normalized = g / (np.linalg.norm(g) + 1e-10)
    
    print("\nLoss along gradient direction:")
    print(f"{'α':>8} {'f(w* + αg)':>15}")
    print("-" * 25)
    
    for alpha in np.linspace(-0.1, 0.1, 5):
        w = w_star + alpha * g
        print(f"{alpha:>8.4f} {logistic_loss(w):>15.6f}")


def example_curvature_directions():
    """Analyze curvature in different directions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Directional Curvature")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Quadratic with known Hessian
    H = np.array([
        [10, 2, 1],
        [2, 5, 1],
        [1, 1, 1]
    ])
    
    eigvals, eigvecs = np.linalg.eigh(H)
    
    print(f"Hessian H:\n{H}")
    print(f"\nEigenvalues: {np.round(eigvals, 4)}")
    print(f"Condition number: {eigvals[-1]/eigvals[0]:.2f}")
    
    # Curvature in different directions
    print("\nDirectional curvature κ_v = v'Hv:")
    
    directions = {
        'e₁ = [1,0,0]': np.array([1, 0, 0]),
        'e₂ = [0,1,0]': np.array([0, 1, 0]),
        'e₃ = [0,0,1]': np.array([0, 0, 1]),
        'v₁ (min eig)': eigvecs[:, 0],
        'v₃ (max eig)': eigvecs[:, -1],
        'random': np.random.randn(3)
    }
    
    print(f"{'Direction':>15} {'Curvature':>12}")
    print("-" * 30)
    
    for name, v in directions.items():
        v = v / np.linalg.norm(v)
        curvature = v @ H @ v
        print(f"{name:>15} {curvature:>12.4f}")
    
    print(f"\nMin curvature (along v₁): {eigvals[0]:.4f}")
    print(f"Max curvature (along v₃): {eigvals[-1]:.4f}")


def example_sam():
    """Sharpness-Aware Minimization concept."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Sharpness-Aware Minimization (SAM)")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 100, 5
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    def gradient(w):
        return X.T @ (X @ w - y) / n
    
    # Standard gradient descent
    w_sgd = np.zeros(d)
    eta = 0.1
    
    for _ in range(100):
        w_sgd = w_sgd - eta * gradient(w_sgd)
    
    # SAM: Sharpness-aware minimization
    w_sam = np.zeros(d)
    rho = 0.05  # Perturbation radius
    
    for _ in range(100):
        g = gradient(w_sam)
        # Compute worst-case perturbation
        eps = rho * g / (np.linalg.norm(g) + 1e-10)
        # Gradient at perturbed point
        g_sam = gradient(w_sam + eps)
        # Update
        w_sam = w_sam - eta * g_sam
    
    # Compute sharpness (max eigenvalue of Hessian)
    H = X.T @ X / n
    sharpness_sgd = np.max(np.linalg.eigvalsh(H))  # Same for quadratic
    
    print("SAM minimizes: max_{||ε||≤ρ} f(w + ε)")
    print("\nComparing SGD vs SAM:")
    print(f"  SGD loss:  {loss(w_sgd):.6f}")
    print(f"  SAM loss:  {loss(w_sam):.6f}")
    
    # Evaluate robustness
    print("\nRobustness to perturbation (ε ~ N(0, 0.01²I)):")
    n_trials = 100
    
    sgd_perturbed_losses = []
    sam_perturbed_losses = []
    
    for _ in range(n_trials):
        eps = 0.01 * np.random.randn(d)
        sgd_perturbed_losses.append(loss(w_sgd + eps))
        sam_perturbed_losses.append(loss(w_sam + eps))
    
    print(f"  SGD mean perturbed loss: {np.mean(sgd_perturbed_losses):.6f}")
    print(f"  SAM mean perturbed loss: {np.mean(sam_perturbed_losses):.6f}")
    
    print("\nNote: For this quadratic problem, both methods behave similarly.")
    print("SAM shows more benefit on non-convex neural network losses.")


def example_saddle_point_ratio():
    """Ratio of saddle points to minima in high dimensions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Saddle Points vs Minima in High-D")
    print("=" * 60)
    
    print("At a random critical point, each Hessian eigenvalue is")
    print("independently + or - with probability 1/2.")
    
    print("\nP(all eigenvalues positive) = (1/2)^d")
    print("\n{'d':>5} {'P(minimum)':>15} {'P(saddle)':>15}")
    print("-" * 40)
    
    for d in [2, 5, 10, 20, 50, 100]:
        p_min = 0.5**d
        p_saddle = 1 - 2 * p_min  # Exclude maxima
        print(f"{d:>5} {p_min:>15.2e} {p_saddle:>15.6f}")
    
    print("\nIn high dimensions, almost all critical points are saddles!")
    
    # Simulate
    print("\nSimulation: Random symmetric matrices")
    for d in [5, 10, 20]:
        n_trials = 1000
        n_minima = 0
        n_saddles = 0
        n_maxima = 0
        
        for _ in range(n_trials):
            # Random symmetric matrix (Wigner)
            A = np.random.randn(d, d)
            A = (A + A.T) / 2
            eigvals = np.linalg.eigvalsh(A)
            
            if np.all(eigvals > 0):
                n_minima += 1
            elif np.all(eigvals < 0):
                n_maxima += 1
            else:
                n_saddles += 1
        
        print(f"d={d}: minima={n_minima/n_trials:.3f}, "
              f"saddles={n_saddles/n_trials:.3f}, "
              f"maxima={n_maxima/n_trials:.3f}")


def example_interpolation_barrier():
    """Measure loss barrier between solutions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Loss Barrier Between Solutions")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Logistic regression with two different solutions
    n, d = 50, 20
    X = np.random.randn(n, d)
    y = np.sign(X @ np.random.randn(d) + 0.1 * np.random.randn(n))
    
    def logistic_loss(w):
        z = X @ w
        z = np.clip(z, -500, 500)
        return np.mean(np.log(1 + np.exp(-y * z)))
    
    def gradient(w):
        z = X @ w
        z = np.clip(z, -500, 500)
        p = 1 / (1 + np.exp(-y * z))
        return -X.T @ (y * (1 - p)) / n
    
    # Train from two different initializations
    w1 = np.random.randn(d) * 0.1
    w2 = np.random.randn(d) * 0.1
    
    for _ in range(500):
        w1 = w1 - 0.5 * gradient(w1)
        w2 = w2 - 0.5 * gradient(w2)
    
    print(f"Two solutions from different initializations:")
    print(f"  w₁ loss: {logistic_loss(w1):.6f}")
    print(f"  w₂ loss: {logistic_loss(w2):.6f}")
    print(f"  ||w₁ - w₂||: {np.linalg.norm(w1 - w2):.4f}")
    
    # Linear interpolation
    print("\nLinear interpolation w(t) = (1-t)w₁ + tw₂:")
    print(f"{'t':>6} {'Loss':>12}")
    print("-" * 20)
    
    ts = np.linspace(0, 1, 11)
    losses = []
    for t in ts:
        w = (1-t) * w1 + t * w2
        loss = logistic_loss(w)
        losses.append(loss)
    
    for i, t in enumerate(ts):
        if t in [0, 0.25, 0.5, 0.75, 1.0]:
            print(f"{t:>6.2f} {losses[i]:>12.6f}")
    
    barrier = max(losses) - max(logistic_loss(w1), logistic_loss(w2))
    print(f"\nBarrier height: {barrier:.6f}")
    
    if barrier < 0.1:
        print("Low barrier → solutions may be mode-connected")
    else:
        print("High barrier → solutions are separated")


if __name__ == "__main__":
    example_critical_points()
    example_condition_number()
    example_saddle_point_escape()
    example_hessian_spectrum()
    example_flatness_sharpness()
    example_mode_connectivity()
    example_gradient_flow()
    example_loss_surface_slice()
    example_curvature_directions()
    example_sam()
    example_saddle_point_ratio()
    example_interpolation_barrier()
