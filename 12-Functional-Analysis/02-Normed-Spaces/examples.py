"""
Normed Spaces - Examples
========================

Implementations demonstrating norm concepts
with applications to machine learning.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable


# =============================================================================
# Example 1: p-Norms and Their Properties
# =============================================================================

def example_1_p_norms():
    """
    Compute and compare different p-norms.
    """
    print("Example 1: p-Norms and Their Properties")
    print("=" * 60)
    
    def p_norm(x: np.ndarray, p: float) -> float:
        """
        Compute p-norm of vector x.
        
        ||x||_p = (Σ |x_i|^p)^{1/p}
        """
        if p == np.inf:
            return np.max(np.abs(x))
        elif p == -np.inf:
            return np.min(np.abs(x))
        elif p == 0:
            return np.sum(x != 0)  # "l0 norm" (not really a norm)
        else:
            return np.power(np.sum(np.power(np.abs(x), p)), 1/p)
    
    def verify_norm_axioms(x: np.ndarray, y: np.ndarray, 
                          c: float, p: float) -> Dict[str, bool]:
        """Verify the three axioms of a norm."""
        results = {}
        
        # N1: Positivity
        results['N1: Positive'] = p_norm(x, p) >= 0
        results['N1: Zero iff zero'] = (p_norm(np.zeros_like(x), p) == 0)
        
        # N2: Homogeneity
        lhs = p_norm(c * x, p)
        rhs = abs(c) * p_norm(x, p)
        results['N2: Homogeneity'] = np.isclose(lhs, rhs)
        
        # N3: Triangle inequality
        lhs = p_norm(x + y, p)
        rhs = p_norm(x, p) + p_norm(y, p)
        results['N3: Triangle ineq'] = lhs <= rhs + 1e-10
        
        return results
    
    # Test vector
    x = np.array([3, -4, 0, 2, -1])
    
    print(f"Vector x = {x}")
    print("\nDifferent p-norms:")
    for p in [0, 1, 2, 3, np.inf]:
        print(f"  ||x||_{p} = {p_norm(x, p):.4f}")
    
    # Verify axioms
    print("\nVerifying norm axioms for p=2:")
    y = np.array([1, 2, -1, 0, 3])
    c = -2.5
    for axiom, satisfied in verify_norm_axioms(x, y, c, 2).items():
        print(f"  {axiom}: {'✓' if satisfied else '✗'}")
    
    return p_norm


# =============================================================================
# Example 2: Norm Equivalence
# =============================================================================

def example_2_norm_equivalence():
    """
    Demonstrate equivalence of norms in finite dimensions.
    """
    print("\nExample 2: Norm Equivalence")
    print("=" * 60)
    
    def norm_equivalence_constants(n: int, p1: float, p2: float,
                                   num_samples: int = 10000) -> Tuple[float, float]:
        """
        Estimate constants c1, c2 such that:
        c1 ||x||_p1 <= ||x||_p2 <= c2 ||x||_p1
        """
        np.random.seed(42)
        
        ratios = []
        for _ in range(num_samples):
            x = np.random.randn(n)
            if np.linalg.norm(x, p1) > 1e-10:
                ratio = np.linalg.norm(x, p2) / np.linalg.norm(x, p1)
                ratios.append(ratio)
        
        return min(ratios), max(ratios)
    
    def theoretical_bounds(n: int, p1: float, p2: float) -> Tuple[float, float]:
        """
        Compute theoretical bounds for common cases.
        """
        if p1 == 2 and p2 == 1:
            return 1.0, np.sqrt(n)
        elif p1 == 2 and p2 == np.inf:
            return 1 / np.sqrt(n), 1.0
        elif p1 == 1 and p2 == np.inf:
            return 1 / n, 1.0
        elif p1 == 1 and p2 == 2:
            return 1 / np.sqrt(n), 1.0
        else:
            return None, None
    
    n = 10
    print(f"Dimension n = {n}")
    print("\nNorm equivalence constants (estimated and theoretical):")
    
    for p1, p2 in [(2, 1), (2, np.inf), (1, np.inf)]:
        c1_est, c2_est = norm_equivalence_constants(n, p1, p2)
        c1_th, c2_th = theoretical_bounds(n, p1, p2)
        
        print(f"\n||·||_{p2} vs ||·||_{p1}:")
        print(f"  Estimated: [{c1_est:.4f}, {c2_est:.4f}]")
        if c1_th is not None:
            print(f"  Theoretical: [{c1_th:.4f}, {c2_th:.4f}]")
    
    return norm_equivalence_constants


# =============================================================================
# Example 3: Norm Balls Visualization
# =============================================================================

def example_3_norm_balls():
    """
    Analyze unit balls for different p-norms.
    """
    print("\nExample 3: Norm Balls")
    print("=" * 60)
    
    def is_in_unit_ball(x: np.ndarray, p: float) -> bool:
        """Check if x is in the unit p-ball."""
        return np.linalg.norm(x, p) <= 1
    
    def unit_ball_volume_mc(p: float, n: int = 2, 
                           num_samples: int = 100000) -> float:
        """
        Estimate volume of unit p-ball using Monte Carlo.
        """
        np.random.seed(42)
        
        # Sample from hypercube [-1, 1]^n
        samples = np.random.uniform(-1, 1, (num_samples, n))
        
        # Count samples in unit ball
        in_ball = np.array([is_in_unit_ball(x, p) for x in samples])
        
        # Volume = (hypercube volume) × (fraction in ball)
        hypercube_volume = 2 ** n
        return hypercube_volume * np.mean(in_ball)
    
    def unit_ball_extreme_points(p: float, n: int = 2) -> np.ndarray:
        """
        Find extreme points of unit p-ball.
        
        For p=1: ±e_i
        For p=∞: vertices of hypercube
        For p=2: all unit vectors (continuous)
        """
        if p == 1:
            # Diamond vertices
            points = []
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1
                points.append(e)
                points.append(-e)
            return np.array(points)
        
        elif p == np.inf:
            # Hypercube vertices
            from itertools import product
            return np.array(list(product([-1, 1], repeat=n)))
        
        else:
            # Sample points on sphere
            angles = np.linspace(0, 2*np.pi, 100)
            if n == 2:
                return np.column_stack([np.cos(angles), np.sin(angles)])
            return None
    
    print("Unit ball properties in 2D:")
    
    for p in [1, 2, 4, np.inf]:
        volume = unit_ball_volume_mc(p, n=2)
        print(f"\np = {p}:")
        print(f"  Volume (area): {volume:.4f}")
        
        extreme = unit_ball_extreme_points(p, n=2)
        if extreme is not None and len(extreme) <= 10:
            print(f"  Extreme points: {len(extreme)}")
    
    # p < 1 is NOT convex
    print("\nNote: p < 1 balls are NOT convex (not a valid norm)")
    
    return unit_ball_volume_mc


# =============================================================================
# Example 4: Matrix Norms
# =============================================================================

def example_4_matrix_norms():
    """
    Different matrix norms and their properties.
    """
    print("\nExample 4: Matrix Norms")
    print("=" * 60)
    
    def frobenius_norm(A: np.ndarray) -> float:
        """Frobenius norm: sqrt(sum of squares)."""
        return np.sqrt(np.sum(A ** 2))
    
    def spectral_norm(A: np.ndarray) -> float:
        """Spectral norm: largest singular value."""
        return np.linalg.svd(A, compute_uv=False)[0]
    
    def nuclear_norm(A: np.ndarray) -> float:
        """Nuclear norm: sum of singular values."""
        return np.sum(np.linalg.svd(A, compute_uv=False))
    
    def operator_norm_p(A: np.ndarray, p: float) -> float:
        """
        Induced operator norm: max ||Ax||_p / ||x||_p
        """
        if p == 2:
            return spectral_norm(A)
        elif p == 1:
            # Max column sum
            return np.max(np.sum(np.abs(A), axis=0))
        elif p == np.inf:
            # Max row sum
            return np.max(np.sum(np.abs(A), axis=1))
        else:
            raise NotImplementedError("Use numerical optimization")
    
    def verify_submultiplicativity(A: np.ndarray, B: np.ndarray,
                                   norm_func: Callable) -> bool:
        """Verify ||AB|| <= ||A|| ||B||."""
        return norm_func(A @ B) <= norm_func(A) * norm_func(B) + 1e-10
    
    # Test matrix
    A = np.array([
        [1, 2, 0],
        [0, 3, 1],
        [1, 0, 2]
    ], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    print("\nMatrix norms:")
    print(f"  ||A||_F (Frobenius) = {frobenius_norm(A):.4f}")
    print(f"  ||A||_2 (Spectral) = {spectral_norm(A):.4f}")
    print(f"  ||A||_* (Nuclear) = {nuclear_norm(A):.4f}")
    print(f"  ||A||_1 (max col sum) = {operator_norm_p(A, 1):.4f}")
    print(f"  ||A||_∞ (max row sum) = {operator_norm_p(A, np.inf):.4f}")
    
    # Verify submultiplicativity
    B = np.random.randn(3, 3)
    print("\nSubmultiplicativity ||AB|| <= ||A|| ||B||:")
    print(f"  Frobenius: {verify_submultiplicativity(A, B, frobenius_norm)}")
    print(f"  Spectral: {verify_submultiplicativity(A, B, spectral_norm)}")
    
    # Relationship between norms
    s = np.linalg.svd(A, compute_uv=False)
    print(f"\nSingular values: {s}")
    print(f"  ||A||_2 = σ_max = {s[0]:.4f}")
    print(f"  ||A||_* = Σσ_i = {sum(s):.4f}")
    print(f"  ||A||_F = √(Σσ_i²) = {np.sqrt(sum(s**2)):.4f}")
    
    return frobenius_norm, spectral_norm, nuclear_norm


# =============================================================================
# Example 5: Dual Norms
# =============================================================================

def example_5_dual_norms():
    """
    Dual norms and Hölder's inequality.
    """
    print("\nExample 5: Dual Norms")
    print("=" * 60)
    
    def dual_norm(y: np.ndarray, p: float) -> float:
        """
        Compute dual norm.
        
        ||y||_* = sup_{||x||_p <= 1} |<x, y>|
        
        For l^p, dual is l^q where 1/p + 1/q = 1.
        """
        if p == 1:
            q = np.inf
        elif p == np.inf:
            q = 1
        else:
            q = p / (p - 1)
        return np.linalg.norm(y, q)
    
    def verify_holder(x: np.ndarray, y: np.ndarray, p: float) -> bool:
        """
        Verify Hölder's inequality: |<x, y>| <= ||x||_p ||y||_q
        """
        q = p / (p - 1) if p > 1 else np.inf
        lhs = np.abs(np.dot(x, y))
        rhs = np.linalg.norm(x, p) * np.linalg.norm(y, q)
        return lhs <= rhs + 1e-10
    
    def find_dual_maximizer(y: np.ndarray, p: float) -> np.ndarray:
        """
        Find x with ||x||_p = 1 that maximizes <x, y>.
        
        Maximizer has form: x_i ∝ sign(y_i) |y_i|^{q-1}
        """
        if p == 1:
            # Max at coordinate with largest |y_i|
            x = np.zeros_like(y)
            idx = np.argmax(np.abs(y))
            x[idx] = np.sign(y[idx])
            return x
        elif p == np.inf:
            # x = sign(y)
            return np.sign(y) / len(y)
        else:
            q = p / (p - 1)
            x = np.sign(y) * np.abs(y) ** (q - 1)
            return x / np.linalg.norm(x, p)
    
    # Test
    y = np.array([1, -2, 3, 0, 1])
    
    print(f"Vector y = {y}")
    print("\nDual norms:")
    
    for p in [1, 2, 3, np.inf]:
        q = p / (p - 1) if 1 < p < np.inf else (np.inf if p == 1 else 1)
        q_str = '∞' if q == np.inf else f'{q:.2f}'
        
        print(f"\n  p = {p}, dual q = {q_str}")
        print(f"    ||y||_q = {dual_norm(y, p):.4f}")
        
        # Verify Hölder
        x = np.random.randn(len(y))
        print(f"    Hölder satisfied: {verify_holder(x, y, p)}")
    
    # Show maximizer
    print("\nMaximizer for ||y||_1* = ||y||_∞:")
    x_max = find_dual_maximizer(y, 1)
    print(f"  x = {x_max}")
    print(f"  <x, y> = {np.dot(x_max, y):.4f}")
    print(f"  ||y||_∞ = {np.linalg.norm(y, np.inf):.4f}")
    
    return dual_norm


# =============================================================================
# Example 6: Lipschitz Functions
# =============================================================================

def example_6_lipschitz():
    """
    Lipschitz continuity and its importance in ML.
    """
    print("\nExample 6: Lipschitz Functions")
    print("=" * 60)
    
    def estimate_lipschitz_constant(f: Callable, domain: np.ndarray,
                                   num_samples: int = 1000) -> float:
        """
        Estimate Lipschitz constant by sampling.
        
        L = max |f(x) - f(y)| / ||x - y||
        """
        n = len(domain)
        np.random.seed(42)
        
        max_ratio = 0
        for _ in range(num_samples):
            i, j = np.random.randint(0, n, 2)
            if i != j:
                diff_f = np.linalg.norm(f(domain[i]) - f(domain[j]))
                diff_x = np.linalg.norm(domain[i] - domain[j])
                if diff_x > 1e-10:
                    ratio = diff_f / diff_x
                    max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    
    def linear_lipschitz(A: np.ndarray) -> float:
        """Lipschitz constant of linear map x -> Ax."""
        return np.linalg.norm(A, 2)
    
    def relu_lipschitz() -> float:
        """Lipschitz constant of ReLU."""
        return 1.0  # |ReLU(x) - ReLU(y)| <= |x - y|
    
    def softmax_lipschitz(d: int) -> float:
        """Upper bound on Lipschitz constant of softmax."""
        return 1.0  # Actually 1 for each output component
    
    # Examples
    print("Lipschitz constants of common functions:")
    
    # Linear
    A = np.random.randn(3, 4)
    print(f"\n  Linear (random 3x4 matrix): L = {linear_lipschitz(A):.4f}")
    
    # ReLU
    relu = lambda x: np.maximum(0, x)
    domain = np.linspace(-2, 2, 100).reshape(-1, 1)
    L_relu = estimate_lipschitz_constant(relu, domain)
    print(f"  ReLU (estimated): L = {L_relu:.4f}")
    print(f"  ReLU (theoretical): L = {relu_lipschitz():.4f}")
    
    # Sigmoid
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    L_sigmoid = estimate_lipschitz_constant(sigmoid, domain)
    print(f"  Sigmoid (estimated): L = {L_sigmoid:.4f}")
    print(f"  Sigmoid (theoretical): L = 0.25")  # max of σ'(x) = σ(x)(1-σ(x))
    
    # Neural network layer
    def layer_lipschitz(W: np.ndarray, activation: str = 'relu') -> float:
        """
        Lipschitz constant of single layer: σ(Wx)
        
        L_layer = ||W||_2 * L_activation
        """
        L_W = np.linalg.norm(W, 2)
        L_act = 1.0 if activation == 'relu' else 0.25  # sigmoid
        return L_W * L_act
    
    W = np.random.randn(10, 5)
    print(f"\n  Neural layer (10x5, ReLU): L = {layer_lipschitz(W, 'relu'):.4f}")
    print(f"  Neural layer (10x5, Sigmoid): L = {layer_lipschitz(W, 'sigmoid'):.4f}")
    
    return estimate_lipschitz_constant


# =============================================================================
# Example 7: Spectral Normalization
# =============================================================================

def example_7_spectral_normalization():
    """
    Spectral normalization for Lipschitz neural networks.
    """
    print("\nExample 7: Spectral Normalization")
    print("=" * 60)
    
    def power_iteration(W: np.ndarray, num_iters: int = 10) -> float:
        """
        Estimate spectral norm using power iteration.
        
        Iterates: u = Wv/||Wv||, v = W^Tu/||W^Tu||
        σ_max ≈ u^T W v
        """
        m, n = W.shape
        
        # Initialize
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(num_iters):
            u = W @ v
            u = u / np.linalg.norm(u)
            v = W.T @ u
            v = v / np.linalg.norm(v)
        
        return float(u @ W @ v)
    
    def spectral_normalize(W: np.ndarray, num_iters: int = 10) -> np.ndarray:
        """
        Normalize W to have spectral norm 1.
        
        W_sn = W / σ_max(W)
        """
        sigma = power_iteration(W, num_iters)
        return W / sigma
    
    class SpectralNormLayer:
        """Layer with spectral normalization."""
        
        def __init__(self, in_features: int, out_features: int):
            self.W = np.random.randn(out_features, in_features)
            self.u = np.random.randn(out_features)
            self.u = self.u / np.linalg.norm(self.u)
            self.v = np.random.randn(in_features)
            self.v = self.v / np.linalg.norm(self.v)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Forward pass with spectral normalization."""
            # Update u, v
            self.v = self.W.T @ self.u
            self.v = self.v / np.linalg.norm(self.v)
            self.u = self.W @ self.v
            self.u = self.u / np.linalg.norm(self.u)
            
            # Compute normalized weight
            sigma = self.u @ self.W @ self.v
            W_sn = self.W / sigma
            
            return W_sn @ x
        
        def get_spectral_norm(self) -> float:
            return self.u @ self.W @ self.v
    
    # Test
    W = np.random.randn(10, 5)
    
    print(f"Original matrix spectral norm:")
    print(f"  Exact (SVD): {np.linalg.norm(W, 2):.4f}")
    print(f"  Power iteration: {power_iteration(W, 10):.4f}")
    
    # Normalize
    W_sn = spectral_normalize(W)
    print(f"\nAfter spectral normalization:")
    print(f"  ||W_sn||_2 = {np.linalg.norm(W_sn, 2):.4f}")
    
    # Network Lipschitz constant
    print("\nMulti-layer network Lipschitz constant:")
    layers = [np.random.randn(10, 8), np.random.randn(8, 6), np.random.randn(6, 4)]
    
    L_original = np.prod([np.linalg.norm(W, 2) for W in layers])
    L_normalized = np.prod([np.linalg.norm(spectral_normalize(W), 2) for W in layers])
    
    print(f"  Original: L = {L_original:.4f}")
    print(f"  Spec normalized: L = {L_normalized:.4f}")
    
    return spectral_normalize


# =============================================================================
# Example 8: Regularization Norms
# =============================================================================

def example_8_regularization():
    """
    Norm-based regularization in ML.
    """
    print("\nExample 8: Regularization Norms")
    print("=" * 60)
    
    def ridge_regression(X: np.ndarray, y: np.ndarray, 
                        lambda_: float) -> np.ndarray:
        """
        min ||y - Xβ||² + λ||β||²
        
        Solution: β = (X^T X + λI)^{-1} X^T y
        """
        n_features = X.shape[1]
        return np.linalg.solve(X.T @ X + lambda_ * np.eye(n_features), X.T @ y)
    
    def lasso_coordinate_descent(X: np.ndarray, y: np.ndarray,
                                lambda_: float, 
                                max_iter: int = 1000,
                                tol: float = 1e-6) -> np.ndarray:
        """
        min ||y - Xβ||² + λ||β||₁
        
        Coordinate descent with soft thresholding.
        """
        n_samples, n_features = X.shape
        beta = np.zeros(n_features)
        
        for _ in range(max_iter):
            beta_old = beta.copy()
            
            for j in range(n_features):
                # Compute residual without j-th feature
                r_j = y - X @ beta + X[:, j] * beta[j]
                
                # Soft threshold update
                rho_j = X[:, j] @ r_j
                z_j = X[:, j] @ X[:, j]
                
                if rho_j < -lambda_ / 2:
                    beta[j] = (rho_j + lambda_ / 2) / z_j
                elif rho_j > lambda_ / 2:
                    beta[j] = (rho_j - lambda_ / 2) / z_j
                else:
                    beta[j] = 0
            
            if np.linalg.norm(beta - beta_old) < tol:
                break
        
        return beta
    
    def elastic_net(X: np.ndarray, y: np.ndarray,
                   lambda1: float, lambda2: float) -> np.ndarray:
        """
        min ||y - Xβ||² + λ₁||β||₁ + λ₂||β||²
        
        Combines L1 and L2 regularization.
        """
        # Can be solved with modified coordinate descent
        # Here using simplified approach
        return lasso_coordinate_descent(X, y, lambda1)  # Simplified
    
    # Generate sparse data
    np.random.seed(42)
    n, p = 100, 20
    true_beta = np.zeros(p)
    true_beta[:5] = [1, -2, 3, -1, 0.5]  # Only 5 non-zero
    
    X = np.random.randn(n, p)
    y = X @ true_beta + 0.1 * np.random.randn(n)
    
    print(f"True sparsity: {np.sum(true_beta != 0)} non-zero out of {p}")
    
    # Compare methods
    lambda_ = 0.5
    
    beta_ridge = ridge_regression(X, y, lambda_)
    beta_lasso = lasso_coordinate_descent(X, y, lambda_)
    
    print(f"\nRidge (L2) sparsity: {np.sum(np.abs(beta_ridge) < 0.01)} zeros")
    print(f"Lasso (L1) sparsity: {np.sum(np.abs(beta_lasso) < 0.01)} zeros")
    
    print(f"\nRecovery error:")
    print(f"  Ridge: ||β - β_true|| = {np.linalg.norm(beta_ridge - true_beta):.4f}")
    print(f"  Lasso: ||β - β_true|| = {np.linalg.norm(beta_lasso - true_beta):.4f}")
    
    return ridge_regression, lasso_coordinate_descent


# =============================================================================
# Example 9: Banach Fixed Point Theorem
# =============================================================================

def example_9_fixed_point():
    """
    Banach fixed point theorem and convergence.
    """
    print("\nExample 9: Banach Fixed Point Theorem")
    print("=" * 60)
    
    def is_contraction(T: Callable, domain: np.ndarray,
                      num_samples: int = 1000) -> Tuple[bool, float]:
        """
        Check if T is a contraction and estimate Lipschitz constant.
        """
        np.random.seed(42)
        max_ratio = 0
        
        for _ in range(num_samples):
            i, j = np.random.randint(0, len(domain), 2)
            if i != j:
                x, y = domain[i], domain[j]
                if np.linalg.norm(x - y) > 1e-10:
                    ratio = np.linalg.norm(T(x) - T(y)) / np.linalg.norm(x - y)
                    max_ratio = max(max_ratio, ratio)
        
        return max_ratio < 1, max_ratio
    
    def fixed_point_iteration(T: Callable, x0: np.ndarray,
                             tol: float = 1e-8,
                             max_iter: int = 1000) -> Tuple[np.ndarray, List]:
        """
        Find fixed point x* = T(x*) by iteration.
        """
        x = x0.copy()
        history = [x.copy()]
        
        for i in range(max_iter):
            x_new = T(x)
            history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                break
            
            x = x_new
        
        return x, history
    
    def convergence_rate(history: List[np.ndarray], 
                        x_star: np.ndarray) -> float:
        """
        Estimate convergence rate from history.
        
        ||x_n - x*|| / ||x_{n-1} - x*|| ≈ α
        """
        errors = [np.linalg.norm(x - x_star) for x in history]
        
        if len(errors) < 3 or errors[-2] < 1e-12:
            return 0
        
        return errors[-1] / errors[-2] if errors[-2] > 1e-12 else 0
    
    # Example: T(x) = (x + 2/x) / 2 (Newton's method for sqrt(2))
    T = lambda x: (x + 2/x) / 2
    
    print("Finding sqrt(2) via fixed point iteration:")
    x0 = np.array([1.0])
    x_star, history = fixed_point_iteration(T, x0)
    
    print(f"  Fixed point: {x_star[0]:.10f}")
    print(f"  Actual sqrt(2): {np.sqrt(2):.10f}")
    print(f"  Iterations: {len(history)}")
    
    # Value iteration example (simple 1D)
    print("\nValue iteration (Bellman operator):")
    gamma = 0.9
    r = np.array([1.0, 2.0, 0.5])  # rewards
    P = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.8, 0.1],
                  [0.2, 0.3, 0.5]])  # transitions
    
    def bellman(V):
        return r + gamma * P @ V
    
    V0 = np.zeros(3)
    V_star, V_history = fixed_point_iteration(bellman, V0)
    
    print(f"  Optimal value: {V_star}")
    print(f"  Iterations to converge: {len(V_history)}")
    
    # Verify contraction
    is_contr, L = is_contraction(bellman, np.random.randn(100, 3))
    print(f"  Is contraction: {is_contr}, Lipschitz ≈ {L:.4f} (γ = {gamma})")
    
    return fixed_point_iteration


# =============================================================================
# Example 10: Completeness and Cauchy Sequences
# =============================================================================

def example_10_completeness():
    """
    Demonstrate completeness and Cauchy sequences.
    """
    print("\nExample 10: Completeness and Cauchy Sequences")
    print("=" * 60)
    
    def is_cauchy(sequence: List[np.ndarray], eps: float = 1e-6) -> bool:
        """
        Check if sequence is Cauchy.
        
        ∀ε>0, ∃N: m,n>N => ||x_m - x_n|| < ε
        """
        n = len(sequence)
        for start in range(n):
            # Check if tail is within epsilon
            tail_diam = max(
                np.linalg.norm(sequence[i] - sequence[j])
                for i in range(start, n)
                for j in range(i+1, n)
            ) if n - start > 1 else 0
            
            if tail_diam < eps:
                return True
        
        return False
    
    def geometric_series(r: float, n_terms: int) -> List[float]:
        """
        Generate partial sums of geometric series.
        
        S_n = 1 + r + r² + ... + r^{n-1}
        """
        sums = []
        s = 0
        for k in range(n_terms):
            s += r ** k
            sums.append(s)
        return sums
    
    def fourier_partial_sums(f: Callable, x: float, 
                           n_terms: int) -> List[float]:
        """
        Fourier series partial sums.
        
        S_n(x) = Σ_{k=-n}^{n} c_k e^{ikx}
        """
        sums = []
        for n in range(1, n_terms + 1):
            s = 0
            for k in range(-n, n + 1):
                # Simple approximation of Fourier coefficient
                ck = np.trapz([f(t) * np.exp(-1j * k * t) 
                              for t in np.linspace(0, 2*np.pi, 100)],
                             np.linspace(0, 2*np.pi, 100)) / (2 * np.pi)
                s += ck * np.exp(1j * k * x)
            sums.append(np.real(s))
        return sums
    
    # Geometric series
    print("Geometric series (converges for |r| < 1):")
    
    for r in [0.5, 0.9, 1.1]:
        sums = geometric_series(r, 20)
        sums_array = [np.array([s]) for s in sums]
        cauchy = is_cauchy(sums_array, eps=0.01)
        limit = 1 / (1 - r) if abs(r) < 1 else np.inf
        
        print(f"  r = {r}: Cauchy = {cauchy}, S_n → {sums[-1]:.4f}, 1/(1-r) = {limit:.4f}")
    
    # Function space example
    print("\nFourier series convergence:")
    f = lambda x: x % (2 * np.pi)  # Sawtooth wave
    x_test = np.pi / 2
    
    sums = fourier_partial_sums(f, x_test, 15)
    print(f"  True value: f({x_test:.2f}) = {f(x_test):.4f}")
    print(f"  Partial sums: {sums[-3:]}")
    
    return is_cauchy


def run_all_examples():
    """Run all normed space examples."""
    print("=" * 70)
    print("NORMED SPACES - EXAMPLES")
    print("=" * 70)
    
    example_1_p_norms()
    example_2_norm_equivalence()
    example_3_norm_balls()
    example_4_matrix_norms()
    example_5_dual_norms()
    example_6_lipschitz()
    example_7_spectral_normalization()
    example_8_regularization()
    example_9_fixed_point()
    example_10_completeness()
    
    print("\n" + "=" * 70)
    print("All normed space examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
