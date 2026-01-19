"""
Normed Spaces - Exercises
=========================

Hands-on exercises for normed space concepts
with applications to machine learning.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


# =============================================================================
# Exercise 1: Custom Norm Implementation
# =============================================================================

class Exercise1:
    """
    Implement a flexible norm class that supports:
    - p-norms for any p >= 1
    - Verification of norm axioms
    - Unit ball membership testing
    """
    
    @staticmethod
    def problem():
        print("Exercise 1: Custom Norm Implementation")
        print("-" * 50)
        print("""
        Implement a Norm class with:
        
        1. __call__(x) - compute the norm
        2. verify_positivity(x) - check ||x|| >= 0 and ||x|| = 0 iff x = 0
        3. verify_homogeneity(x, c) - check ||cx|| = |c| ||x||
        4. verify_triangle(x, y) - check ||x + y|| <= ||x|| + ||y||
        5. unit_ball_membership(x) - check if ||x|| <= 1
        
        Support p-norms where ||x||_p = (Σ|x_i|^p)^{1/p}
        
        Test your implementation for p = 1, 2, and infinity.
        """)
    
    @staticmethod
    def solution():
        class Norm:
            """Flexible p-norm implementation."""
            
            def __init__(self, p: float = 2):
                if p < 1 and p != float('inf'):
                    raise ValueError("p must be >= 1 for valid norm")
                self.p = p
            
            def __call__(self, x: np.ndarray) -> float:
                """Compute p-norm of x."""
                x = np.asarray(x, dtype=float)
                
                if self.p == np.inf:
                    return np.max(np.abs(x))
                else:
                    return np.power(np.sum(np.power(np.abs(x), self.p)), 
                                   1 / self.p)
            
            def verify_positivity(self, x: np.ndarray) -> Tuple[bool, str]:
                """Check N1: ||x|| >= 0 and ||x|| = 0 iff x = 0."""
                norm_x = self(x)
                
                if norm_x < 0:
                    return False, f"Norm is negative: {norm_x}"
                
                # Check zero vector
                zero = np.zeros_like(x)
                if self(zero) != 0:
                    return False, f"||0|| != 0: {self(zero)}"
                
                # Check non-zero
                if np.all(x == 0) and norm_x != 0:
                    return False, "x = 0 but ||x|| != 0"
                
                if not np.all(x == 0) and norm_x == 0:
                    return False, "x != 0 but ||x|| = 0"
                
                return True, "Positivity satisfied"
            
            def verify_homogeneity(self, x: np.ndarray, c: float) -> Tuple[bool, str]:
                """Check N2: ||cx|| = |c| ||x||."""
                lhs = self(c * x)
                rhs = abs(c) * self(x)
                
                if np.isclose(lhs, rhs):
                    return True, f"||cx|| = {lhs:.6f}, |c|||x|| = {rhs:.6f}"
                else:
                    return False, f"||cx|| = {lhs:.6f} != |c|||x|| = {rhs:.6f}"
            
            def verify_triangle(self, x: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
                """Check N3: ||x + y|| <= ||x|| + ||y||."""
                lhs = self(x + y)
                rhs = self(x) + self(y)
                
                if lhs <= rhs + 1e-10:
                    return True, f"||x+y|| = {lhs:.6f} <= {rhs:.6f}"
                else:
                    return False, f"||x+y|| = {lhs:.6f} > {rhs:.6f}"
            
            def unit_ball_membership(self, x: np.ndarray) -> bool:
                """Check if ||x|| <= 1."""
                return self(x) <= 1.0
        
        # Test
        print("\nSolution:")
        x = np.array([1, -2, 3])
        y = np.array([2, 1, -1])
        c = -2.5
        
        for p in [1, 2, np.inf]:
            print(f"\n  p = {p}:")
            norm = Norm(p)
            
            print(f"    ||x|| = {norm(x):.4f}")
            print(f"    Positivity: {norm.verify_positivity(x)}")
            print(f"    Homogeneity: {norm.verify_homogeneity(x, c)}")
            print(f"    Triangle: {norm.verify_triangle(x, y)}")
        
        return Norm


# =============================================================================
# Exercise 2: Norm Equivalence Analysis
# =============================================================================

class Exercise2:
    """
    Analyze norm equivalence in finite dimensions.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 2: Norm Equivalence Analysis")
        print("-" * 50)
        print("""
        For R^n, find the constants c₁, c₂ such that:
        
            c₁ ||x||_p <= ||x||_q <= c₂ ||x||_p
        
        Tasks:
        1. Derive theoretical bounds for:
           - p=2, q=1
           - p=2, q=∞
           - p=1, q=∞
        
        2. Implement empirical estimation via sampling
        
        3. Verify bounds on random vectors
        
        Key relationships:
        - ||x||_∞ <= ||x||_2 <= √n ||x||_∞
        - ||x||_∞ <= ||x||_1 <= n ||x||_∞
        - ||x||_2 <= ||x||_1 <= √n ||x||_2
        """)
    
    @staticmethod
    def solution():
        def theoretical_bounds(n: int, p: float, q: float) -> Tuple[float, float]:
            """
            Compute theoretical norm equivalence bounds.
            
            Returns (c1, c2) such that c1 ||x||_p <= ||x||_q <= c2 ||x||_p
            """
            if p == q:
                return 1.0, 1.0
            
            # Key relationships
            if p == 2 and q == 1:
                # ||x||_2 <= ||x||_1 <= √n ||x||_2
                return 1.0, np.sqrt(n)
            elif p == 2 and q == np.inf:
                # 1/√n ||x||_2 <= ||x||_∞ <= ||x||_2
                return 1/np.sqrt(n), 1.0
            elif p == 1 and q == 2:
                # 1/√n ||x||_1 <= ||x||_2 <= ||x||_1
                return 1/np.sqrt(n), 1.0
            elif p == 1 and q == np.inf:
                # 1/n ||x||_1 <= ||x||_∞ <= ||x||_1
                return 1/n, 1.0
            elif p == np.inf and q == 2:
                # ||x||_∞ <= ||x||_2 <= √n ||x||_∞
                return 1.0, np.sqrt(n)
            elif p == np.inf and q == 1:
                # ||x||_∞ <= ||x||_1 <= n ||x||_∞
                return 1.0, n
            else:
                return None, None
        
        def empirical_bounds(n: int, p: float, q: float, 
                           num_samples: int = 10000) -> Tuple[float, float]:
            """Estimate bounds via random sampling."""
            np.random.seed(42)
            
            ratios = []
            for _ in range(num_samples):
                x = np.random.randn(n)
                norm_p = np.linalg.norm(x, p)
                norm_q = np.linalg.norm(x, q)
                
                if norm_p > 1e-10:
                    ratios.append(norm_q / norm_p)
            
            return min(ratios), max(ratios)
        
        def verify_bounds(n: int, p: float, q: float, 
                         num_tests: int = 1000) -> bool:
            """Verify theoretical bounds hold for random vectors."""
            c1, c2 = theoretical_bounds(n, p, q)
            if c1 is None:
                return False
            
            np.random.seed(0)
            for _ in range(num_tests):
                x = np.random.randn(n)
                norm_p = np.linalg.norm(x, p)
                norm_q = np.linalg.norm(x, q)
                
                if not (c1 * norm_p <= norm_q + 1e-10 and 
                       norm_q <= c2 * norm_p + 1e-10):
                    return False
            
            return True
        
        print("\nSolution:")
        n = 10
        
        test_cases = [(2, 1), (2, np.inf), (1, np.inf)]
        
        for p, q in test_cases:
            c1_th, c2_th = theoretical_bounds(n, p, q)
            c1_emp, c2_emp = empirical_bounds(n, p, q)
            verified = verify_bounds(n, p, q)
            
            q_str = '∞' if q == np.inf else str(q)
            print(f"\n  n={n}, ||·||_{q_str} vs ||·||_{p}:")
            print(f"    Theoretical: [{c1_th:.4f}, {c2_th:.4f}]")
            print(f"    Empirical: [{c1_emp:.4f}, {c2_emp:.4f}]")
            print(f"    Verified: {verified}")
        
        return theoretical_bounds, empirical_bounds


# =============================================================================
# Exercise 3: Matrix Norm Relationships
# =============================================================================

class Exercise3:
    """
    Explore relationships between different matrix norms.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 3: Matrix Norm Relationships")
        print("-" * 50)
        print("""
        For an m×n matrix A with singular values σ₁ >= σ₂ >= ... >= σ_r:
        
        1. Implement Frobenius, spectral, and nuclear norms
        
        2. Verify the relationships:
           - ||A||_2 <= ||A||_F <= √r ||A||_2
           - ||A||_2 <= ||A||_* <= √r ||A||_2
           - ||A||_F² = Σσᵢ²
           - ||A||_* = Σσᵢ
        
        3. Show submultiplicativity: ||AB|| <= ||A|| ||B||
        
        4. For gradient descent, verify ||A - ηA^TA||_2 < 1 
           when 0 < η < 2/σ₁²
        """)
    
    @staticmethod
    def solution():
        def frobenius_norm(A: np.ndarray) -> float:
            """||A||_F = √(Σ|a_ij|²) = √(Σσᵢ²)"""
            return np.sqrt(np.sum(A ** 2))
        
        def spectral_norm(A: np.ndarray) -> float:
            """||A||_2 = σ_max"""
            return np.linalg.norm(A, 2)
        
        def nuclear_norm(A: np.ndarray) -> float:
            """||A||_* = Σσᵢ"""
            return np.sum(np.linalg.svd(A, compute_uv=False))
        
        def verify_relationships(A: np.ndarray) -> Dict[str, bool]:
            """Verify matrix norm relationships."""
            s = np.linalg.svd(A, compute_uv=False)
            r = np.sum(s > 1e-10)  # rank
            
            norm_F = frobenius_norm(A)
            norm_2 = spectral_norm(A)
            norm_star = nuclear_norm(A)
            
            results = {}
            
            # ||A||_2 <= ||A||_F <= √r ||A||_2
            results['||A||_2 <= ||A||_F'] = norm_2 <= norm_F + 1e-10
            results['||A||_F <= √r ||A||_2'] = norm_F <= np.sqrt(r) * norm_2 + 1e-10
            
            # ||A||_F = √(Σσᵢ²)
            results['||A||_F = √(Σσᵢ²)'] = np.isclose(norm_F, np.sqrt(np.sum(s**2)))
            
            # ||A||_* = Σσᵢ
            results['||A||_* = Σσᵢ'] = np.isclose(norm_star, np.sum(s))
            
            return results
        
        def verify_submultiplicativity(A: np.ndarray, B: np.ndarray) -> Dict[str, bool]:
            """Verify ||AB|| <= ||A|| ||B|| for different norms."""
            results = {}
            
            for name, norm_func in [('Frobenius', frobenius_norm),
                                   ('Spectral', spectral_norm)]:
                lhs = norm_func(A @ B)
                rhs = norm_func(A) * norm_func(B)
                results[name] = lhs <= rhs + 1e-10
            
            return results
        
        def gradient_descent_convergence(A: np.ndarray, eta: float) -> float:
            """
            For x_{k+1} = x_k - η A^T(Ax_k - b) = (I - ηA^TA)x_k + ηA^Tb
            
            Convergence requires ||I - ηA^TA||_2 < 1
            """
            n = A.shape[1]
            M = np.eye(n) - eta * A.T @ A
            return spectral_norm(M)
        
        print("\nSolution:")
        
        A = np.array([[1, 2, 0],
                     [0, 3, 1],
                     [2, 0, 1]], dtype=float)
        
        print(f"  Matrix A (3×3):")
        print(f"  Singular values: {np.linalg.svd(A, compute_uv=False)}")
        
        print(f"\n  Norms:")
        print(f"    ||A||_F = {frobenius_norm(A):.4f}")
        print(f"    ||A||_2 = {spectral_norm(A):.4f}")
        print(f"    ||A||_* = {nuclear_norm(A):.4f}")
        
        print(f"\n  Relationships:")
        for rel, satisfied in verify_relationships(A).items():
            print(f"    {rel}: {'✓' if satisfied else '✗'}")
        
        print(f"\n  Submultiplicativity with random B:")
        B = np.random.randn(3, 3)
        for norm_name, satisfied in verify_submultiplicativity(A, B).items():
            print(f"    {norm_name}: {'✓' if satisfied else '✗'}")
        
        print(f"\n  Gradient descent convergence:")
        sigma_max = spectral_norm(A)
        for eta in [0.01, 0.1, 0.5, 2/sigma_max**2]:
            conv = gradient_descent_convergence(A, eta)
            print(f"    η={eta:.4f}: ||I - ηA^TA||_2 = {conv:.4f} {'< 1 ✓' if conv < 1 else '>= 1 ✗'}")
        
        return verify_relationships


# =============================================================================
# Exercise 4: Lipschitz Neural Network
# =============================================================================

class Exercise4:
    """
    Build a provably Lipschitz neural network.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 4: Lipschitz Neural Network")
        print("-" * 50)
        print("""
        Implement a neural network with guaranteed Lipschitz constant.
        
        1. Each layer f_i(x) = σ(W_i x + b_i) has Lipschitz constant
           L_i = ||W_i||_2 × L_σ
        
        2. Network f = f_L ∘ ... ∘ f_1 has Lipschitz constant
           L = ∏ L_i
        
        Tasks:
        - Implement spectral normalization using power iteration
        - Create layers with L_i = 1
        - Verify network Lipschitz constant empirically
        - Apply to 1-Lipschitz classifier for robustness
        """)
    
    @staticmethod
    def solution():
        class LipschitzLinear:
            """Linear layer with spectral normalization."""
            
            def __init__(self, in_features: int, out_features: int):
                scale = np.sqrt(2 / in_features)
                self.W = np.random.randn(out_features, in_features) * scale
                self.b = np.zeros(out_features)
                
                # For power iteration
                self.u = np.random.randn(out_features)
                self.u /= np.linalg.norm(self.u)
            
            def spectral_norm(self, num_iters: int = 5) -> float:
                """Estimate spectral norm using power iteration."""
                u = self.u.copy()
                
                for _ in range(num_iters):
                    v = self.W.T @ u
                    v /= np.linalg.norm(v) + 1e-12
                    u = self.W @ v
                    u /= np.linalg.norm(u) + 1e-12
                
                self.u = u
                return u @ self.W @ v
            
            def forward(self, x: np.ndarray) -> np.ndarray:
                """Forward with spectral normalization."""
                sigma = self.spectral_norm()
                W_norm = self.W / sigma
                return W_norm @ x + self.b
            
            def lipschitz_constant(self) -> float:
                return 1.0  # After normalization
        
        class LipschitzNetwork:
            """Multi-layer Lipschitz network."""
            
            def __init__(self, layer_sizes: List[int], 
                        activation: str = 'relu'):
                self.layers = []
                for i in range(len(layer_sizes) - 1):
                    self.layers.append(
                        LipschitzLinear(layer_sizes[i], layer_sizes[i+1])
                    )
                
                self.activation = activation
                # ReLU has L=1, other activations need adjustment
                self.L_activation = 1.0 if activation == 'relu' else 0.25
            
            def _activate(self, x: np.ndarray) -> np.ndarray:
                if self.activation == 'relu':
                    return np.maximum(0, x)
                elif self.activation == 'sigmoid':
                    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                return x
            
            def forward(self, x: np.ndarray) -> np.ndarray:
                for i, layer in enumerate(self.layers):
                    x = layer.forward(x)
                    if i < len(self.layers) - 1:
                        x = self._activate(x)
                return x
            
            def lipschitz_constant(self) -> float:
                """Compute network Lipschitz constant."""
                L = 1.0
                for i, layer in enumerate(self.layers):
                    L *= layer.lipschitz_constant()
                    if i < len(self.layers) - 1:
                        L *= self.L_activation
                return L
            
            def verify_lipschitz(self, num_samples: int = 1000) -> float:
                """Empirically estimate Lipschitz constant."""
                input_dim = self.layers[0].W.shape[1]
                max_ratio = 0
                
                np.random.seed(42)
                for _ in range(num_samples):
                    x1 = np.random.randn(input_dim)
                    x2 = np.random.randn(input_dim)
                    
                    y1 = self.forward(x1)
                    y2 = self.forward(x2)
                    
                    dx = np.linalg.norm(x1 - x2)
                    dy = np.linalg.norm(y1 - y2)
                    
                    if dx > 1e-10:
                        max_ratio = max(max_ratio, dy / dx)
                
                return max_ratio
        
        print("\nSolution:")
        
        # Create network
        net = LipschitzNetwork([10, 32, 16, 1], activation='relu')
        
        print(f"  Network: 10 -> 32 -> 16 -> 1")
        print(f"  Theoretical Lipschitz constant: {net.lipschitz_constant():.4f}")
        print(f"  Empirical Lipschitz constant: {net.verify_lipschitz():.4f}")
        
        # Demonstrate robustness
        print(f"\n  Robustness guarantee:")
        x = np.random.randn(10)
        y = net.forward(x)
        
        epsilon = 0.1
        L = net.lipschitz_constant()
        
        print(f"    For ||δ|| <= {epsilon}, output change <= {L * epsilon:.4f}")
        
        return LipschitzNetwork


# =============================================================================
# Exercise 5: Dual Norm Computation
# =============================================================================

class Exercise5:
    """
    Compute dual norms and verify Hölder's inequality.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 5: Dual Norm Computation")
        print("-" * 50)
        print("""
        The dual norm is defined as:
        
            ||y||_* = sup_{||x|| <= 1} <x, y>
        
        Tasks:
        1. Implement dual norm computation for p-norms
           - Dual of l^p is l^q where 1/p + 1/q = 1
        
        2. Find the maximizer x* achieving ||y||_*
        
        3. Verify Hölder's inequality:
           |<x, y>| <= ||x||_p ||y||_q
        
        4. Apply to matrix norms:
           - Dual of spectral norm is nuclear norm
        """)
    
    @staticmethod
    def solution():
        def dual_exponent(p: float) -> float:
            """Find q such that 1/p + 1/q = 1."""
            if p == 1:
                return np.inf
            elif p == np.inf:
                return 1
            else:
                return p / (p - 1)
        
        def dual_norm(y: np.ndarray, p: float) -> float:
            """Compute dual norm of y with respect to p-norm."""
            q = dual_exponent(p)
            return np.linalg.norm(y, q)
        
        def dual_maximizer(y: np.ndarray, p: float) -> np.ndarray:
            """
            Find x with ||x||_p = 1 maximizing <x, y>.
            
            Maximizer: x_i ∝ sign(y_i) |y_i|^{q-1}
            """
            q = dual_exponent(p)
            
            if p == 1:
                # Maximizer at largest coordinate
                x = np.zeros_like(y)
                idx = np.argmax(np.abs(y))
                x[idx] = np.sign(y[idx])
                return x
            elif p == np.inf:
                # x = sign(y), normalized
                x = np.sign(y)
                return x / np.linalg.norm(x, np.inf)
            else:
                # General case
                x = np.sign(y) * np.abs(y) ** (q - 1)
                return x / np.linalg.norm(x, p)
        
        def verify_holder(x: np.ndarray, y: np.ndarray, p: float) -> bool:
            """Verify |<x, y>| <= ||x||_p ||y||_q."""
            q = dual_exponent(p)
            
            lhs = np.abs(np.dot(x, y))
            rhs = np.linalg.norm(x, p) * np.linalg.norm(y, q)
            
            return lhs <= rhs + 1e-10
        
        def matrix_dual_norms(A: np.ndarray):
            """Show spectral and nuclear are dual."""
            # Spectral norm
            spec = np.linalg.norm(A, 2)
            
            # Nuclear norm
            nuc = np.sum(np.linalg.svd(A, compute_uv=False))
            
            # Verify: ||A||_2 = sup_{||B||_* <= 1} <A, B>
            # <A, B> = trace(A^T B)
            
            return spec, nuc
        
        print("\nSolution:")
        
        y = np.array([1, -3, 2, 0, 4])
        
        print(f"  Vector y = {y}")
        print(f"\n  Dual norms:")
        
        for p in [1, 2, 3, np.inf]:
            q = dual_exponent(p)
            q_str = '∞' if q == np.inf else f'{q:.2f}'
            
            print(f"\n    p = {p}, dual q = {q_str}")
            print(f"      ||y||_q = {dual_norm(y, p):.4f}")
            
            # Find and verify maximizer
            x_star = dual_maximizer(y, p)
            inner = np.dot(x_star, y)
            print(f"      Maximizer achieves: <x*, y> = {inner:.4f}")
            
            # Verify Hölder
            x_random = np.random.randn(len(y))
            print(f"      Hölder satisfied: {verify_holder(x_random, y, p)}")
        
        # Matrix example
        print(f"\n  Matrix dual norms:")
        A = np.random.randn(3, 4)
        spec, nuc = matrix_dual_norms(A)
        print(f"    ||A||_2 (spectral) = {spec:.4f}")
        print(f"    ||A||_* (nuclear) = {nuc:.4f}")
        print(f"    (Spectral and nuclear norms are dual to each other)")
        
        return dual_norm, dual_maximizer


# =============================================================================
# Exercise 6: Fixed Point Iteration
# =============================================================================

class Exercise6:
    """
    Implement and analyze fixed point iteration.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 6: Fixed Point Iteration")
        print("-" * 50)
        print("""
        Banach fixed point theorem: If T: X -> X is a contraction
        (||T(x) - T(y)|| <= α||x - y|| with α < 1), then:
        
        1. T has a unique fixed point x* = T(x*)
        2. x_{n+1} = T(x_n) converges to x* from any x_0
        3. ||x_n - x*|| <= α^n/(1-α) ||x_1 - x_0||
        
        Tasks:
        1. Implement fixed point iteration
        2. Verify contraction property
        3. Estimate convergence rate
        4. Apply to value iteration in RL
        """)
    
    @staticmethod
    def solution():
        def fixed_point_iteration(T: Callable, x0: np.ndarray,
                                 tol: float = 1e-10,
                                 max_iter: int = 1000) -> Tuple[np.ndarray, int, List]:
            """
            Find fixed point via iteration.
            
            Returns: (fixed_point, iterations, history)
            """
            x = x0.copy()
            history = [x.copy()]
            
            for i in range(max_iter):
                x_new = T(x)
                history.append(x_new.copy())
                
                if np.linalg.norm(x_new - x) < tol:
                    return x_new, i + 1, history
                
                x = x_new
            
            return x, max_iter, history
        
        def estimate_contraction_constant(T: Callable, x0: np.ndarray,
                                        num_iters: int = 20) -> float:
            """Estimate contraction constant α."""
            np.random.seed(42)
            
            max_ratio = 0
            x = x0.copy()
            
            for _ in range(num_iters):
                x1 = x + 0.1 * np.random.randn(*x.shape)
                
                Tx = T(x)
                Tx1 = T(x1)
                
                dx = np.linalg.norm(x - x1)
                dTx = np.linalg.norm(Tx - Tx1)
                
                if dx > 1e-10:
                    max_ratio = max(max_ratio, dTx / dx)
                
                x = T(x)
            
            return max_ratio
        
        def convergence_analysis(history: List[np.ndarray], 
                               x_star: np.ndarray) -> Dict:
            """Analyze convergence from iteration history."""
            errors = [np.linalg.norm(x - x_star) for x in history]
            
            # Estimate convergence rate
            rates = []
            for i in range(1, len(errors) - 1):
                if errors[i] > 1e-14:
                    rates.append(errors[i+1] / errors[i])
            
            return {
                'final_error': errors[-1],
                'iterations': len(history) - 1,
                'estimated_rate': np.mean(rates) if rates else 0,
                'errors': errors[:10]  # First 10 errors
            }
        
        def value_iteration(P: np.ndarray, r: np.ndarray, 
                          gamma: float) -> np.ndarray:
            """
            Solve V = r + γPV via fixed point iteration.
            
            Bellman operator: T(V) = r + γPV
            Contraction with α = γ
            """
            def bellman(V):
                return r + gamma * P @ V
            
            V0 = np.zeros_like(r)
            V_star, iters, history = fixed_point_iteration(bellman, V0)
            
            return V_star, iters, history
        
        print("\nSolution:")
        
        # Example 1: sqrt(2) via Newton
        print("  Finding √2 via x_{n+1} = (x_n + 2/x_n)/2:")
        T = lambda x: (x + 2/x) / 2
        x0 = np.array([1.0])
        
        x_star, iters, history = fixed_point_iteration(T, x0)
        alpha = estimate_contraction_constant(T, np.array([1.5]))
        
        print(f"    Fixed point: {x_star[0]:.10f}")
        print(f"    √2 = {np.sqrt(2):.10f}")
        print(f"    Iterations: {iters}")
        print(f"    Estimated α: {alpha:.4f}")
        
        # Example 2: Value iteration
        print(f"\n  Value iteration (γ=0.9):")
        
        P = np.array([[0.7, 0.2, 0.1],
                     [0.1, 0.8, 0.1],
                     [0.2, 0.2, 0.6]])
        r = np.array([1.0, 2.0, 0.5])
        gamma = 0.9
        
        V_star, iters, history = value_iteration(P, r, gamma)
        
        print(f"    Optimal values: {V_star}")
        print(f"    Iterations: {iters}")
        print(f"    Contraction constant = γ = {gamma}")
        
        analysis = convergence_analysis(
            [np.array(h) for h in history], V_star
        )
        print(f"    Estimated convergence rate: {analysis['estimated_rate']:.4f}")
        
        return fixed_point_iteration, value_iteration


# =============================================================================
# Exercise 7: Regularization Comparison
# =============================================================================

class Exercise7:
    """
    Compare L1, L2, and elastic net regularization.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 7: Regularization Comparison")
        print("-" * 50)
        print("""
        Compare different regularization methods:
        
        1. L2 (Ridge): min ||y - Xβ||² + λ||β||₂²
           - Solution: β = (X^TX + λI)^{-1}X^Ty
           - Shrinks coefficients
        
        2. L1 (Lasso): min ||y - Xβ||² + λ||β||₁
           - Promotes sparsity
           - Requires iterative solution
        
        3. Elastic Net: min ||y - Xβ||² + λ₁||β||₁ + λ₂||β||₂²
           - Combines both effects
        
        Tasks:
        - Implement all three methods
        - Compare sparsity and recovery
        - Analyze bias-variance tradeoff
        """)
    
    @staticmethod
    def solution():
        def ridge_regression(X: np.ndarray, y: np.ndarray,
                           lambda_: float) -> np.ndarray:
            """L2 regularized regression."""
            n, p = X.shape
            return np.linalg.solve(X.T @ X + lambda_ * np.eye(p), X.T @ y)
        
        def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
            """Soft thresholding operator for L1."""
            return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        
        def lasso_regression(X: np.ndarray, y: np.ndarray,
                           lambda_: float, max_iter: int = 1000,
                           tol: float = 1e-6) -> np.ndarray:
            """L1 regularized regression via coordinate descent."""
            n, p = X.shape
            beta = np.zeros(p)
            
            # Precompute
            XX = X.T @ X
            Xy = X.T @ y
            
            for _ in range(max_iter):
                beta_old = beta.copy()
                
                for j in range(p):
                    # Partial residual
                    r_j = Xy[j] - XX[j] @ beta + XX[j, j] * beta[j]
                    
                    # Soft threshold update
                    beta[j] = soft_threshold(np.array([r_j]), lambda_/2)[0] / XX[j, j]
                
                if np.linalg.norm(beta - beta_old) < tol:
                    break
            
            return beta
        
        def elastic_net(X: np.ndarray, y: np.ndarray,
                       lambda1: float, lambda2: float,
                       max_iter: int = 1000) -> np.ndarray:
            """Elastic net via coordinate descent."""
            n, p = X.shape
            beta = np.zeros(p)
            
            for _ in range(max_iter):
                beta_old = beta.copy()
                
                for j in range(p):
                    r = y - X @ beta + X[:, j] * beta[j]
                    z_j = X[:, j] @ r
                    x_norm_sq = X[:, j] @ X[:, j]
                    
                    # Update with both penalties
                    beta[j] = soft_threshold(np.array([z_j]), lambda1/2)[0] / (x_norm_sq + lambda2)
                
                if np.linalg.norm(beta - beta_old) < 1e-6:
                    break
            
            return beta
        
        def compare_methods(X: np.ndarray, y: np.ndarray,
                          beta_true: np.ndarray,
                          lambda_: float) -> Dict:
            """Compare all three regularization methods."""
            results = {}
            
            # Ridge
            beta_ridge = ridge_regression(X, y, lambda_)
            results['Ridge'] = {
                'beta': beta_ridge,
                'sparsity': np.sum(np.abs(beta_ridge) < 0.01),
                'error': np.linalg.norm(beta_ridge - beta_true)
            }
            
            # Lasso
            beta_lasso = lasso_regression(X, y, lambda_)
            results['Lasso'] = {
                'beta': beta_lasso,
                'sparsity': np.sum(np.abs(beta_lasso) < 0.01),
                'error': np.linalg.norm(beta_lasso - beta_true)
            }
            
            # Elastic Net
            beta_enet = elastic_net(X, y, lambda_/2, lambda_/2)
            results['Elastic Net'] = {
                'beta': beta_enet,
                'sparsity': np.sum(np.abs(beta_enet) < 0.01),
                'error': np.linalg.norm(beta_enet - beta_true)
            }
            
            return results
        
        print("\nSolution:")
        
        # Generate sparse problem
        np.random.seed(42)
        n, p = 100, 20
        
        # True sparse coefficients
        beta_true = np.zeros(p)
        beta_true[:5] = [3, -2, 1.5, -1, 0.5]
        
        X = np.random.randn(n, p)
        y = X @ beta_true + 0.5 * np.random.randn(n)
        
        print(f"  Problem: n={n}, p={p}")
        print(f"  True sparsity: {np.sum(beta_true != 0)} non-zero coefficients")
        
        lambda_ = 1.0
        results = compare_methods(X, y, beta_true, lambda_)
        
        print(f"\n  Results (λ = {lambda_}):")
        print(f"  {'Method':<15} {'Sparsity':<12} {'Error':<12}")
        print(f"  {'-'*40}")
        
        for method, res in results.items():
            print(f"  {method:<15} {res['sparsity']:<12} {res['error']:.4f}")
        
        return compare_methods


# =============================================================================
# Exercise 8: Operator Norm Computation
# =============================================================================

class Exercise8:
    """
    Compute operator norms for linear maps.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 8: Operator Norm Computation")
        print("-" * 50)
        print("""
        The operator norm (induced norm) is:
        
            ||A||_{p→q} = sup_{||x||_p = 1} ||Ax||_q
        
        Common cases:
        - ||A||_{2→2} = σ_max(A) (spectral norm)
        - ||A||_{1→1} = max column sum of |A|
        - ||A||_{∞→∞} = max row sum of |A|
        
        Tasks:
        1. Implement operator norm for p = q = 1, 2, ∞
        2. Verify submultiplicativity
        3. For non-square matrices, handle different input/output dims
        4. Apply to analyzing layer-wise growth in neural networks
        """)
    
    @staticmethod
    def solution():
        def operator_norm(A: np.ndarray, p: float = 2) -> float:
            """
            Compute ||A||_{p→p}.
            """
            if p == 1:
                # Max column l1 norm
                return np.max(np.sum(np.abs(A), axis=0))
            elif p == 2:
                # Largest singular value
                return np.linalg.norm(A, 2)
            elif p == np.inf:
                # Max row l1 norm
                return np.max(np.sum(np.abs(A), axis=1))
            else:
                # Numerical optimization for general p
                m, n = A.shape
                from scipy.optimize import minimize
                
                def neg_ratio(x):
                    x = x / np.linalg.norm(x, p)
                    return -np.linalg.norm(A @ x, p)
                
                x0 = np.random.randn(n)
                result = minimize(neg_ratio, x0, method='BFGS')
                return -result.fun
        
        def verify_operator_norm_properties(A: np.ndarray, B: np.ndarray,
                                           p: float = 2) -> Dict[str, bool]:
            """Verify key properties of operator norms."""
            results = {}
            
            # ||AB|| <= ||A|| ||B||
            if A.shape[1] == B.shape[0]:
                lhs = operator_norm(A @ B, p)
                rhs = operator_norm(A, p) * operator_norm(B, p)
                results['Submultiplicative'] = lhs <= rhs + 1e-10
            
            # ||cA|| = |c| ||A||
            c = -2.5
            lhs = operator_norm(c * A, p)
            rhs = abs(c) * operator_norm(A, p)
            results['Homogeneous'] = np.isclose(lhs, rhs)
            
            # ||A + B|| <= ||A|| + ||B||
            if A.shape == B.shape:
                lhs = operator_norm(A + B, p)
                rhs = operator_norm(A, p) + operator_norm(B, p)
                results['Triangle'] = lhs <= rhs + 1e-10
            
            return results
        
        def network_growth_analysis(layer_weights: List[np.ndarray],
                                   p: float = 2) -> Dict:
            """Analyze signal growth through network layers."""
            results = {
                'layer_norms': [],
                'cumulative_norm': 1.0,
                'max_amplification': []
            }
            
            cum = 1.0
            for i, W in enumerate(layer_weights):
                norm = operator_norm(W, p)
                results['layer_norms'].append(norm)
                cum *= norm
                results['max_amplification'].append(cum)
            
            results['cumulative_norm'] = cum
            return results
        
        print("\nSolution:")
        
        A = np.array([[1, 2, 3],
                     [4, 5, 6]], dtype=float)
        
        print(f"  Matrix A (2×3):")
        print(f"    ||A||_1 = {operator_norm(A, 1):.4f} (max col sum)")
        print(f"    ||A||_2 = {operator_norm(A, 2):.4f} (spectral)")
        print(f"    ||A||_∞ = {operator_norm(A, np.inf):.4f} (max row sum)")
        
        # Verify properties
        B = np.random.randn(3, 2)
        print(f"\n  Properties (p=2):")
        for prop, satisfied in verify_operator_norm_properties(A, B, 2).items():
            print(f"    {prop}: {'✓' if satisfied else '✗'}")
        
        # Network analysis
        print(f"\n  Neural network layer analysis:")
        layers = [np.random.randn(128, 64),
                 np.random.randn(64, 32),
                 np.random.randn(32, 10)]
        
        analysis = network_growth_analysis(layers)
        
        print(f"    Layer norms: {[f'{n:.2f}' for n in analysis['layer_norms']]}")
        print(f"    Max amplification: {[f'{a:.2f}' for a in analysis['max_amplification']]}")
        print(f"    Total amplification: {analysis['cumulative_norm']:.2f}")
        
        return operator_norm


# =============================================================================
# Exercise 9: Proximal Operators
# =============================================================================

class Exercise9:
    """
    Implement proximal operators for regularization.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 9: Proximal Operators")
        print("-" * 50)
        print("""
        The proximal operator is:
        
            prox_{λf}(v) = argmin_x { f(x) + (1/2λ)||x - v||² }
        
        Common proximal operators:
        - prox_{λ||·||₁}(v) = soft_threshold(v, λ)
        - prox_{λ||·||₂}(v) = (1 - λ/||v||)₊ v  (group lasso)
        - prox_{λ||·||²}(v) = v/(1 + 2λ)
        
        Tasks:
        1. Implement proximal operators
        2. Use in proximal gradient descent
        3. Apply to sparse signal recovery
        """)
    
    @staticmethod
    def solution():
        def prox_l1(v: np.ndarray, lambda_: float) -> np.ndarray:
            """Proximal operator for L1 norm (soft thresholding)."""
            return np.sign(v) * np.maximum(np.abs(v) - lambda_, 0)
        
        def prox_l2(v: np.ndarray, lambda_: float) -> np.ndarray:
            """Proximal operator for L2 norm (group shrinkage)."""
            norm_v = np.linalg.norm(v)
            if norm_v == 0:
                return np.zeros_like(v)
            return max(1 - lambda_ / norm_v, 0) * v
        
        def prox_l2_squared(v: np.ndarray, lambda_: float) -> np.ndarray:
            """Proximal operator for squared L2 norm."""
            return v / (1 + 2 * lambda_)
        
        def prox_nuclear(V: np.ndarray, lambda_: float) -> np.ndarray:
            """Proximal operator for nuclear norm (singular value thresholding)."""
            U, s, Vt = np.linalg.svd(V, full_matrices=False)
            s_thresh = np.maximum(s - lambda_, 0)
            return U @ np.diag(s_thresh) @ Vt
        
        def proximal_gradient_descent(X: np.ndarray, y: np.ndarray,
                                     lambda_: float,
                                     prox: Callable,
                                     max_iter: int = 1000,
                                     step_size: float = None) -> np.ndarray:
            """
            Proximal gradient descent for:
            min (1/2)||y - Xβ||² + λ g(β)
            
            β_{k+1} = prox_{t·λ·g}(β_k - t·X^T(Xβ_k - y))
            """
            n, p = X.shape
            beta = np.zeros(p)
            
            # Step size from Lipschitz constant
            if step_size is None:
                L = np.linalg.norm(X.T @ X, 2)
                step_size = 1 / L
            
            for _ in range(max_iter):
                # Gradient step
                grad = X.T @ (X @ beta - y)
                beta_grad = beta - step_size * grad
                
                # Proximal step
                beta_new = prox(beta_grad, step_size * lambda_)
                
                if np.linalg.norm(beta_new - beta) < 1e-8:
                    break
                
                beta = beta_new
            
            return beta
        
        print("\nSolution:")
        
        # Demonstrate proximal operators
        v = np.array([3.0, -1.0, 0.5, -2.0, 0.1])
        lambda_ = 0.5
        
        print(f"  Original v = {v}")
        print(f"  λ = {lambda_}")
        print(f"\n  Proximal operators:")
        print(f"    prox_L1(v) = {prox_l1(v, lambda_)}")
        print(f"    prox_L2(v) = {prox_l2(v, lambda_)}")
        print(f"    prox_L2²(v) = {prox_l2_squared(v, lambda_)}")
        
        # Sparse signal recovery
        print(f"\n  Sparse signal recovery:")
        np.random.seed(42)
        
        n, p = 50, 100
        k = 10  # sparsity
        
        # True sparse signal
        beta_true = np.zeros(p)
        support = np.random.choice(p, k, replace=False)
        beta_true[support] = np.random.randn(k) * 2
        
        X = np.random.randn(n, p) / np.sqrt(n)
        y = X @ beta_true + 0.1 * np.random.randn(n)
        
        # Recover with proximal gradient
        beta_recovered = proximal_gradient_descent(X, y, 0.1, prox_l1)
        
        print(f"    True support: {sorted(support)[:5]}... ({k} non-zero)")
        print(f"    Recovered non-zero: {np.sum(np.abs(beta_recovered) > 0.01)}")
        print(f"    Recovery error: {np.linalg.norm(beta_recovered - beta_true):.4f}")
        
        return prox_l1, prox_l2, proximal_gradient_descent


# =============================================================================
# Exercise 10: Complete Regularization Pipeline
# =============================================================================

class Exercise10:
    """
    Build a complete regularization pipeline with cross-validation.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 10: Complete Regularization Pipeline")
        print("-" * 50)
        print("""
        Build a production-ready regularization pipeline:
        
        1. Implement multiple regularization methods
        2. Use cross-validation for λ selection
        3. Standardize features
        4. Evaluate on held-out test set
        5. Compare sparsity, accuracy, and stability
        """)
    
    @staticmethod
    def solution():
        class RegularizedRegressor:
            """Complete regularized regression with CV."""
            
            def __init__(self, method: str = 'lasso'):
                self.method = method
                self.beta = None
                self.mean = None
                self.std = None
                self.best_lambda = None
            
            def _standardize(self, X: np.ndarray, 
                           fit: bool = False) -> np.ndarray:
                if fit:
                    self.mean = X.mean(axis=0)
                    self.std = X.std(axis=0) + 1e-10
                return (X - self.mean) / self.std
            
            def _solve(self, X: np.ndarray, y: np.ndarray,
                      lambda_: float) -> np.ndarray:
                n, p = X.shape
                
                if self.method == 'ridge':
                    return np.linalg.solve(X.T @ X + lambda_ * np.eye(p), X.T @ y)
                
                elif self.method == 'lasso':
                    beta = np.zeros(p)
                    for _ in range(1000):
                        beta_old = beta.copy()
                        for j in range(p):
                            r = y - X @ beta + X[:, j] * beta[j]
                            z = X[:, j] @ r
                            d = X[:, j] @ X[:, j]
                            beta[j] = np.sign(z) * max(abs(z) - lambda_/2, 0) / d
                        if np.linalg.norm(beta - beta_old) < 1e-6:
                            break
                    return beta
                
                elif self.method == 'elastic_net':
                    beta = np.zeros(p)
                    l1 = lambda_ / 2
                    l2 = lambda_ / 2
                    for _ in range(1000):
                        beta_old = beta.copy()
                        for j in range(p):
                            r = y - X @ beta + X[:, j] * beta[j]
                            z = X[:, j] @ r
                            d = X[:, j] @ X[:, j] + l2
                            beta[j] = np.sign(z) * max(abs(z) - l1, 0) / d
                        if np.linalg.norm(beta - beta_old) < 1e-6:
                            break
                    return beta
                
                else:
                    raise ValueError(f"Unknown method: {self.method}")
            
            def cross_validate(self, X: np.ndarray, y: np.ndarray,
                             lambdas: List[float],
                             n_folds: int = 5) -> float:
                """Select best λ via cross-validation."""
                n = len(y)
                fold_size = n // n_folds
                
                cv_scores = []
                
                for lambda_ in lambdas:
                    fold_errors = []
                    
                    for fold in range(n_folds):
                        # Split
                        val_start = fold * fold_size
                        val_end = (fold + 1) * fold_size
                        
                        val_idx = np.arange(val_start, val_end)
                        train_idx = np.concatenate([
                            np.arange(0, val_start),
                            np.arange(val_end, n)
                        ])
                        
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        # Fit
                        beta = self._solve(X_train, y_train, lambda_)
                        
                        # Evaluate
                        y_pred = X_val @ beta
                        mse = np.mean((y_val - y_pred) ** 2)
                        fold_errors.append(mse)
                    
                    cv_scores.append(np.mean(fold_errors))
                
                best_idx = np.argmin(cv_scores)
                self.best_lambda = lambdas[best_idx]
                
                return self.best_lambda
            
            def fit(self, X: np.ndarray, y: np.ndarray,
                   lambda_: float = None):
                """Fit the model."""
                X_std = self._standardize(X, fit=True)
                
                if lambda_ is None:
                    if self.best_lambda is None:
                        lambdas = np.logspace(-4, 2, 20)
                        self.cross_validate(X_std, y, lambdas)
                    lambda_ = self.best_lambda
                
                self.beta = self._solve(X_std, y, lambda_)
                return self
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                X_std = self._standardize(X)
                return X_std @ self.beta
            
            def sparsity(self, threshold: float = 0.01) -> int:
                return np.sum(np.abs(self.beta) < threshold)
        
        print("\nSolution:")
        
        # Generate data
        np.random.seed(42)
        n_train, n_test, p = 200, 50, 50
        
        beta_true = np.zeros(p)
        beta_true[:8] = [3, -2, 1.5, -1, 0.8, -0.5, 2, -1.2]
        
        X_train = np.random.randn(n_train, p)
        X_test = np.random.randn(n_test, p)
        
        y_train = X_train @ beta_true + 0.5 * np.random.randn(n_train)
        y_test = X_test @ beta_true + 0.5 * np.random.randn(n_test)
        
        print(f"  Data: n_train={n_train}, n_test={n_test}, p={p}")
        print(f"  True sparsity: {np.sum(beta_true != 0)} non-zero")
        
        # Compare methods
        results = {}
        for method in ['ridge', 'lasso', 'elastic_net']:
            model = RegularizedRegressor(method=method)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            
            results[method] = {
                'best_lambda': model.best_lambda,
                'sparsity': model.sparsity(),
                'test_mse': mse
            }
        
        print(f"\n  Results:")
        print(f"  {'Method':<15} {'Best λ':<12} {'Sparsity':<12} {'Test MSE':<12}")
        print(f"  {'-'*50}")
        
        for method, res in results.items():
            print(f"  {method:<15} {res['best_lambda']:.4f}      "
                  f"{res['sparsity']:<12} {res['test_mse']:.4f}")
        
        return RegularizedRegressor


def run_all_exercises():
    """Run all exercises."""
    print("=" * 70)
    print("NORMED SPACES - EXERCISES")
    print("=" * 70)
    
    Exercise1.problem()
    Exercise1.solution()
    
    Exercise2.problem()
    Exercise2.solution()
    
    Exercise3.problem()
    Exercise3.solution()
    
    Exercise4.problem()
    Exercise4.solution()
    
    Exercise5.problem()
    Exercise5.solution()
    
    Exercise6.problem()
    Exercise6.solution()
    
    Exercise7.problem()
    Exercise7.solution()
    
    Exercise8.problem()
    Exercise8.solution()
    
    Exercise9.problem()
    Exercise9.solution()
    
    Exercise10.problem()
    Exercise10.solution()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_exercises()
