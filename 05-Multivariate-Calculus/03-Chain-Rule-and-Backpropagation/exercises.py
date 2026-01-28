"""
Chain Rule and Backpropagation - Exercises
==========================================
Practice problems for chain rule and backpropagation.
"""

import numpy as np


class BackpropExercises:
    """Exercises for chain rule and backpropagation."""
    
    def exercise_1_scalar_chain(self):
        """
        Exercise 1: Scalar Chain Rule
        
        For f(x) = e^(sin(x²)), compute df/dx
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Scalar Chain Rule")
        
        print("\nf(x) = e^(sin(x²))")
        print("\nLet u = x², v = sin(u), f = e^v")
        
        print("\nChain rule:")
        print("  df/dx = df/dv · dv/du · du/dx")
        print("        = e^v · cos(u) · 2x")
        print("        = e^(sin(x²)) · cos(x²) · 2x")
        print("        = 2x · cos(x²) · e^(sin(x²))")
        
        x = 1.0
        
        # Analytical
        df_dx = 2*x * np.cos(x**2) * np.exp(np.sin(x**2))
        
        # Numerical
        h = 1e-7
        f = lambda x: np.exp(np.sin(x**2))
        df_dx_numerical = (f(x+h) - f(x-h)) / (2*h)
        
        print(f"\nAt x = {x}:")
        print(f"  Analytical: {df_dx:.6f}")
        print(f"  Numerical:  {df_dx_numerical:.6f}")
    
    def exercise_2_vector_chain(self):
        """
        Exercise 2: Vector Chain Rule
        
        For L = ||Ax - b||², compute ∂L/∂x
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Vector Chain Rule")
        
        print("\nL = ||Ax - b||² = (Ax - b)ᵀ(Ax - b)")
        
        print("\nLet y = Ax - b, then L = yᵀy")
        
        print("\n∂L/∂y = 2y = 2(Ax - b)")
        print("∂y/∂x = A")
        
        print("\n∂L/∂x = (∂y/∂x)ᵀ · ∂L/∂y")
        print("      = Aᵀ · 2(Ax - b)")
        print("      = 2Aᵀ(Ax - b)")
        
        # Verify
        A = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([1, 2, 3])
        x = np.array([0.5, 0.5])
        
        # Analytical
        grad_analytical = 2 * A.T @ (A @ x - b)
        
        # Numerical
        h = 1e-7
        grad_numerical = np.zeros(2)
        for i in range(2):
            x_plus = x.copy(); x_plus[i] += h
            x_minus = x.copy(); x_minus[i] -= h
            L_plus = np.sum((A @ x_plus - b)**2)
            L_minus = np.sum((A @ x_minus - b)**2)
            grad_numerical[i] = (L_plus - L_minus) / (2*h)
        
        print(f"\nVerification:")
        print(f"  Analytical: {grad_analytical.round(4)}")
        print(f"  Numerical:  {grad_numerical.round(4)}")
    
    def exercise_3_two_layer_backprop(self):
        """
        Exercise 3: Two-Layer Network Backprop
        
        Network: x → W₁ → ReLU → W₂ → MSE
        Derive all gradients
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Two-Layer Network Backprop")
        
        print("\nForward pass:")
        print("  z₁ = W₁x + b₁")
        print("  a₁ = ReLU(z₁)")
        print("  z₂ = W₂a₁ + b₂")
        print("  L = ||z₂ - y||²")
        
        print("\nBackward pass:")
        print("  ∂L/∂z₂ = 2(z₂ - y)")
        print("  ∂L/∂W₂ = (∂L/∂z₂) · a₁ᵀ")
        print("  ∂L/∂b₂ = ∂L/∂z₂")
        print("  ∂L/∂a₁ = W₂ᵀ · (∂L/∂z₂)")
        print("  ∂L/∂z₁ = (∂L/∂a₁) ⊙ ReLU'(z₁)")
        print("  ∂L/∂W₁ = (∂L/∂z₁) · xᵀ")
        print("  ∂L/∂b₁ = ∂L/∂z₁")
        
        # Implementation
        np.random.seed(42)
        x = np.array([1.0, 2.0])
        y = np.array([0.5])
        
        W1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        b1 = np.zeros(3)
        W2 = np.array([[0.1, 0.2, 0.3]])
        b2 = np.zeros(1)
        
        # Forward
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2
        L = np.sum((z2 - y)**2)
        
        # Backward
        dL_dz2 = 2 * (z2 - y)
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2
        dL_da1 = W2.T @ dL_dz2
        dL_dz1 = dL_da1.flatten() * (z1 > 0)
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1
        
        print(f"\n--- Implementation ---")
        print(f"Loss = {L:.4f}")
        print(f"∂L/∂W₂ = {dL_dW2.round(4)}")
        print(f"∂L/∂W₁ = \n{dL_dW1.round(4)}")
    
    def exercise_4_softmax_gradient(self):
        """
        Exercise 4: Softmax + Cross-Entropy Gradient
        
        Show that for softmax + cross-entropy, ∂L/∂z = p - y
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Softmax + Cross-Entropy Gradient")
        
        print("\nSetup:")
        print("  p = softmax(z): pᵢ = e^zᵢ / Σⱼ e^zⱼ")
        print("  L = -Σᵢ yᵢ log(pᵢ)")
        
        print("\nDerivation:")
        print("  ∂L/∂pᵢ = -yᵢ/pᵢ")
        print("  ∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ - pⱼ)")
        
        print("\n  ∂L/∂zⱼ = Σᵢ (∂L/∂pᵢ)(∂pᵢ/∂zⱼ)")
        print("         = Σᵢ (-yᵢ/pᵢ) · pᵢ(δᵢⱼ - pⱼ)")
        print("         = Σᵢ -yᵢ(δᵢⱼ - pⱼ)")
        print("         = -yⱼ + pⱼ Σᵢ yᵢ")
        print("         = pⱼ - yⱼ  (since Σyᵢ = 1)")
        
        print("\nTherefore: ∂L/∂z = p - y")
        
        # Verify
        def softmax(z):
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        
        z = np.array([1.0, 2.0, 3.0])
        y = np.array([0, 1, 0])  # One-hot
        p = softmax(z)
        
        grad_simple = p - y
        
        # Numerical
        h = 1e-7
        grad_numerical = np.zeros(3)
        for j in range(3):
            z_plus = z.copy(); z_plus[j] += h
            p_plus = softmax(z_plus)
            L_plus = -np.sum(y * np.log(p_plus + 1e-10))
            
            z_minus = z.copy(); z_minus[j] -= h
            p_minus = softmax(z_minus)
            L_minus = -np.sum(y * np.log(p_minus + 1e-10))
            
            grad_numerical[j] = (L_plus - L_minus) / (2*h)
        
        print(f"\nVerification:")
        print(f"  p - y = {grad_simple.round(4)}")
        print(f"  Numerical = {grad_numerical.round(4)}")
    
    def exercise_5_gradient_check(self):
        """
        Exercise 5: Implement Gradient Checking
        
        Write a function to check analytical vs numerical gradients.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Gradient Checking")
        
        def gradient_check(f, grad_f, theta, h=1e-7, verbose=True):
            """
            Check analytical gradient against numerical gradient.
            
            Returns True if check passes.
            """
            # Analytical
            grad_analytical = grad_f(theta)
            
            # Numerical
            grad_numerical = np.zeros_like(theta)
            for i in range(len(theta)):
                theta_plus = theta.copy()
                theta_plus[i] += h
                theta_minus = theta.copy()
                theta_minus[i] -= h
                grad_numerical[i] = (f(theta_plus) - f(theta_minus)) / (2*h)
            
            # Relative error
            rel_error = np.linalg.norm(grad_analytical - grad_numerical) / (
                np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical) + 1e-10
            )
            
            passed = rel_error < 1e-5
            
            if verbose:
                print(f"  Analytical: {grad_analytical.round(6)}")
                print(f"  Numerical:  {grad_numerical.round(6)}")
                print(f"  Rel error:  {rel_error:.2e}")
                print(f"  Status:     {'PASSED ✓' if passed else 'FAILED ✗'}")
            
            return passed
        
        # Test
        def f(theta):
            x, y = theta
            return x**2 * np.sin(y) + y**3
        
        def grad_f(theta):
            x, y = theta
            return np.array([
                2*x*np.sin(y),
                x**2*np.cos(y) + 3*y**2
            ])
        
        theta = np.array([2.0, 1.0])
        print(f"Testing at θ = {theta}")
        gradient_check(f, grad_f, theta)
    
    def exercise_6_batch_normalization(self):
        """
        Exercise 6: Batch Normalization Gradient
        
        For y = (x - μ) / σ where μ, σ are batch statistics,
        derive ∂L/∂x
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Batch Normalization Gradient")
        
        print("\nBatch normalization:")
        print("  μ = (1/n) Σᵢ xᵢ")
        print("  σ² = (1/n) Σᵢ (xᵢ - μ)²")
        print("  yᵢ = (xᵢ - μ) / σ")
        
        print("\nThe gradient is complex because μ and σ depend on all xᵢ!")
        
        print("\nKey insight:")
        print("  ∂μ/∂xⱼ = 1/n")
        print("  ∂σ²/∂xⱼ = (2/n)(xⱼ - μ)")
        
        print("\nFinal gradient (simplified):")
        print("  ∂L/∂xᵢ = (1/σ)[∂L/∂yᵢ - (1/n)Σⱼ∂L/∂yⱼ - (yᵢ/n)Σⱼyⱼ∂L/∂yⱼ]")
        
        # Implementation
        np.random.seed(42)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        n = len(x)
        
        # Forward
        mu = np.mean(x)
        var = np.var(x)
        sigma = np.sqrt(var + 1e-8)
        y = (x - mu) / sigma
        
        # Assume upstream gradient
        dL_dy = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Backward
        dL_dx = (1/sigma) * (
            dL_dy - np.mean(dL_dy) - y * np.mean(y * dL_dy)
        )
        
        print(f"\n--- Implementation ---")
        print(f"x = {x}")
        print(f"y = {y.round(4)}")
        print(f"∂L/∂y = {dL_dy}")
        print(f"∂L/∂x = {dL_dx.round(4)}")
    
    def exercise_7_attention_gradient(self):
        """
        Exercise 7: Scaled Dot-Product Attention Gradient
        
        For attention = softmax(QKᵀ/√d)V, find ∂L/∂Q
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Attention Gradient")
        
        print("\nAttention mechanism:")
        print("  S = QKᵀ / √d  (scores)")
        print("  A = softmax(S)  (attention weights)")
        print("  O = AV  (output)")
        
        print("\nBackward pass:")
        print("  ∂L/∂A = (∂L/∂O) Vᵀ")
        print("  ∂L/∂S = softmax_backward(∂L/∂A)")
        print("  ∂L/∂Q = (∂L/∂S) K / √d")
        
        # Simple implementation
        np.random.seed(42)
        d = 4
        seq_len = 3
        
        Q = np.random.randn(seq_len, d) * 0.1
        K = np.random.randn(seq_len, d) * 0.1
        V = np.random.randn(seq_len, d) * 0.1
        
        # Forward
        scores = Q @ K.T / np.sqrt(d)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        O = A @ V
        L = np.sum(O**2)  # Simple loss
        
        print(f"\nDimensions:")
        print(f"  Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
        print(f"  Scores: {scores.shape}, A: {A.shape}, O: {O.shape}")
        
        # Backward
        dL_dO = 2 * O
        dL_dA = dL_dO @ V.T
        
        # Softmax backward: dL/dS_ij = A_ij * (dL/dA_ij - Σ_k A_ik * dL/dA_ik)
        dL_dS = A * (dL_dA - np.sum(A * dL_dA, axis=1, keepdims=True))
        
        dL_dQ = dL_dS @ K / np.sqrt(d)
        
        print(f"\nGradients:")
        print(f"  ∂L/∂Q shape: {dL_dQ.shape}")
    
    def exercise_8_residual_connection(self):
        """
        Exercise 8: Residual Connection Gradient
        
        For y = x + f(x), show that gradient flows directly through skip connection.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Residual Connection Gradient")
        
        print("\nResidual block: y = x + f(x)")
        
        print("\n∂y/∂x = 1 + ∂f/∂x")
        print("\n∂L/∂x = ∂L/∂y · ∂y/∂x")
        print("      = ∂L/∂y · (1 + ∂f/∂x)")
        print("      = ∂L/∂y + ∂L/∂y · ∂f/∂x")
        
        print("\nThe '1' in the gradient means:")
        print("  - Gradient flows DIRECTLY from output to input")
        print("  - Even if ∂f/∂x → 0 (vanishing), gradient still flows!")
        print("  - This enables training very deep networks")
        
        # Demonstration
        print("\n--- Demonstration ---")
        
        def deep_net_without_residual(x, n_layers):
            """Chain of tanh layers - gradients vanish."""
            for _ in range(n_layers):
                x = np.tanh(x)
            return x
        
        def deep_net_with_residual(x, n_layers):
            """Chain of tanh layers with residual connections."""
            for _ in range(n_layers):
                x = x + np.tanh(x)  # Residual!
            return x
        
        x = 0.5
        n_layers = 20
        h = 1e-7
        
        # Gradient without residual
        y1 = deep_net_without_residual(x, n_layers)
        grad1 = (deep_net_without_residual(x+h, n_layers) - 
                 deep_net_without_residual(x-h, n_layers)) / (2*h)
        
        # Gradient with residual
        y2 = deep_net_with_residual(x, n_layers)
        grad2 = (deep_net_with_residual(x+h, n_layers) - 
                 deep_net_with_residual(x-h, n_layers)) / (2*h)
        
        print(f"With {n_layers} layers:")
        print(f"  Without residual: gradient = {grad1:.6f}")
        print(f"  With residual:    gradient = {grad2:.6f}")
        print("\nResidual connections prevent vanishing gradients!")
    
    def exercise_9_lstm_gradient(self):
        """
        Exercise 9: LSTM Gradient Flow
        
        Explain why LSTM solves vanishing gradient better than vanilla RNN.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: LSTM Gradient Flow")
        
        print("\nVanilla RNN:")
        print("  hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ)")
        print("  ∂hₜ/∂hₜ₋₁ = Wₕ · diag(tanh')")
        print("  Over T steps: gradient scales as (Wₕ)^T · Π tanh' → vanishes!")
        
        print("\nLSTM:")
        print("  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ  (cell state update)")
        print("\n  Key insight: ∂cₜ/∂cₜ₋₁ = fₜ (forget gate)")
        print("  If fₜ ≈ 1, gradient flows unimpeded through time!")
        
        print("\nGradient path through cell state:")
        print("  ∂cₜ/∂c₁ = Π_{i=2}^{T} fᵢ")
        print("  If all forget gates ≈ 1, gradient ≈ 1")
        
        print("\nThis is similar to residual connections!")
        print("  - Cell state acts like a 'highway' for gradients")
        print("  - Gates learn what to remember/forget")
    
    def exercise_10_custom_layer(self):
        """
        Exercise 10: Implement Custom Layer Backward Pass
        
        Implement forward and backward for: y = x³ + 2x
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Custom Layer Implementation")
        
        class CubicLayer:
            """Layer computing y = x³ + 2x"""
            
            def forward(self, x):
                self.x = x  # Cache for backward
                return x**3 + 2*x
            
            def backward(self, grad_output):
                # dy/dx = 3x² + 2
                return grad_output * (3 * self.x**2 + 2)
        
        # Test
        layer = CubicLayer()
        x = np.array([1.0, 2.0, 3.0])
        
        # Forward
        y = layer.forward(x)
        print(f"x = {x}")
        print(f"y = x³ + 2x = {y}")
        
        # Backward
        grad_output = np.array([1.0, 1.0, 1.0])
        grad_input = layer.backward(grad_output)
        print(f"\n∂L/∂y = {grad_output}")
        print(f"∂L/∂x = ∂L/∂y · (3x² + 2) = {grad_input}")
        
        # Verify
        print("\n--- Verification ---")
        print(f"3x² + 2 at x = {x}: {3*x**2 + 2}")
        
        # Numerical check
        h = 1e-7
        grad_numerical = np.zeros(3)
        for i in range(3):
            x_plus = x.copy(); x_plus[i] += h
            x_minus = x.copy(); x_minus[i] -= h
            y_plus = x_plus**3 + 2*x_plus
            y_minus = x_minus**3 + 2*x_minus
            # Assuming L = sum(y)
            grad_numerical[i] = (np.sum(y_plus) - np.sum(y_minus)) / (2*h)
        
        print(f"Numerical gradient: {grad_numerical.round(4)}")
        print(f"Matches analytical: {np.allclose(grad_input, grad_numerical)}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = BackpropExercises()
    
    print("CHAIN RULE AND BACKPROPAGATION EXERCISES")
    print("=" * 70)
    
    exercises.solution_1()
    print("\n" + "=" * 70)
    
    exercises.solution_2()
    print("\n" + "=" * 70)
    
    exercises.solution_3()
    print("\n" + "=" * 70)
    
    exercises.solution_4()
    print("\n" + "=" * 70)
    
    exercises.solution_5()
    print("\n" + "=" * 70)
    
    exercises.solution_6()
    print("\n" + "=" * 70)
    
    exercises.solution_7()
    print("\n" + "=" * 70)
    
    exercises.solution_8()
    print("\n" + "=" * 70)
    
    exercises.solution_9()
    print("\n" + "=" * 70)
    
    exercises.solution_10()


if __name__ == "__main__":
    run_all_exercises()
