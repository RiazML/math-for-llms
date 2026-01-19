"""
Optimization Landscape Analysis - Exercises
===========================================
Practice problems for loss surface analysis.
"""

import numpy as np


class OptimizationLandscapeExercises:
    """Exercises for optimization landscape analysis."""
    
    def exercise_1_critical_point_classification(self):
        """
        Exercise 1: Classify Critical Points
        
        Find and classify all critical points of a function.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Critical Point Classification")
        print("=" * 60)
        
        print("f(x,y) = x³ - 3x + y²")
        print("\n∇f = [3x² - 3, 2y]")
        print("Setting ∇f = 0:")
        print("  3x² - 3 = 0 → x = ±1")
        print("  2y = 0 → y = 0")
        print("\nCritical points: (1, 0) and (-1, 0)")
        
        print("\nHessian:")
        print("H = [[6x, 0], [0, 2]]")
        
        print("\nAt (1, 0):")
        H1 = np.array([[6, 0], [0, 2]])
        eigvals1 = np.linalg.eigvalsh(H1)
        print(f"  H = {H1.tolist()}")
        print(f"  Eigenvalues: {eigvals1}")
        print("  Classification: LOCAL MINIMUM (all positive)")
        
        print("\nAt (-1, 0):")
        H2 = np.array([[-6, 0], [0, 2]])
        eigvals2 = np.linalg.eigvalsh(H2)
        print(f"  H = {H2.tolist()}")
        print(f"  Eigenvalues: {eigvals2}")
        print("  Classification: SADDLE POINT (mixed signs)")
    
    def exercise_2_condition_number(self):
        """
        Exercise 2: Compute Condition Number
        
        Analyze condition number and its effect.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Condition Number Analysis")
        print("=" * 60)
        
        # Different Hessians
        matrices = {
            'A₁': np.array([[2, 0], [0, 2]]),
            'A₂': np.array([[10, 0], [0, 1]]),
            'A₃': np.array([[5, 4], [4, 5]]),
            'A₄': np.array([[100, 99], [99, 100]])
        }
        
        print("Condition number κ = λ_max / λ_min")
        print(f"\n{'Matrix':>6} {'λ_min':>10} {'λ_max':>10} {'κ':>10} {'GD rate':>12}")
        print("-" * 55)
        
        for name, A in matrices.items():
            eigvals = np.linalg.eigvalsh(A)
            lambda_min, lambda_max = eigvals[0], eigvals[-1]
            kappa = lambda_max / lambda_min
            gd_rate = (kappa - 1) / (kappa + 1)  # Convergence rate
            
            print(f"{name:>6} {lambda_min:>10.2f} {lambda_max:>10.2f} "
                  f"{kappa:>10.2f} {gd_rate:>12.4f}")
        
        print("\nGD convergence rate = (κ-1)/(κ+1)")
        print("Higher κ → slower convergence")
    
    def exercise_3_saddle_detection(self):
        """
        Exercise 3: Detect Saddle Points
        
        Identify saddle points in a loss landscape.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Saddle Point Detection")
        print("=" * 60)
        
        def f(x, y):
            return x**4 - 2*x**2 + y**2
        
        def grad(x, y):
            return np.array([4*x**3 - 4*x, 2*y])
        
        def hessian(x, y):
            return np.array([[12*x**2 - 4, 0], [0, 2]])
        
        print("f(x,y) = x⁴ - 2x² + y²")
        print("\nFinding critical points (∇f = 0):")
        print("  4x³ - 4x = 0 → x(x² - 1) = 0 → x ∈ {-1, 0, 1}")
        print("  2y = 0 → y = 0")
        
        critical_points = [(0, 0), (1, 0), (-1, 0)]
        
        print(f"\n{'Point':>10} {'f value':>10} {'Eigenvalues':>20} {'Type':>15}")
        print("-" * 60)
        
        for x, y in critical_points:
            H = hessian(x, y)
            eigvals = np.linalg.eigvalsh(H)
            f_val = f(x, y)
            
            if np.all(eigvals > 0):
                pt_type = "Local minimum"
            elif np.all(eigvals < 0):
                pt_type = "Local maximum"
            else:
                pt_type = "Saddle point"
            
            print(f"{str((x,y)):>10} {f_val:>10.2f} "
                  f"{str(np.round(eigvals, 2)):>20} {pt_type:>15}")
    
    def exercise_4_hessian_spectrum(self):
        """
        Exercise 4: Analyze Hessian Spectrum
        
        Study eigenvalue distribution.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Hessian Spectrum")
        print("=" * 60)
        
        np.random.seed(42)
        
        d = 50
        
        # Create Hessian with NN-like spectrum
        # Bulk: small eigenvalues, Outliers: large eigenvalues
        bulk = 0.1 * np.abs(np.random.randn(d - 3))
        outliers = np.array([10.0, 5.0, 2.0])
        eigvals_true = np.concatenate([bulk, outliers])
        
        # Create symmetric positive definite matrix
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        H = Q @ np.diag(eigvals_true) @ Q.T
        
        # Compute eigenvalues
        eigvals = np.sort(np.linalg.eigvalsh(H))[::-1]
        
        print(f"Hessian dimension: {d} × {d}")
        
        print("\nEigenvalue statistics:")
        print(f"  Max:  {eigvals[0]:.4f}")
        print(f"  Min:  {eigvals[-1]:.4f}")
        print(f"  Mean: {np.mean(eigvals):.4f}")
        print(f"  κ:    {eigvals[0]/eigvals[-1]:.1f}")
        
        # Distribution
        thresholds = [0.1, 0.5, 1.0, 2.0]
        print("\nEigenvalue distribution:")
        for t in thresholds:
            frac = np.mean(eigvals > t)
            print(f"  Fraction > {t}: {frac:.2%}")
        
        print("\nEffective dimension (eigenvalues > 1% of max):")
        eff_dim = np.sum(eigvals > 0.01 * eigvals[0])
        print(f"  {eff_dim} out of {d}")
    
    def exercise_5_flatness_measure(self):
        """
        Exercise 5: Measure Flatness
        
        Compute flatness measures of minima.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Flatness Measures")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 100, 10
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        # Train to minimum
        w_star = np.linalg.lstsq(X, y, rcond=None)[0]
        
        print("Linear regression solution found")
        print(f"Loss at minimum: {loss(w_star):.6f}")
        
        # Compute Hessian
        H = X.T @ X / n
        eigvals = np.linalg.eigvalsh(H)
        
        print("\nFlatness measures:")
        
        # 1. Maximum eigenvalue (sharpness)
        sharpness = np.max(eigvals)
        print(f"  1. Sharpness (λ_max): {sharpness:.4f}")
        
        # 2. Trace of Hessian
        trace_H = np.sum(eigvals)
        print(f"  2. Trace(H): {trace_H:.4f}")
        
        # 3. Determinant (volume of ellipsoid)
        log_det = np.sum(np.log(eigvals + 1e-10))
        print(f"  3. log|H|: {log_det:.4f}")
        
        # 4. Random perturbation test
        n_samples = 100
        eps = 0.1
        perturbed_losses = []
        for _ in range(n_samples):
            delta = eps * np.random.randn(d)
            perturbed_losses.append(loss(w_star + delta))
        
        flatness_pert = np.mean(perturbed_losses) - loss(w_star)
        print(f"  4. E[f(w* + ε) - f(w*)] (ε~N(0,0.1²I)): {flatness_pert:.6f}")
    
    def exercise_6_gradient_flow(self):
        """
        Exercise 6: Gradient Flow Analysis
        
        Analyze continuous-time gradient dynamics.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Gradient Flow")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Simple quadratic
        A = np.array([[4, 1], [1, 2]])
        b = np.array([1, 1])
        
        def loss(w):
            return 0.5 * w @ A @ w - b @ w
        
        def gradient(w):
            return A @ w - b
        
        w_opt = np.linalg.solve(A, b)
        eigvals = np.linalg.eigvalsh(A)
        
        print("Gradient flow: dw/dt = -∇f(w)")
        print(f"For quadratic f(w) = 0.5 w'Aw - b'w:")
        print(f"  dw/dt = -(Aw - b)")
        print(f"\nEigenvalues of A: {eigvals}")
        print(f"Optimal: w* = {np.round(w_opt, 4)}")
        
        # Simulate gradient flow
        w = np.array([0.0, 0.0])
        dt = 0.01
        
        print(f"\n{'t':>8} {'||w - w*||':>15} {'f(w)':>15}")
        print("-" * 40)
        
        for step in range(501):
            t = step * dt
            if step % 100 == 0:
                error = np.linalg.norm(w - w_opt)
                f_val = loss(w)
                print(f"{t:>8.2f} {error:>15.6e} {f_val:>15.6f}")
            
            # Euler step
            w = w - dt * gradient(w)
        
        print("\nConvergence rate = λ_min = {:.4f}".format(eigvals[0]))
    
    def exercise_7_mode_connectivity(self):
        """
        Exercise 7: Test Mode Connectivity
        
        Check if two minima are connected.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Mode Connectivity")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Logistic regression
        n, d = 50, 10
        X = np.random.randn(n, d)
        y = np.sign(np.random.randn(n))
        
        def loss(w):
            z = np.clip(X @ w, -500, 500)
            return np.mean(np.log(1 + np.exp(-y * z)))
        
        def gradient(w):
            z = np.clip(X @ w, -500, 500)
            p = 1 / (1 + np.exp(-y * z))
            return -X.T @ (y * (1 - p)) / n
        
        # Train from two different initializations
        results = []
        for seed in [0, 100]:
            np.random.seed(seed)
            w = 0.1 * np.random.randn(d)
            for _ in range(500):
                w = w - 0.5 * gradient(w)
            results.append(w)
        
        w1, w2 = results
        print(f"Found two solutions:")
        print(f"  w₁ loss: {loss(w1):.6f}")
        print(f"  w₂ loss: {loss(w2):.6f}")
        print(f"  ||w₁ - w₂||: {np.linalg.norm(w1 - w2):.4f}")
        
        # Linear connectivity test
        print("\nLinear interpolation:")
        max_loss = 0
        ts = np.linspace(0, 1, 21)
        for t in ts:
            w_t = (1-t) * w1 + t * w2
            l = loss(w_t)
            max_loss = max(max_loss, l)
        
        barrier = max_loss - max(loss(w1), loss(w2))
        print(f"  Max interpolated loss: {max_loss:.6f}")
        print(f"  Barrier height: {barrier:.6f}")
        
        if barrier < 0.05:
            print("  → Solutions are approximately mode-connected!")
        else:
            print("  → Significant barrier between solutions")
    
    def exercise_8_curvature_analysis(self):
        """
        Exercise 8: Directional Curvature
        
        Analyze curvature in specific directions.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Directional Curvature")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Create Hessian
        d = 5
        A = np.random.randn(d, d)
        H = A.T @ A  # Positive definite
        
        eigvals, eigvecs = np.linalg.eigh(H)
        
        print(f"Hessian eigenvalues: {np.round(eigvals, 4)}")
        
        print("\nDirectional curvature κ_v = v'Hv/||v||²:")
        
        # Test directions
        directions = {
            'v₁ (min eigenvector)': eigvecs[:, 0],
            'v₅ (max eigenvector)': eigvecs[:, -1],
            'random 1': np.random.randn(d),
            'random 2': np.random.randn(d),
            'e₁': np.array([1, 0, 0, 0, 0])
        }
        
        print(f"\n{'Direction':>20} {'Curvature':>12}")
        print("-" * 35)
        
        for name, v in directions.items():
            v = v / np.linalg.norm(v)
            curvature = v @ H @ v
            print(f"{name:>20} {curvature:>12.4f}")
        
        print(f"\nMin possible curvature: {eigvals[0]:.4f}")
        print(f"Max possible curvature: {eigvals[-1]:.4f}")
    
    def exercise_9_sgd_escape(self):
        """
        Exercise 9: SGD Escaping Saddles
        
        Show how noise helps escape saddle points.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: SGD Escaping Saddles")
        print("=" * 60)
        
        np.random.seed(42)
        
        # f(x,y) = x² - y² (saddle at origin)
        def f(w):
            return w[0]**2 - w[1]**2
        
        def grad(w):
            return np.array([2*w[0], -2*w[1]])
        
        # Start near saddle
        w0 = np.array([0.01, 0.01])
        eta = 0.1
        
        print("f(x,y) = x² - y² (saddle at origin)")
        print(f"Starting at {w0}")
        
        # Pure GD
        w_gd = w0.copy()
        gd_history = [w_gd.copy()]
        
        for _ in range(100):
            w_gd = w_gd - eta * grad(w_gd)
            gd_history.append(w_gd.copy())
        
        # SGD (with noise)
        noise_levels = [0.01, 0.05, 0.1]
        sgd_results = {}
        
        for noise in noise_levels:
            np.random.seed(42)
            w_sgd = w0.copy()
            sgd_history = [w_sgd.copy()]
            
            for _ in range(100):
                g = grad(w_sgd) + noise * np.random.randn(2)
                w_sgd = w_sgd - eta * g
                sgd_history.append(w_sgd.copy())
            
            sgd_results[noise] = np.array(sgd_history)
        
        gd_history = np.array(gd_history)
        
        print(f"\n{'Method':>15} {'Final |y|':>12} {'Final f':>12}")
        print("-" * 45)
        
        final_w = gd_history[-1]
        print(f"{'GD':>15} {abs(final_w[1]):>12.6f} {f(final_w):>12.6f}")
        
        for noise, history in sgd_results.items():
            final_w = history[-1]
            print(f"{'SGD σ='+str(noise):>15} {abs(final_w[1]):>12.6f} {f(final_w):>12.6f}")
        
        print("\nNoise helps escape along negative curvature direction (y)")
    
    def exercise_10_sam_implementation(self):
        """
        Exercise 10: Implement SAM
        
        Implement Sharpness-Aware Minimization.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: SAM Implementation")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 100, 10
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + 0.2 * np.random.randn(n)
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        def gradient(w):
            return X.T @ (X @ w - y) / n
        
        eta = 0.1
        rho = 0.1  # Perturbation radius
        
        # Standard GD
        w_gd = np.zeros(d)
        for _ in range(100):
            w_gd = w_gd - eta * gradient(w_gd)
        
        # SAM
        w_sam = np.zeros(d)
        for _ in range(100):
            g = gradient(w_sam)
            # Ascent step to find worst perturbation
            eps = rho * g / (np.linalg.norm(g) + 1e-12)
            # Gradient at perturbed point
            g_sam = gradient(w_sam + eps)
            # Descent step
            w_sam = w_sam - eta * g_sam
        
        print("SAM Algorithm:")
        print("1. Compute ε = ρ × ∇f(w) / ||∇f(w)||")
        print("2. Compute gradient at w + ε")
        print("3. Update w using this gradient")
        
        print(f"\nρ = {rho}, η = {eta}")
        print(f"\n{'Method':>10} {'Loss':>12} {'||w||':>12}")
        print("-" * 40)
        print(f"{'GD':>10} {loss(w_gd):>12.6f} {np.linalg.norm(w_gd):>12.4f}")
        print(f"{'SAM':>10} {loss(w_sam):>12.6f} {np.linalg.norm(w_sam):>12.4f}")
        
        # Test robustness
        print("\nRobustness test (perturbation σ = 0.05):")
        
        n_test = 100
        gd_robust = []
        sam_robust = []
        
        for _ in range(n_test):
            eps = 0.05 * np.random.randn(d)
            gd_robust.append(loss(w_gd + eps))
            sam_robust.append(loss(w_sam + eps))
        
        print(f"{'':>10} {'Mean perturbed loss':>20}")
        print(f"{'GD':>10} {np.mean(gd_robust):>20.6f}")
        print(f"{'SAM':>10} {np.mean(sam_robust):>20.6f}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = OptimizationLandscapeExercises()
    
    print("OPTIMIZATION LANDSCAPE EXERCISES")
    print("=" * 70)
    
    exercises.solution_1()
    exercises.solution_2()
    exercises.solution_3()
    exercises.solution_4()
    exercises.solution_5()
    exercises.solution_6()
    exercises.solution_7()
    exercises.solution_8()
    exercises.solution_9()
    exercises.solution_10()


if __name__ == "__main__":
    run_all_exercises()
