"""
Expectation, Variance, and Moments - Exercises
=============================================
Practice problems for expectation and variance.
"""

import numpy as np
from scipy import stats


class ExpectationExercises:
    """Exercises for expectation and variance."""
    
    def exercise_1_linearity(self):
        """
        Exercise 1: Prove Linearity of Expectation
        
        Prove E[X + Y] = E[X] + E[Y] (no independence needed).
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Linearity of Expectation")
        
        print("\nProve: E[X + Y] = E[X] + E[Y]")
        
        print("\nProof (discrete case):")
        print("  E[X + Y] = Σₓ Σᵧ (x + y) P(X=x, Y=y)")
        print("           = Σₓ Σᵧ x·P(X=x, Y=y) + Σₓ Σᵧ y·P(X=x, Y=y)")
        print("           = Σₓ x Σᵧ P(X=x, Y=y) + Σᵧ y Σₓ P(X=x, Y=y)")
        print("           = Σₓ x·P(X=x) + Σᵧ y·P(Y=y)")
        print("           = E[X] + E[Y]  □")
        
        print("\nKey insight: We used Σᵧ P(X=x, Y=y) = P(X=x) (marginalization)")
        print("             No independence assumption needed!")
        
        print("\nGeneralization: E[Σᵢ aᵢXᵢ + b] = Σᵢ aᵢE[Xᵢ] + b")
        
        # Verify with dependent variables
        np.random.seed(42)
        n = 100000
        X = np.random.randn(n)
        Y = X + np.random.randn(n)  # Y depends on X!
        
        print("\n--- Verification with DEPENDENT variables ---")
        print(f"  X ~ N(0,1), Y = X + Z where Z ~ N(0,1)")
        print(f"  E[X] = {X.mean():.4f}")
        print(f"  E[Y] = {Y.mean():.4f}")
        print(f"  E[X] + E[Y] = {X.mean() + Y.mean():.4f}")
        print(f"  E[X + Y] = {(X+Y).mean():.4f}")
        print("  Equal! Linearity holds for dependent variables ✓")
    
    def exercise_2_sum_variance(self):
        """
        Exercise 2: Variance of Sum
        
        Derive Var(aX + bY) for correlated X, Y.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Variance of aX + bY")
        
        print("\nDerive: Var(aX + bY)")
        
        print("\nLet Z = aX + bY, μ_Z = aμ_X + bμ_Y")
        
        print("\nVar(Z) = E[(Z - μ_Z)²]")
        print("       = E[(aX + bY - aμ_X - bμ_Y)²]")
        print("       = E[(a(X-μ_X) + b(Y-μ_Y))²]")
        print("       = E[a²(X-μ_X)² + 2ab(X-μ_X)(Y-μ_Y) + b²(Y-μ_Y)²]")
        print("       = a²E[(X-μ_X)²] + 2abE[(X-μ_X)(Y-μ_Y)] + b²E[(Y-μ_Y)²]")
        print("       = a²Var(X) + 2ab·Cov(X,Y) + b²Var(Y)")
        
        print("\nSpecial cases:")
        print("  • a=b=1: Var(X+Y) = Var(X) + 2Cov(X,Y) + Var(Y)")
        print("  • a=1, b=-1: Var(X-Y) = Var(X) - 2Cov(X,Y) + Var(Y)")
        print("  • Independent: Var(aX+bY) = a²Var(X) + b²Var(Y)")
        
        # Verify
        np.random.seed(42)
        n = 100000
        X = np.random.normal(0, 2, n)
        Y = 0.6 * X + np.random.normal(0, 1.5, n)
        
        a, b = 2, 3
        Z = a*X + b*Y
        
        var_X = X.var()
        var_Y = Y.var()
        cov_XY = np.cov(X, Y)[0, 1]
        
        formula_result = a**2 * var_X + 2*a*b*cov_XY + b**2 * var_Y
        
        print(f"\n--- Verification ---")
        print(f"  a={a}, b={b}")
        print(f"  Var(X) = {var_X:.4f}")
        print(f"  Var(Y) = {var_Y:.4f}")
        print(f"  Cov(X,Y) = {cov_XY:.4f}")
        print(f"  Formula: {a}²×{var_X:.4f} + 2×{a}×{b}×{cov_XY:.4f} + {b}²×{var_Y:.4f}")
        print(f"         = {formula_result:.4f}")
        print(f"  Actual Var({a}X + {b}Y) = {Z.var():.4f}")
    
    def exercise_3_exponential_mgf(self):
        """
        Exercise 3: Exponential MGF
        
        Find MGF of Exponential(λ) and derive mean and variance.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Exponential MGF")
        
        print("\nX ~ Exponential(λ), f(x) = λe^{-λx} for x ≥ 0")
        
        print("\nMGF:")
        print("  M(t) = E[e^{tX}]")
        print("       = ∫₀^∞ e^{tx} · λe^{-λx} dx")
        print("       = λ ∫₀^∞ e^{(t-λ)x} dx")
        print("       = λ · [e^{(t-λ)x}/(t-λ)]₀^∞")
        print("       = λ · (0 - 1/(t-λ))    for t < λ")
        print("       = λ/(λ-t)")
        
        print("\nMean (first derivative at 0):")
        print("  M(t) = λ(λ-t)^{-1}")
        print("  M'(t) = λ(λ-t)^{-2}")
        print("  M'(0) = λ/λ² = 1/λ = E[X]")
        
        print("\nSecond moment:")
        print("  M''(t) = 2λ(λ-t)^{-3}")
        print("  M''(0) = 2λ/λ³ = 2/λ² = E[X²]")
        
        print("\nVariance:")
        print("  Var(X) = E[X²] - E[X]²")
        print("         = 2/λ² - (1/λ)²")
        print("         = 2/λ² - 1/λ²")
        print("         = 1/λ²")
        
        # Verify
        lam = 3
        np.random.seed(42)
        X = np.random.exponential(1/lam, 100000)
        
        print(f"\n--- Verification: λ = {lam} ---")
        print(f"  Theory: E[X] = 1/{lam} = {1/lam:.4f}")
        print(f"          Var(X) = 1/{lam}² = {1/lam**2:.4f}")
        print(f"  Simulated: E[X] = {X.mean():.4f}")
        print(f"             Var(X) = {X.var():.4f}")
    
    def exercise_4_jensen(self):
        """
        Exercise 4: Prove Jensen's Inequality (Discrete)
        
        For convex g: g(E[X]) ≤ E[g(X)]
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Jensen's Inequality")
        
        print("\nJensen's Inequality: For convex g, g(E[X]) ≤ E[g(X)]")
        
        print("\nProof (discrete case with two points):")
        print("  Let X take values x₁, x₂ with probs p, 1-p")
        print("  E[X] = px₁ + (1-p)x₂")
        
        print("\n  Since g is convex:")
        print("  g(px₁ + (1-p)x₂) ≤ pg(x₁) + (1-p)g(x₂)")
        print("  (definition of convexity)")
        
        print("\n  LHS = g(E[X])")
        print("  RHS = E[g(X)]")
        
        print("\n  Therefore: g(E[X]) ≤ E[g(X)] □")
        
        print("\nApplications:")
        print("  1. g(x) = x² (convex): E[X]² ≤ E[X²]")
        print("     → Var(X) = E[X²] - E[X]² ≥ 0")
        
        print("\n  2. g(x) = -log(x) (convex): -log(E[X]) ≤ E[-log(X)]")
        print("     → log(E[X]) ≥ E[log(X)]")
        print("     → E[X] ≥ exp(E[log(X)]) (arithmetic ≥ geometric mean)")
        
        print("\n  3. g(x) = e^x (convex): e^{E[X]} ≤ E[e^X]")
        
        # Verify
        np.random.seed(42)
        X = np.random.exponential(1, 100000)
        
        print("\n--- Verification ---")
        print(f"  g(x) = x²: E[X]² = {X.mean()**2:.4f}, E[X²] = {(X**2).mean():.4f}")
        print(f"           {X.mean()**2:.4f} ≤ {(X**2).mean():.4f} ✓")
        
        print(f"\n  g(x) = log: E[log(X)] = {np.log(X).mean():.4f}, log(E[X]) = {np.log(X.mean()):.4f}")
        print(f"             {np.log(X).mean():.4f} ≤ {np.log(X.mean()):.4f} ✓")
    
    def exercise_5_unbiased_variance(self):
        """
        Exercise 5: Unbiased Variance Estimator
        
        Show that S² = Σ(Xᵢ - X̄)²/(n-1) is unbiased for σ².
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Unbiased Variance Estimator")
        
        print("\nShow: E[S²] = σ² where S² = Σᵢ(Xᵢ - X̄)²/(n-1)")
        
        print("\nProof:")
        print("  First, expand Σᵢ(Xᵢ - X̄)²:")
        print("  Σᵢ(Xᵢ - X̄)² = Σᵢ(Xᵢ - μ + μ - X̄)²")
        print("               = Σᵢ[(Xᵢ - μ) - (X̄ - μ)]²")
        print("               = Σᵢ(Xᵢ - μ)² - 2(X̄-μ)Σᵢ(Xᵢ-μ) + n(X̄-μ)²")
        print("               = Σᵢ(Xᵢ - μ)² - 2n(X̄-μ)² + n(X̄-μ)²")
        print("               = Σᵢ(Xᵢ - μ)² - n(X̄-μ)²")
        
        print("\n  Taking expectation:")
        print("  E[Σᵢ(Xᵢ - X̄)²] = E[Σᵢ(Xᵢ - μ)²] - nE[(X̄-μ)²]")
        print("                  = nσ² - n·Var(X̄)")
        print("                  = nσ² - n·(σ²/n)")
        print("                  = nσ² - σ²")
        print("                  = (n-1)σ²")
        
        print("\n  Therefore:")
        print("  E[Σᵢ(Xᵢ - X̄)²/(n-1)] = (n-1)σ²/(n-1) = σ² □")
        
        print("\nWhy divide by n-1?")
        print("  X̄ is closer to data points than μ is.")
        print("  Σ(Xᵢ - X̄)² underestimates Σ(Xᵢ - μ)²")
        print("  The n-1 corrects for this 'degrees of freedom' loss.")
        
        # Verify
        np.random.seed(42)
        mu, sigma = 5, 2
        n_samples = 10000
        sample_size = 20
        
        biased = []
        unbiased = []
        for _ in range(n_samples):
            sample = np.random.normal(mu, sigma, sample_size)
            biased.append(sample.var())        # Divide by n
            unbiased.append(sample.var(ddof=1))  # Divide by n-1
        
        print(f"\n--- Verification: True σ² = {sigma**2} ---")
        print(f"  E[S²_n] (biased) = {np.mean(biased):.4f}")
        print(f"  E[S²_{'{n-1}'}] (unbiased) = {np.mean(unbiased):.4f}")
    
    def exercise_6_total_variance(self):
        """
        Exercise 6: Law of Total Variance
        
        Prove: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Law of Total Variance")
        
        print("\nProve: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])")
        
        print("\nProof:")
        print("  Var(Y|X) = E[Y²|X] - (E[Y|X])²")
        
        print("\n  E[Var(Y|X)] = E[E[Y²|X] - (E[Y|X])²]")
        print("              = E[E[Y²|X]] - E[(E[Y|X])²]")
        print("              = E[Y²] - E[(E[Y|X])²]")
        
        print("\n  Var(E[Y|X]) = E[(E[Y|X])²] - (E[E[Y|X]])²")
        print("              = E[(E[Y|X])²] - (E[Y])²")
        
        print("\n  Adding:")
        print("  E[Var(Y|X)] + Var(E[Y|X])")
        print("    = E[Y²] - E[(E[Y|X])²] + E[(E[Y|X])²] - E[Y]²")
        print("    = E[Y²] - E[Y]²")
        print("    = Var(Y)  □")
        
        print("\nInterpretation:")
        print("  • E[Var(Y|X)]: 'Within-group' variance")
        print("  • Var(E[Y|X]): 'Between-group' variance")
        print("  • Total = Within + Between")
        
        # Verify
        np.random.seed(42)
        n = 100000
        
        X = np.random.randint(0, 3, n)
        mu_Y = [0, 5, 10]
        sigma_Y = [1, 2, 3]
        
        Y = np.array([np.random.normal(mu_Y[x], sigma_Y[x]) for x in X])
        
        # Compute components
        E_Var_Y_X = sum((sigma_Y[k]**2) * (X == k).mean() for k in range(3))
        
        E_Y_X = np.array([mu_Y[x] for x in X])
        Var_E_Y_X = E_Y_X.var()
        
        print(f"\n--- Verification ---")
        print(f"  Var(Y) = {Y.var():.4f}")
        print(f"  E[Var(Y|X)] = {E_Var_Y_X:.4f}")
        print(f"  Var(E[Y|X]) = {Var_E_Y_X:.4f}")
        print(f"  Sum = {E_Var_Y_X + Var_E_Y_X:.4f}")
    
    def exercise_7_bias_variance(self):
        """
        Exercise 7: Bias-Variance Decomposition
        
        Derive: E[(Y - f̂(X))²] = Var(f̂) + Bias²(f̂) + σ²
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Bias-Variance Decomposition")
        
        print("\nSetup: Y = f(X) + ε where E[ε] = 0, Var(ε) = σ²")
        print("       f̂(X) is our predictor")
        
        print("\nDerive: E[(Y - f̂(X))²] = Var(f̂) + Bias²(f̂) + σ²")
        
        print("\nProof:")
        print("  E[(Y - f̂)²] = E[(Y - f + f - E[f̂] + E[f̂] - f̂)²]")
        print("              = E[((Y-f) + (f-E[f̂]) + (E[f̂]-f̂))²]")
        
        print("\n  Expanding (cross terms vanish due to independence):")
        print("  = E[(Y-f)²] + E[(f-E[f̂])²] + E[(E[f̂]-f̂)²]")
        print("  = E[ε²] + (f-E[f̂])² + E[(f̂-E[f̂])²]")
        print("  = σ² + Bias(f̂)² + Var(f̂)")
        
        print("\nComponents:")
        print("  • σ²: Irreducible error (noise)")
        print("  • Bias²: Error from wrong model assumptions")
        print("  • Var: Error from sensitivity to training data")
        
        print("\nTradeoff:")
        print("  • Simple models: high bias, low variance")
        print("  • Complex models: low bias, high variance")
        print("  • Optimal: minimize total = bias² + variance")
        
        # Illustration
        print("\n--- Illustration ---")
        np.random.seed(42)
        
        # True function
        f_true = lambda x: np.sin(x)
        
        # Generate data
        n_datasets = 1000
        n_points = 20
        sigma = 0.3
        
        x_test = 1.5
        
        # Simple model (high bias)
        simple_preds = []
        # Complex model (high variance)
        complex_preds = []
        
        for _ in range(n_datasets):
            x = np.random.uniform(0, np.pi, n_points)
            y = f_true(x) + np.random.randn(n_points) * sigma
            
            # Simple: constant (mean)
            simple_preds.append(y.mean())
            
            # Complex: polynomial fit
            coeffs = np.polyfit(x, y, 10)
            complex_preds.append(np.polyval(coeffs, x_test))
        
        simple_preds = np.array(simple_preds)
        complex_preds = np.array(complex_preds)
        y_true = f_true(x_test)
        
        print(f"  At x = {x_test}, true f(x) = {y_true:.4f}")
        print(f"\n  Simple model (constant):")
        print(f"    Bias = E[f̂] - f = {simple_preds.mean() - y_true:.4f}")
        print(f"    Var = {simple_preds.var():.4f}")
        
        print(f"\n  Complex model (degree-10 poly):")
        print(f"    Bias = E[f̂] - f = {complex_preds.mean() - y_true:.4f}")
        print(f"    Var = {complex_preds.var():.4f}")
    
    def exercise_8_portfolio(self):
        """
        Exercise 8: Portfolio Variance
        
        Find optimal weights to minimize Var(portfolio).
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Portfolio Variance")
        
        print("\nPortfolio: P = w₁R₁ + w₂R₂ where w₁ + w₂ = 1")
        
        print("\nVar(P) = w₁²Var(R₁) + w₂²Var(R₂) + 2w₁w₂Cov(R₁,R₂)")
        print("       = w₁²σ₁² + (1-w₁)²σ₂² + 2w₁(1-w₁)ρσ₁σ₂")
        
        print("\nMinimize by taking derivative:")
        print("  d/dw₁ = 2w₁σ₁² - 2(1-w₁)σ₂² + 2(1-2w₁)ρσ₁σ₂ = 0")
        
        print("\nSolving:")
        print("  w₁* = (σ₂² - ρσ₁σ₂) / (σ₁² + σ₂² - 2ρσ₁σ₂)")
        
        # Example
        sigma1, sigma2 = 0.2, 0.3
        rho = 0.5
        
        w1_opt = (sigma2**2 - rho*sigma1*sigma2) / (sigma1**2 + sigma2**2 - 2*rho*sigma1*sigma2)
        w2_opt = 1 - w1_opt
        
        var_opt = w1_opt**2 * sigma1**2 + w2_opt**2 * sigma2**2 + 2*w1_opt*w2_opt*rho*sigma1*sigma2
        
        print(f"\n--- Example: σ₁={sigma1}, σ₂={sigma2}, ρ={rho} ---")
        print(f"  Optimal weights: w₁* = {w1_opt:.4f}, w₂* = {w2_opt:.4f}")
        print(f"  Portfolio std: σ_P = {np.sqrt(var_opt):.4f}")
        
        print(f"\n  Compare to individual:")
        print(f"    σ₁ = {sigma1}, σ₂ = {sigma2}")
        print(f"    Equal weight (0.5, 0.5): σ_P = {np.sqrt(0.25*sigma1**2 + 0.25*sigma2**2 + 0.5*rho*sigma1*sigma2):.4f}")
        
        print("\nKey insight: Diversification reduces risk when ρ < 1")
    
    def exercise_9_chebyshev(self):
        """
        Exercise 9: Apply Chebyshev's Inequality
        
        Find bounds on P(|X - μ| ≥ kσ).
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Chebyshev's Inequality")
        
        print("\nChebyshev: P(|X - μ| ≥ kσ) ≤ 1/k²")
        
        print("\nProof:")
        print("  By Markov's inequality for Y = (X-μ)²:")
        print("  P((X-μ)² ≥ (kσ)²) ≤ E[(X-μ)²]/(kσ)²")
        print("                     = σ²/(k²σ²)")
        print("                     = 1/k²  □")
        
        print("\nBounds:")
        print("  k=2: P(|X-μ| ≥ 2σ) ≤ 0.25   → at least 75% within 2σ")
        print("  k=3: P(|X-μ| ≥ 3σ) ≤ 0.111  → at least 89% within 3σ")
        print("  k=4: P(|X-μ| ≥ 4σ) ≤ 0.0625 → at least 94% within 4σ")
        
        # Compare to Normal
        print("\n--- Comparison with Normal ---")
        print("  For Normal, actual probabilities:")
        for k in [2, 3, 4]:
            chebyshev = 1/k**2
            normal = 2 * (1 - stats.norm.cdf(k))
            print(f"    k={k}: Chebyshev ≤ {chebyshev:.4f}, Normal actual = {normal:.6f}")
        
        print("\n  Chebyshev is distribution-free but loose.")
        print("  Normal has much lighter tails than worst case.")
    
    def exercise_10_mse_decomposition(self):
        """
        Exercise 10: MSE Decomposition
        
        Show MSE(θ̂) = Var(θ̂) + Bias²(θ̂)
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: MSE Decomposition")
        
        print("\nMSE(θ̂) = E[(θ̂ - θ)²]")
        
        print("\nProof:")
        print("  E[(θ̂ - θ)²] = E[(θ̂ - E[θ̂] + E[θ̂] - θ)²]")
        print("               = E[((θ̂ - E[θ̂]) + (E[θ̂] - θ))²]")
        print("               = E[(θ̂ - E[θ̂])²] + 2E[(θ̂-E[θ̂])(E[θ̂]-θ)] + (E[θ̂]-θ)²")
        
        print("\n  The cross term:")
        print("  E[(θ̂ - E[θ̂])(E[θ̂] - θ)] = (E[θ̂] - θ)·E[θ̂ - E[θ̂]]")
        print("                             = (E[θ̂] - θ)·0 = 0")
        
        print("\n  Therefore:")
        print("  MSE = E[(θ̂ - E[θ̂])²] + (E[θ̂] - θ)²")
        print("      = Var(θ̂) + Bias(θ̂)²  □")
        
        print("\nImplications:")
        print("  • Unbiased (Bias=0): MSE = Var")
        print("  • Biased estimator can have lower MSE if variance is reduced enough")
        
        # Example: estimating variance
        print("\n--- Example: Variance Estimation ---")
        np.random.seed(42)
        
        true_var = 4
        n_samples = 10000
        sample_size = 10
        
        # Unbiased (n-1)
        unbiased_est = []
        # Biased (n)
        biased_est = []
        
        for _ in range(n_samples):
            sample = np.random.normal(0, 2, sample_size)
            unbiased_est.append(sample.var(ddof=1))
            biased_est.append(sample.var())
        
        unbiased_est = np.array(unbiased_est)
        biased_est = np.array(biased_est)
        
        print(f"  Estimating σ² = {true_var}")
        
        bias_unbiased = unbiased_est.mean() - true_var
        var_unbiased = unbiased_est.var()
        mse_unbiased = bias_unbiased**2 + var_unbiased
        
        bias_biased = biased_est.mean() - true_var
        var_biased = biased_est.var()
        mse_biased = bias_biased**2 + var_biased
        
        print(f"\n  Unbiased (n-1):")
        print(f"    Bias = {bias_unbiased:.4f}")
        print(f"    Var = {var_unbiased:.4f}")
        print(f"    MSE = {mse_unbiased:.4f}")
        
        print(f"\n  Biased (n):")
        print(f"    Bias = {bias_biased:.4f}")
        print(f"    Var = {var_biased:.4f}")
        print(f"    MSE = {mse_biased:.4f}")
        
        print(f"\n  The biased estimator has slightly lower MSE!")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = ExpectationExercises()
    
    print("EXPECTATION AND VARIANCE EXERCISES")
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
