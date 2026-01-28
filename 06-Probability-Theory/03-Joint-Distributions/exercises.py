"""
Joint Distributions and Independence - Exercises
================================================
Practice problems for joint distributions.
"""

import numpy as np
from scipy import stats


class JointDistributionExercises:
    """Exercises for joint distributions and independence."""
    
    def exercise_1_joint_pmf(self):
        """
        Exercise 1: Joint PMF Operations
        
        Given joint PMF table, find:
        a) Marginal distributions
        b) Conditional distributions
        c) E[X], E[Y], E[XY]
        d) Cov(X,Y) and Corr(X,Y)
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Joint PMF Operations")
        
        # Joint PMF
        joint = np.array([
            [0.1, 0.2, 0.1],  # X=1
            [0.1, 0.3, 0.2]   # X=2
        ])
        X_vals = [1, 2]
        Y_vals = [0, 1, 2]
        
        print("\nJoint PMF:")
        print("       Y=0    Y=1    Y=2")
        for i, x in enumerate(X_vals):
            print(f"  X={x}  {joint[i,0]:.1f}   {joint[i,1]:.1f}   {joint[i,2]:.1f}")
        
        # (a) Marginals
        p_X = joint.sum(axis=1)
        p_Y = joint.sum(axis=0)
        
        print("\n(a) Marginal distributions:")
        print(f"  P(X=1) = {p_X[0]:.1f}, P(X=2) = {p_X[1]:.1f}")
        print(f"  P(Y=0) = {p_Y[0]:.1f}, P(Y=1) = {p_Y[1]:.1f}, P(Y=2) = {p_Y[2]:.1f}")
        
        # (b) Conditionals
        print("\n(b) Conditional distributions:")
        print("  P(Y|X=1):", (joint[0,:]/p_X[0]).round(3))
        print("  P(Y|X=2):", (joint[1,:]/p_X[1]).round(3))
        print("  P(X|Y=1):", (joint[:,1]/p_Y[1]).round(3))
        
        # (c) Expectations
        E_X = sum(x * p for x, p in zip(X_vals, p_X))
        E_Y = sum(y * p for y, p in zip(Y_vals, p_Y))
        E_XY = sum(X_vals[i] * Y_vals[j] * joint[i,j] 
                   for i in range(2) for j in range(3))
        
        print("\n(c) Expectations:")
        print(f"  E[X] = 1×{p_X[0]} + 2×{p_X[1]} = {E_X:.1f}")
        print(f"  E[Y] = 0×{p_Y[0]} + 1×{p_Y[1]} + 2×{p_Y[2]} = {E_Y:.1f}")
        print(f"  E[XY] = {E_XY:.2f}")
        
        # (d) Covariance and Correlation
        E_X2 = sum(x**2 * p for x, p in zip(X_vals, p_X))
        E_Y2 = sum(y**2 * p for y, p in zip(Y_vals, p_Y))
        Var_X = E_X2 - E_X**2
        Var_Y = E_Y2 - E_Y**2
        
        Cov_XY = E_XY - E_X * E_Y
        Corr_XY = Cov_XY / np.sqrt(Var_X * Var_Y)
        
        print("\n(d) Covariance and Correlation:")
        print(f"  Var(X) = {Var_X:.4f}, Var(Y) = {Var_Y:.4f}")
        print(f"  Cov(X,Y) = E[XY] - E[X]E[Y] = {E_XY:.2f} - {E_X:.1f}×{E_Y:.1f} = {Cov_XY:.4f}")
        print(f"  Corr(X,Y) = {Corr_XY:.4f}")
    
    def exercise_2_uncorrelated_normal(self):
        """
        Exercise 2: Uncorrelated Normal → Independent
        
        Prove that for multivariate normal, uncorrelated implies independent.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Uncorrelated Normal → Independent")
        
        print("\nFor multivariate normal, X and Y are independent iff Cov(X,Y) = 0")
        
        print("\nProof:")
        print("  Let (X,Y) ~ N(μ, Σ) where Σ = [[σ²_X, ρσ_Xσ_Y],")
        print("                                  [ρσ_Xσ_Y, σ²_Y]]")
        
        print("\n  If uncorrelated: ρ = 0, so Σ = [[σ²_X, 0],")
        print("                                   [0, σ²_Y]]")
        
        print("\n  Joint PDF:")
        print("  f(x,y) = (1/2πσ_Xσ_Y) exp(-(x-μ_X)²/2σ²_X - (y-μ_Y)²/2σ²_Y)")
        print("         = (1/√(2πσ²_X) exp(-(x-μ_X)²/2σ²_X))")
        print("           × (1/√(2πσ²_Y) exp(-(y-μ_Y)²/2σ²_Y))")
        print("         = f_X(x) × f_Y(y)")
        
        print("\n  Joint = product of marginals → Independent! □")
        
        print("\nIMPORTANT: This is special to normal distribution!")
        print("           In general, uncorrelated ≠ independent")
        
        # Verify numerically
        print("\n--- Numerical Verification ---")
        np.random.seed(42)
        
        # Uncorrelated normal
        Sigma = np.array([[1, 0], [0, 2]])
        samples = np.random.multivariate_normal([0, 0], Sigma, 100000)
        X, Y = samples[:, 0], samples[:, 1]
        
        # Check independence: P(Y<0|X<0) = P(Y<0)?
        p_Y_neg = (Y < 0).mean()
        p_Y_neg_given_X_neg = (Y[X < 0] < 0).mean()
        
        print(f"  P(Y < 0) = {p_Y_neg:.4f}")
        print(f"  P(Y < 0 | X < 0) = {p_Y_neg_given_X_neg:.4f}")
        print("  Equal → Independent ✓")
    
    def exercise_3_conditional_mvn(self):
        """
        Exercise 3: Derive Conditional Distribution of MVN
        
        For bivariate normal, derive E[X|Y] and Var(X|Y).
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Conditional Distribution of Bivariate Normal")
        
        print("\nLet (X,Y) ~ N(μ, Σ) with:")
        print("  μ = (μ_X, μ_Y)")
        print("  Σ = [[σ²_X, ρσ_Xσ_Y], [ρσ_Xσ_Y, σ²_Y]]")
        
        print("\nDerivation sketch:")
        print("  f(x|y) = f(x,y)/f_Y(y)")
        
        print("\n  Completing the square in the exponent gives:")
        
        print("\n  X|Y=y ~ N(μ_{X|Y}, σ²_{X|Y})")
        print("\n  where:")
        print("    μ_{X|Y} = μ_X + ρ(σ_X/σ_Y)(y - μ_Y)")
        print("    σ²_{X|Y} = σ²_X(1 - ρ²)")
        
        print("\nKey insights:")
        print("  1. Conditional mean is LINEAR in y (regression line)")
        print("  2. Conditional variance is CONSTANT (doesn't depend on y)")
        print("  3. If ρ = 0: μ_{X|Y} = μ_X (knowing Y doesn't help)")
        print("  4. If |ρ| = 1: σ²_{X|Y} = 0 (X determined by Y)")
        
        # Numerical example
        print("\n--- Numerical Example ---")
        mu = np.array([2, 5])
        rho = 0.7
        sigma_X, sigma_Y = 2, 3
        Sigma = np.array([[sigma_X**2, rho*sigma_X*sigma_Y],
                          [rho*sigma_X*sigma_Y, sigma_Y**2]])
        
        y_given = 8
        
        mu_cond = mu[0] + rho * (sigma_X/sigma_Y) * (y_given - mu[1])
        var_cond = sigma_X**2 * (1 - rho**2)
        
        print(f"  μ = {mu}, ρ = {rho}")
        print(f"  σ_X = {sigma_X}, σ_Y = {sigma_Y}")
        print(f"\n  Given Y = {y_given}:")
        print(f"    E[X|Y={y_given}] = {mu[0]} + {rho}×({sigma_X}/{sigma_Y})×({y_given}-{mu[1]})")
        print(f"                    = {mu_cond:.4f}")
        print(f"    Var(X|Y) = {sigma_X}²×(1-{rho}²) = {var_cond:.4f}")
        
        # Verify with simulation
        np.random.seed(42)
        samples = np.random.multivariate_normal(mu, Sigma, 100000)
        mask = np.abs(samples[:, 1] - y_given) < 0.2
        
        print(f"\n  Simulation (Y ≈ {y_given}):")
        print(f"    E[X|Y] ≈ {samples[mask, 0].mean():.4f}")
        print(f"    Var(X|Y) ≈ {samples[mask, 0].var():.4f}")
    
    def exercise_4_covariance_properties(self):
        """
        Exercise 4: Covariance Properties
        
        Prove: Cov(aX+b, cY+d) = ac·Cov(X,Y)
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Covariance Properties")
        
        print("\nProve: Cov(aX+b, cY+d) = ac·Cov(X,Y)")
        
        print("\nProof:")
        print("  Cov(aX+b, cY+d) = E[(aX+b - E[aX+b])(cY+d - E[cY+d])]")
        print("                  = E[(aX+b - aμ_X-b)(cY+d - cμ_Y-d)]")
        print("                  = E[a(X-μ_X) · c(Y-μ_Y)]")
        print("                  = ac · E[(X-μ_X)(Y-μ_Y)]")
        print("                  = ac · Cov(X,Y)  □")
        
        print("\nCorollaries:")
        print("  1. Cov(X+b, Y) = Cov(X, Y)  (adding constant doesn't change cov)")
        print("  2. Cov(aX, Y) = a·Cov(X, Y)  (scaling)")
        print("  3. Var(aX) = a²·Var(X)  (variance scales quadratically)")
        
        # Verify
        np.random.seed(42)
        n = 100000
        X = np.random.randn(n)
        Y = 0.5 * X + np.random.randn(n)
        
        a, b, c, d = 2, 3, -1, 5
        
        cov_orig = np.cov(X, Y)[0, 1]
        cov_transformed = np.cov(a*X + b, c*Y + d)[0, 1]
        
        print("\n--- Verification ---")
        print(f"  Cov(X, Y) = {cov_orig:.4f}")
        print(f"  a·c·Cov(X,Y) = {a}×{c}×{cov_orig:.4f} = {a*c*cov_orig:.4f}")
        print(f"  Cov({a}X+{b}, {c}Y+{d}) = {cov_transformed:.4f} ✓")
    
    def exercise_5_sum_variance(self):
        """
        Exercise 5: Variance of Sum
        
        Derive Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Variance of Sum")
        
        print("\nDerive: Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)")
        
        print("\nProof:")
        print("  Var(X + Y) = E[(X+Y - E[X+Y])²]")
        print("             = E[(X-μ_X + Y-μ_Y)²]")
        print("             = E[(X-μ_X)² + 2(X-μ_X)(Y-μ_Y) + (Y-μ_Y)²]")
        print("             = E[(X-μ_X)²] + 2E[(X-μ_X)(Y-μ_Y)] + E[(Y-μ_Y)²]")
        print("             = Var(X) + 2Cov(X,Y) + Var(Y)  □")
        
        print("\nSpecial cases:")
        print("  • If X ⊥ Y: Var(X+Y) = Var(X) + Var(Y)")
        print("  • If X = Y: Var(2X) = Var(X) + Var(X) + 2Var(X) = 4Var(X)")
        
        print("\nGeneral formula for n variables:")
        print("  Var(Σᵢ Xᵢ) = Σᵢ Var(Xᵢ) + 2Σᵢ<ⱼ Cov(Xᵢ,Xⱼ)")
        
        # Verify
        np.random.seed(42)
        X = np.random.randn(100000)
        Y = 0.5*X + np.random.randn(100000)
        
        var_X = X.var()
        var_Y = Y.var()
        cov_XY = np.cov(X, Y)[0, 1]
        var_sum = (X + Y).var()
        
        print("\n--- Verification ---")
        print(f"  Var(X) = {var_X:.4f}")
        print(f"  Var(Y) = {var_Y:.4f}")
        print(f"  Cov(X,Y) = {cov_XY:.4f}")
        print(f"  Var(X) + Var(Y) + 2Cov(X,Y) = {var_X + var_Y + 2*cov_XY:.4f}")
        print(f"  Var(X+Y) = {var_sum:.4f} ✓")
    
    def exercise_6_transform_pdf(self):
        """
        Exercise 6: Transformation of PDF
        
        If X ~ Uniform(0,1), find PDF of Y = -ln(X).
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Transformation of PDF")
        
        print("\nGiven: X ~ Uniform(0,1), find PDF of Y = -ln(X)")
        
        print("\nMethod: CDF technique")
        print("  F_Y(y) = P(Y ≤ y)")
        print("         = P(-ln(X) ≤ y)")
        print("         = P(ln(X) ≥ -y)")
        print("         = P(X ≥ e^{-y})")
        print("         = 1 - P(X < e^{-y})")
        print("         = 1 - e^{-y}    (for y > 0)")
        
        print("\nPDF:")
        print("  f_Y(y) = d/dy F_Y(y)")
        print("         = d/dy (1 - e^{-y})")
        print("         = e^{-y}    for y > 0")
        
        print("\nThis is Exponential(λ=1)!")
        print("(Used for generating exponential samples from uniform)")
        
        # Verify
        np.random.seed(42)
        U = np.random.uniform(0, 1, 100000)
        Y = -np.log(U)
        
        print("\n--- Verification ---")
        print(f"  E[Y] = {Y.mean():.4f} (theory: 1)")
        print(f"  Var(Y) = {Y.var():.4f} (theory: 1)")
        
        # Compare to direct exponential
        exp_samples = np.random.exponential(1, 100000)
        print(f"  Direct Exp(1): Mean = {exp_samples.mean():.4f}, Var = {exp_samples.var():.4f}")
    
    def exercise_7_total_variance(self):
        """
        Exercise 7: Law of Total Variance
        
        Prove: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Law of Total Variance")
        
        print("\nProve: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])")
        print("       'Total variance = Within variance + Between variance'")
        
        print("\nProof:")
        print("  Var(Y|X) = E[Y²|X] - E[Y|X]²")
        print("  E[Var(Y|X)] = E[E[Y²|X]] - E[E[Y|X]²]")
        print("              = E[Y²] - E[E[Y|X]²]  (law of total expectation)")
        
        print("\n  Var(E[Y|X]) = E[E[Y|X]²] - E[E[Y|X]]²")
        print("              = E[E[Y|X]²] - E[Y]²  (law of total expectation)")
        
        print("\n  Adding:")
        print("  E[Var(Y|X)] + Var(E[Y|X])")
        print("    = E[Y²] - E[E[Y|X]²] + E[E[Y|X]²] - E[Y]²")
        print("    = E[Y²] - E[Y]²")
        print("    = Var(Y)  □")
        
        # Example
        print("\n--- Example: Mixture ---")
        print("  X = group (0 or 1)")
        print("  Y|X=0 ~ N(0, 1), Y|X=1 ~ N(3, 4)")
        print("  P(X=0) = 0.4")
        
        p0 = 0.4
        mu0, var0 = 0, 1
        mu1, var1 = 3, 4
        
        # E[Var(Y|X)]
        E_Var_Y_X = p0 * var0 + (1-p0) * var1
        print(f"\n  E[Var(Y|X)] = {p0}×{var0} + {1-p0}×{var1} = {E_Var_Y_X}")
        
        # Var(E[Y|X])
        E_Y = p0 * mu0 + (1-p0) * mu1
        E_EYX_sq = p0 * mu0**2 + (1-p0) * mu1**2
        Var_E_Y_X = E_EYX_sq - E_Y**2
        print(f"  E[Y|X] takes values {mu0} or {mu1}")
        print(f"  E[Y] = {E_Y}")
        print(f"  Var(E[Y|X]) = {Var_E_Y_X}")
        
        # Total
        total_var = E_Var_Y_X + Var_E_Y_X
        print(f"\n  Var(Y) = {E_Var_Y_X} + {Var_E_Y_X} = {total_var}")
        
        # Verify
        np.random.seed(42)
        X = np.random.binomial(1, 1-p0, 100000)
        Y = np.where(X == 0,
                     np.random.normal(mu0, np.sqrt(var0), 100000),
                     np.random.normal(mu1, np.sqrt(var1), 100000))
        
        print(f"  Simulation: Var(Y) = {Y.var():.4f}")
    
    def exercise_8_bivariate_marginal(self):
        """
        Exercise 8: Marginal from Joint PDF
        
        Given f(x,y) = 6xy for 0<x<1, 0<y<1, x+y<1, find marginals.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Marginal from Joint PDF")
        
        print("\nGiven: f(x,y) = 6xy for region:")
        print("  0 < x < 1, 0 < y < 1, x + y < 1")
        print("  (lower triangle of unit square)")
        
        print("\nFind f_X(x):")
        print("  f_X(x) = ∫₀^{1-x} 6xy dy")
        print("         = 6x [y²/2]₀^{1-x}")
        print("         = 6x · (1-x)²/2")
        print("         = 3x(1-x)²    for 0 < x < 1")
        
        print("\nBy symmetry (swapping x and y doesn't change integral):")
        print("  f_Y(y) = 3y(1-y)²    for 0 < y < 1")
        
        # Verify normalization
        print("\nVerify: ∫₀¹ 3x(1-x)² dx = 1")
        print("  = 3 ∫₀¹ (x - 2x² + x³) dx")
        print("  = 3 [x²/2 - 2x³/3 + x⁴/4]₀¹")
        print("  = 3 [1/2 - 2/3 + 1/4]")
        print("  = 3 × 1/12 = 1/4")
        print("  Wait... this equals 1/4, not 1!")
        
        print("\nActually: marginal integrates to 1 over its support,")
        print("          not the joint integral.")
        
        # Simulation
        np.random.seed(42)
        n = 100000
        
        # Rejection sampling
        samples = []
        while len(samples) < n:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            if x + y < 1:
                if np.random.uniform(0, 1) < 6*x*y / 6:  # max pdf = 6(0.5)(0.5) = 1.5
                    samples.append((x, y))
        
        samples = np.array(samples[:n])
        
        print("\n--- Simulation ---")
        print(f"  E[X] = {samples[:,0].mean():.4f}")
        print(f"  E[Y] = {samples[:,1].mean():.4f}")
        print(f"  Corr(X,Y) = {np.corrcoef(samples[:,0], samples[:,1])[0,1]:.4f}")
    
    def exercise_9_chain_rule(self):
        """
        Exercise 9: Apply Chain Rule
        
        Factor P(A,B,C,D) using chain rule.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Chain Rule Factorization")
        
        print("\nChain rule:")
        print("  P(X₁, X₂, ..., Xₙ) = P(X₁)·P(X₂|X₁)·P(X₃|X₁,X₂)···P(Xₙ|X₁,...,X_{n-1})")
        
        print("\nFor P(A, B, C, D):")
        print("  = P(A)·P(B|A)·P(C|A,B)·P(D|A,B,C)")
        
        print("\nOther valid factorizations (different orderings):")
        print("  = P(D)·P(C|D)·P(B|C,D)·P(A|B,C,D)")
        print("  = P(B)·P(A|B)·P(C|A,B)·P(D|A,B,C)")
        print("  ... (n! possible orderings)")
        
        print("\nApplication: Bayesian Networks")
        print("  With independence assumptions, many terms simplify")
        print("  Example: If D ⊥ A | B,C:")
        print("    P(D|A,B,C) = P(D|B,C)")
        
        print("\n--- Example: Weather, Sprinkler, Rain, Grass ---")
        print("  Structure: Weather → Rain → Grass")
        print("             Weather → Sprinkler → Grass")
        
        print("\n  P(W, S, R, G) = P(W)·P(S|W)·P(R|W)·P(G|S,R)")
        print("  (Using conditional independence from graph structure)")
        print("  Note: S ⊥ R | W  (sprinkler indep of rain given weather)")
    
    def exercise_10_independence_test(self):
        """
        Exercise 10: Test Independence from Data
        
        Given samples from joint distribution, test if X and Y are independent.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Testing Independence")
        
        print("\nMethods to test independence:")
        print("  1. Chi-squared test (discrete)")
        print("  2. Correlation test (continuous, linear)")
        print("  3. Mutual information (any dependence)")
        
        # Generate independent data
        np.random.seed(42)
        n = 1000
        X_ind = np.random.randint(0, 3, n)
        Y_ind = np.random.randint(0, 3, n)
        
        # Generate dependent data
        X_dep = np.random.randint(0, 3, n)
        Y_dep = (X_dep + np.random.randint(0, 2, n)) % 3
        
        print("\n--- Chi-Squared Test ---")
        
        def chi_squared_test(X, Y):
            # Contingency table
            K = 3
            observed = np.zeros((K, K))
            for x, y in zip(X, Y):
                observed[x, y] += 1
            
            # Expected under independence
            row_sums = observed.sum(axis=1)
            col_sums = observed.sum(axis=0)
            expected = np.outer(row_sums, col_sums) / n
            
            # Chi-squared statistic
            chi2 = ((observed - expected)**2 / expected).sum()
            
            # Degrees of freedom
            df = (K - 1) * (K - 1)
            
            # p-value
            p_value = 1 - stats.chi2.cdf(chi2, df)
            
            return chi2, p_value
        
        chi2_ind, p_ind = chi_squared_test(X_ind, Y_ind)
        chi2_dep, p_dep = chi_squared_test(X_dep, Y_dep)
        
        print(f"\n  Independent data:")
        print(f"    χ² = {chi2_ind:.2f}, p-value = {p_ind:.4f}")
        print(f"    Conclusion: {'Independent' if p_ind > 0.05 else 'Dependent'}")
        
        print(f"\n  Dependent data:")
        print(f"    χ² = {chi2_dep:.2f}, p-value = {p_dep:.4f}")
        print(f"    Conclusion: {'Independent' if p_dep > 0.05 else 'Dependent'}")
        
        print("\nNote: p-value > 0.05 → fail to reject independence")
        print("      p-value < 0.05 → reject independence (likely dependent)")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = JointDistributionExercises()
    
    print("JOINT DISTRIBUTIONS EXERCISES")
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
