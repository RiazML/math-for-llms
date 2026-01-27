"""
KL Divergence - Exercises
=========================
Practice problems for KL divergence.
"""

import numpy as np
from scipy import stats


class KLDivergenceExercises:
    """Exercises for KL divergence."""
    
    def exercise_1_basic_computation(self):
        """
        Exercise 1: Compute KL Divergence
        
        Calculate KL divergence for discrete distributions.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Basic KL Computation")
        print("=" * 60)
        
        def kl_divergence(p, q):
            """D_KL(P || Q)"""
            p, q = np.array(p), np.array(q)
            mask = p > 0
            return np.sum(p[mask] * np.log(p[mask] / q[mask]))
        
        # a) Simple example
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        
        print("a) P = [0.5, 0.5], Q = [0.25, 0.75]")
        kl = kl_divergence(p, q)
        print(f"   D_KL(P || Q) = 0.5 log(0.5/0.25) + 0.5 log(0.5/0.75)")
        print(f"               = 0.5 log(2) + 0.5 log(2/3)")
        print(f"               = {kl:.4f} nats")
        
        # b) Reverse
        kl_reverse = kl_divergence(q, p)
        print(f"\nb) D_KL(Q || P) = {kl_reverse:.4f} nats")
        print(f"   Note: D_KL(P||Q) ≠ D_KL(Q||P) (asymmetric)")
        
        # c) Multiple outcomes
        p3 = np.array([0.7, 0.2, 0.1])
        q3 = np.array([0.33, 0.33, 0.34])
        
        kl3 = kl_divergence(p3, q3)
        print(f"\nc) P = {p3}, Q ≈ uniform")
        print(f"   D_KL(P || Q) = {kl3:.4f} nats")
        
        # d) P = Q
        kl_same = kl_divergence(p, p)
        print(f"\nd) When P = Q: D_KL(P || P) = {kl_same:.4f}")
    
    def exercise_2_bernoulli_kl(self):
        """
        Exercise 2: Bernoulli KL Divergence
        
        Derive and compute KL for Bernoulli distributions.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Bernoulli KL Divergence")
        print("=" * 60)
        
        def bernoulli_kl(p, q):
            """KL(Bern(p) || Bern(q))"""
            if p == 0:
                return np.log(1/(1-q)) if q < 1 else float('inf')
            if p == 1:
                return np.log(1/q) if q > 0 else float('inf')
            return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))
        
        print("Derivation:")
        print("D_KL(Bern(p) || Bern(q))")
        print("= Σ_x P(x) log(P(x)/Q(x))")
        print("= p log(p/q) + (1-p) log((1-p)/(1-q))")
        
        print("\nNumerical examples:")
        print(f"{'p':>6} {'q':>6} {'D_KL':>12}")
        print("-" * 28)
        
        test_cases = [
            (0.5, 0.5),
            (0.7, 0.5),
            (0.9, 0.5),
            (0.9, 0.1),
            (0.5, 0.9),
        ]
        
        for p, q in test_cases:
            kl = bernoulli_kl(p, q)
            print(f"{p:>6.1f} {q:>6.1f} {kl:>12.4f}")
        
        print("\nKey insight: KL large when p ≈ 1 but q ≈ 0 (or vice versa)")
    
    def exercise_3_gaussian_kl(self):
        """
        Exercise 3: Gaussian KL Divergence
        
        Derive the closed-form KL for Gaussians.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Gaussian KL Divergence")
        print("=" * 60)
        
        print("Derivation for N(μ₁,σ₁²) || N(μ₂,σ₂²):")
        print("\nD_KL = ∫ p(x) log(p(x)/q(x)) dx")
        print("     = ∫ p(x) [log p(x) - log q(x)] dx")
        print("     = -H(p) - ∫ p(x) log q(x) dx")
        print("\nFor Gaussian:")
        print("log q(x) = -½log(2πσ₂²) - (x-μ₂)²/(2σ₂²)")
        print("\n∫ p(x) log q(x) dx = -½log(2πσ₂²) - E_p[(X-μ₂)²]/(2σ₂²)")
        print("                    = -½log(2πσ₂²) - [σ₁² + (μ₁-μ₂)²]/(2σ₂²)")
        print("\nH(p) = ½log(2πeσ₁²)")
        print("\nCombining:")
        print("D_KL = log(σ₂/σ₁) + [σ₁² + (μ₁-μ₂)²]/(2σ₂²) - ½")
        
        def gaussian_kl(mu1, s1, mu2, s2):
            return np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5
        
        print("\nNumerical verification:")
        print(f"{'μ₁':>6} {'σ₁':>6} {'μ₂':>6} {'σ₂':>6} {'D_KL':>12}")
        print("-" * 42)
        
        cases = [
            (0, 1, 0, 1),    # Same
            (0, 1, 1, 1),    # Different mean
            (0, 1, 0, 2),    # Different variance
            (1, 2, 0, 1),    # Both different
        ]
        
        for mu1, s1, mu2, s2 in cases:
            kl = gaussian_kl(mu1, s1, mu2, s2)
            print(f"{mu1:>6} {s1:>6} {mu2:>6} {s2:>6} {kl:>12.4f}")
    
    def exercise_4_properties(self):
        """
        Exercise 4: KL Properties
        
        Verify key properties of KL divergence.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: KL Properties")
        print("=" * 60)
        
        def kl(p, q):
            p, q = np.array(p), np.array(q)
            mask = p > 0
            return np.sum(p[mask] * np.log(p[mask] / q[mask]))
        
        p = np.array([0.3, 0.5, 0.2])
        q = np.array([0.2, 0.3, 0.5])
        
        print("a) Non-negativity: D_KL(P || Q) ≥ 0")
        print(f"   P = {p}, Q = {q}")
        print(f"   D_KL(P || Q) = {kl(p, q):.4f} ≥ 0 ✓")
        
        print("\nb) Zero iff P = Q:")
        print(f"   D_KL(P || P) = {kl(p, p):.6f} ✓")
        
        print("\nc) Asymmetry: D_KL(P || Q) ≠ D_KL(Q || P)")
        print(f"   D_KL(P || Q) = {kl(p, q):.4f}")
        print(f"   D_KL(Q || P) = {kl(q, p):.4f}")
        
        print("\nd) Not a metric (triangle inequality fails):")
        r = np.array([0.33, 0.34, 0.33])
        print(f"   R = {r}")
        d_pr = kl(p, r)
        d_rq = kl(r, q)
        d_pq = kl(p, q)
        print(f"   D_KL(P || R) + D_KL(R || Q) = {d_pr + d_rq:.4f}")
        print(f"   D_KL(P || Q) = {d_pq:.4f}")
        print(f"   Triangle inequality {'holds' if d_pq <= d_pr + d_rq else 'FAILS'}")
    
    def exercise_5_cross_entropy(self):
        """
        Exercise 5: Cross-Entropy Relationship
        
        Verify H(P,Q) = H(P) + D_KL(P || Q).
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Cross-Entropy Relationship")
        print("=" * 60)
        
        def entropy(p):
            p = p[p > 0]
            return -np.sum(p * np.log(p))
        
        def cross_entropy(p, q):
            mask = p > 0
            return -np.sum(p[mask] * np.log(q[mask]))
        
        def kl(p, q):
            mask = p > 0
            return np.sum(p[mask] * np.log(p[mask] / q[mask]))
        
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.33, 0.33, 0.34])
        
        H_p = entropy(p)
        H_pq = cross_entropy(p, q)
        D_kl = kl(p, q)
        
        print(f"P = {p}")
        print(f"Q = {q}")
        print(f"\nH(P) = {H_p:.4f} nats")
        print(f"H(P, Q) = {H_pq:.4f} nats")
        print(f"D_KL(P || Q) = {D_kl:.4f} nats")
        
        print(f"\nVerification: H(P) + D_KL = {H_p + D_kl:.4f}")
        print(f"              H(P, Q) = {H_pq:.4f}")
        
        print("\nImplication for ML:")
        print("  min_Q H(P, Q) = min_Q [H(P) + D_KL(P || Q)]")
        print("                = H(P) + min_Q D_KL(P || Q)")
        print("  Since H(P) is constant, minimizing cross-entropy")
        print("  is equivalent to minimizing KL divergence!")
    
    def exercise_6_forward_reverse(self):
        """
        Exercise 6: Forward vs Reverse KL
        
        Understand the difference in behavior.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Forward vs Reverse KL")
        print("=" * 60)
        
        # Bimodal target
        print("Target P: mixture of two modes")
        print("  Mode 1: probability mass at x=1")
        print("  Mode 2: probability mass at x=4")
        
        # Discrete approximation
        x = np.arange(0, 6)
        p = np.zeros(6)
        p[1] = 0.5  # Mode at x=1
        p[4] = 0.5  # Mode at x=4
        
        def kl(p, q):
            mask = (p > 0) & (q > 0)
            return np.sum(p[mask] * np.log(p[mask] / q[mask]))
        
        print(f"\nP = {p}")
        
        # Different approximations Q
        q_covers_both = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])  # Broad
        q_mode1 = np.array([0.1, 0.8, 0.1, 0.0, 0.0, 0.0])        # First mode
        q_mode2 = np.array([0.0, 0.0, 0.0, 0.1, 0.8, 0.1])        # Second mode
        q_middle = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0])       # Between modes
        
        qs = {
            'Covers both': q_covers_both,
            'Mode 1 only': q_mode1,
            'Mode 2 only': q_mode2,
            'Between modes': q_middle
        }
        
        print(f"\n{'Q':>20} {'Forward D_KL(P||Q)':>20} {'Reverse D_KL(Q||P)':>20}")
        print("-" * 65)
        
        for name, q in qs.items():
            # Add small epsilon for numerical stability
            q_safe = q + 1e-10
            q_safe = q_safe / q_safe.sum()
            
            forward = kl(p, q_safe)
            reverse = kl(q_safe, p + 1e-10)
            
            print(f"{name:>20} {forward:>20.4f} {reverse:>20.4f}")
        
        print("\nForward KL: penalizes Q=0 where P>0 (zero-avoiding)")
        print("  → Prefers broad Q covering all modes")
        print("\nReverse KL: penalizes P=0 where Q>0 (zero-forcing)")
        print("  → Prefers narrow Q focusing on one mode")
    
    def exercise_7_vae_kl(self):
        """
        Exercise 7: VAE KL Term
        
        Derive and compute KL for VAE.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: VAE KL Term")
        print("=" * 60)
        
        print("VAE encoder: q(z|x) = N(μ, σ²I)")
        print("Prior: p(z) = N(0, I)")
        print("\nKL divergence (per dimension):")
        print("D_KL = log(1/σ) + (σ² + μ²)/2 - 1/2")
        print("     = -log(σ) + (σ² + μ²)/2 - 1/2")
        print("     = -½[1 + log(σ²) - μ² - σ²]")
        print("\nIn code (using log_var = log(σ²)):")
        print("kl = -0.5 * (1 + log_var - mu^2 - exp(log_var))")
        
        def vae_kl(mu, log_var):
            """KL divergence from N(mu, exp(log_var)) to N(0, 1)."""
            return -0.5 * (1 + log_var - mu**2 - np.exp(log_var))
        
        print(f"\n{'μ':>8} {'log(σ²)':>10} {'σ':>8} {'KL':>10}")
        print("-" * 42)
        
        cases = [
            (0.0, 0.0),   # σ=1, μ=0 → same as prior
            (1.0, 0.0),   # σ=1, μ≠0
            (0.0, 1.0),   # σ>1, μ=0
            (0.0, -1.0),  # σ<1, μ=0
            (2.0, 0.5),   # Both different
        ]
        
        for mu, log_var in cases:
            sigma = np.sqrt(np.exp(log_var))
            kl = vae_kl(mu, log_var)
            print(f"{mu:>8.1f} {log_var:>10.1f} {sigma:>8.2f} {kl:>10.4f}")
        
        print("\nMinimum KL = 0 when μ=0, σ²=1 (matches prior)")
    
    def exercise_8_jensen_shannon(self):
        """
        Exercise 8: Jensen-Shannon Divergence
        
        Compute and analyze JS divergence.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Jensen-Shannon Divergence")
        print("=" * 60)
        
        def kl(p, q):
            mask = (p > 0) & (q > 0)
            return np.sum(p[mask] * np.log(p[mask] / q[mask]))
        
        def js(p, q):
            m = 0.5 * (p + q)
            return 0.5 * kl(p, m) + 0.5 * kl(q, m)
        
        print("Jensen-Shannon divergence:")
        print("D_JS(P || Q) = ½ D_KL(P || M) + ½ D_KL(Q || M)")
        print("where M = ½(P + Q)")
        
        p = np.array([0.9, 0.1, 0.0])
        q = np.array([0.0, 0.1, 0.9])
        
        print(f"\nP = {p}")
        print(f"Q = {q}")
        
        # Handle disjoint support
        p_safe = p + 1e-10
        q_safe = q + 1e-10
        p_safe /= p_safe.sum()
        q_safe /= q_safe.sum()
        
        js_div = js(p_safe, q_safe)
        
        print(f"\nD_JS(P || Q) = {js_div:.4f}")
        print(f"D_JS(Q || P) = {js(q_safe, p_safe):.4f} (symmetric!)")
        
        print(f"\nProperties:")
        print(f"  1. Symmetric: D_JS(P||Q) = D_JS(Q||P)")
        print(f"  2. Bounded: 0 ≤ D_JS ≤ log(2) ≈ {np.log(2):.4f}")
        print(f"  3. √D_JS is a proper metric")
        print(f"  4. Works with disjoint support (unlike KL)")
    
    def exercise_9_chain_rule(self):
        """
        Exercise 9: KL Chain Rule
        
        Apply chain rule for joint distributions.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: KL Chain Rule")
        print("=" * 60)
        
        print("Chain rule:")
        print("D_KL(P(X,Y) || Q(X,Y)) = D_KL(P(X) || Q(X))")
        print("                       + E_P(X)[D_KL(P(Y|X) || Q(Y|X))]")
        
        # Joint distributions
        P_XY = np.array([
            [0.4, 0.1],
            [0.1, 0.4]
        ])
        
        Q_XY = np.array([
            [0.25, 0.25],
            [0.25, 0.25]
        ])
        
        print(f"\nP(X,Y):\n{P_XY}")
        print(f"\nQ(X,Y):\n{Q_XY}")
        
        # Direct computation
        def kl(p, q):
            p_flat = p.flatten()
            q_flat = q.flatten()
            mask = p_flat > 0
            return np.sum(p_flat[mask] * np.log(p_flat[mask] / q_flat[mask]))
        
        kl_joint = kl(P_XY, Q_XY)
        
        # Chain rule computation
        P_X = P_XY.sum(axis=1)
        Q_X = Q_XY.sum(axis=1)
        
        kl_marginal = kl(P_X, Q_X)
        
        # Conditional KL
        kl_cond = 0
        for i in range(2):
            P_Y_given_X = P_XY[i, :] / P_X[i]
            Q_Y_given_X = Q_XY[i, :] / Q_X[i]
            kl_cond += P_X[i] * kl(P_Y_given_X, Q_Y_given_X)
        
        print(f"\nDirect: D_KL(P(X,Y) || Q(X,Y)) = {kl_joint:.4f}")
        print(f"\nChain rule:")
        print(f"  D_KL(P(X) || Q(X)) = {kl_marginal:.4f}")
        print(f"  E_P[D_KL(P(Y|X) || Q(Y|X))] = {kl_cond:.4f}")
        print(f"  Sum = {kl_marginal + kl_cond:.4f}")
    
    def exercise_10_kl_estimation(self):
        """
        Exercise 10: Estimating KL from Samples
        
        Estimate KL divergence using Monte Carlo.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: KL Estimation")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Two Gaussians
        mu_p, sigma_p = 0, 1
        mu_q, sigma_q = 0.5, 1.5
        
        # True KL
        true_kl = (np.log(sigma_q / sigma_p) + 
                   (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5)
        
        print(f"P = N({mu_p}, {sigma_p}²)")
        print(f"Q = N({mu_q}, {sigma_q}²)")
        print(f"True D_KL(P || Q) = {true_kl:.4f}")
        
        print("\nMonte Carlo estimation:")
        print("D_KL ≈ (1/n) Σ log(p(xᵢ)/q(xᵢ)) where xᵢ ~ P")
        
        print(f"\n{'n':>8} {'Estimate':>12} {'Std Err':>12} {'Rel Error':>12}")
        print("-" * 50)
        
        for n in [100, 500, 1000, 5000]:
            estimates = []
            
            for _ in range(50):
                samples = np.random.normal(mu_p, sigma_p, n)
                
                log_p = stats.norm.logpdf(samples, mu_p, sigma_p)
                log_q = stats.norm.logpdf(samples, mu_q, sigma_q)
                
                kl_est = np.mean(log_p - log_q)
                estimates.append(kl_est)
            
            mean_est = np.mean(estimates)
            std_err = np.std(estimates)
            rel_err = abs(mean_est - true_kl) / true_kl
            
            print(f"{n:>8} {mean_est:>12.4f} {std_err:>12.4f} {rel_err:>12.2%}")
        
        print("\nEstimate improves with more samples (√n convergence)")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = KLDivergenceExercises()
    
    print("KL DIVERGENCE EXERCISES")
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
