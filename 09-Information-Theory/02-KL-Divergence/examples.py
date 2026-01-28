"""
KL Divergence - Examples
========================
Computing and applying KL divergence.
"""

import numpy as np
from scipy import stats
from scipy.special import kl_div as scipy_kl


def example_basic_kl():
    """Basic KL divergence computation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic KL Divergence")
    print("=" * 60)
    
    def kl_divergence(p, q):
        """Compute KL(P || Q) for discrete distributions."""
        p = np.array(p)
        q = np.array(q)
        # Avoid log(0)
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    # Example distributions
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q1 = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform
    q2 = np.array([0.5, 0.3, 0.15, 0.05])    # Similar to p
    q3 = np.array([0.1, 0.2, 0.3, 0.4])      # Reversed
    
    print(f"P = {p}")
    print(f"\nD_KL(P || Q) in nats:")
    print(f"  Q = Uniform{list(q1)}: {kl_divergence(p, q1):.4f}")
    print(f"  Q = Similar {list(q2)}: {kl_divergence(p, q2):.4f}")
    print(f"  Q = Reversed{list(q3)}: {kl_divergence(p, q3):.4f}")
    
    # Show asymmetry
    print(f"\nAsymmetry demonstration:")
    print(f"  D_KL(P || Q_similar) = {kl_divergence(p, q2):.4f}")
    print(f"  D_KL(Q_similar || P) = {kl_divergence(q2, p):.4f}")


def example_bernoulli_kl():
    """KL divergence between Bernoulli distributions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Bernoulli KL Divergence")
    print("=" * 60)
    
    def bernoulli_kl(p, q):
        """KL divergence between Bernoulli(p) and Bernoulli(q)."""
        if p == 0:
            return -np.log(1 - q) if q < 1 else float('inf')
        if p == 1:
            return -np.log(q) if q > 0 else float('inf')
        
        kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        return kl
    
    print("D_KL(Bernoulli(p) || Bernoulli(q))")
    print("= p log(p/q) + (1-p) log((1-p)/(1-q))")
    
    p = 0.7
    print(f"\nFixed p = {p}:")
    print(f"{'q':>8} {'D_KL':>12}")
    print("-" * 25)
    
    for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
        kl = bernoulli_kl(p, q)
        print(f"{q:>8.1f} {kl:>12.4f}")
    
    print("\nMinimum at q = p (KL = 0)")


def example_gaussian_kl():
    """KL divergence between Gaussian distributions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Gaussian KL Divergence")
    print("=" * 60)
    
    def gaussian_kl(mu1, sigma1, mu2, sigma2):
        """KL divergence between two univariate Gaussians."""
        return (np.log(sigma2 / sigma1) + 
                (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5)
    
    print("D_KL(N(μ₁,σ₁²) || N(μ₂,σ₂²))")
    print("= log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2")
    
    # Effect of mean difference
    print("\nEffect of mean difference (σ₁ = σ₂ = 1):")
    print(f"{'μ₁':>6} {'μ₂':>6} {'D_KL':>12}")
    print("-" * 28)
    
    for mu1 in [0, 0, 0, 1, 2]:
        for mu2 in [0, 1, 2]:
            if mu1 <= mu2:
                kl = gaussian_kl(mu1, 1, mu2, 1)
                print(f"{mu1:>6} {mu2:>6} {kl:>12.4f}")
    
    # Effect of variance difference
    print("\nEffect of variance difference (μ₁ = μ₂ = 0):")
    print(f"{'σ₁':>6} {'σ₂':>6} {'D_KL':>12}")
    print("-" * 28)
    
    for sigma1 in [0.5, 1.0, 2.0]:
        for sigma2 in [0.5, 1.0, 2.0]:
            kl = gaussian_kl(0, sigma1, 0, sigma2)
            print(f"{sigma1:>6.1f} {sigma2:>6.1f} {kl:>12.4f}")


def example_forward_vs_reverse():
    """Forward vs reverse KL divergence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Forward vs Reverse KL")
    print("=" * 60)
    
    # True distribution: mixture of two Gaussians
    def p_true(x):
        """Bimodal distribution."""
        return 0.5 * stats.norm.pdf(x, -2, 0.5) + 0.5 * stats.norm.pdf(x, 2, 0.5)
    
    # Approximate with single Gaussian
    x = np.linspace(-5, 5, 1000)
    p = p_true(x)
    p = p / np.sum(p)  # Normalize to sum to 1 for discrete approx
    
    def kl_divergence(p, q):
        mask = (p > 1e-10) & (q > 1e-10)
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    # Forward KL: minimize D_KL(P || Q)
    # Optimal single Gaussian should cover both modes
    print("True P: mixture of N(-2, 0.5²) and N(2, 0.5²)")
    print("\nApproximating with single Gaussian Q = N(μ, σ²)")
    
    print("\nForward KL: D_KL(P || Q) - zero avoiding")
    print("Optimal Q tends to cover both modes (be broad)")
    
    # Try different approximations
    approximations = [
        ("N(0, 0.5)", 0, 0.5),
        ("N(0, 2.0)", 0, 2.0),
        ("N(0, 3.0)", 0, 3.0),
        ("N(-2, 0.5)", -2, 0.5),
        ("N(2, 0.5)", 2, 0.5),
    ]
    
    print(f"\n{'Q':>15} {'Forward KL':>15} {'Reverse KL':>15}")
    print("-" * 50)
    
    for name, mu, sigma in approximations:
        q = stats.norm.pdf(x, mu, sigma)
        q = q / np.sum(q)
        
        forward_kl = kl_divergence(p, q)
        reverse_kl = kl_divergence(q, p)
        
        print(f"{name:>15} {forward_kl:>15.4f} {reverse_kl:>15.4f}")
    
    print("\nForward KL prefers broad Q (covers all of P)")
    print("Reverse KL prefers narrow Q (focuses on one mode)")


def example_cross_entropy_relation():
    """Relationship between KL, entropy, and cross-entropy."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: KL, Entropy, and Cross-Entropy")
    print("=" * 60)
    
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    
    def cross_entropy(p, q):
        mask = p > 0
        return -np.sum(p[mask] * np.log(q[mask]))
    
    def kl_divergence(p, q):
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    
    H_p = entropy(p)
    H_pq = cross_entropy(p, q)
    D_kl = kl_divergence(p, q)
    
    print("Relationship: H(P, Q) = H(P) + D_KL(P || Q)")
    print(f"\nP = {p}")
    print(f"Q = {q}")
    print(f"\nH(P) = {H_p:.4f} nats")
    print(f"D_KL(P || Q) = {D_kl:.4f} nats")
    print(f"H(P, Q) = {H_pq:.4f} nats")
    print(f"\nVerification: H(P) + D_KL = {H_p + D_kl:.4f}")
    
    print("\n→ Minimizing cross-entropy = minimizing KL (when P fixed)")


def example_mutual_information():
    """Mutual information as KL divergence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Mutual Information as KL Divergence")
    print("=" * 60)
    
    # Joint distribution
    P_XY = np.array([
        [0.3, 0.1],
        [0.1, 0.5]
    ])
    
    print("Joint distribution P(X, Y):")
    print(P_XY)
    
    # Marginals
    P_X = P_XY.sum(axis=1)
    P_Y = P_XY.sum(axis=0)
    
    print(f"\nP(X) = {P_X}")
    print(f"P(Y) = {P_Y}")
    
    # Product of marginals
    P_X_P_Y = np.outer(P_X, P_Y)
    print(f"\nP(X)P(Y) = \n{P_X_P_Y}")
    
    # KL divergence
    def kl_divergence(p, q):
        p_flat = p.flatten()
        q_flat = q.flatten()
        mask = p_flat > 0
        return np.sum(p_flat[mask] * np.log(p_flat[mask] / q_flat[mask]))
    
    I_XY = kl_divergence(P_XY, P_X_P_Y)
    
    # Alternative: I(X;Y) = H(X) + H(Y) - H(X,Y)
    def entropy(p):
        p_flat = p.flatten()
        p_flat = p_flat[p_flat > 0]
        return -np.sum(p_flat * np.log(p_flat))
    
    H_X = entropy(P_X)
    H_Y = entropy(P_Y)
    H_XY = entropy(P_XY)
    I_XY_alt = H_X + H_Y - H_XY
    
    print(f"\nMutual Information:")
    print(f"  I(X;Y) = D_KL(P(X,Y) || P(X)P(Y)) = {I_XY:.4f} nats")
    print(f"  I(X;Y) = H(X) + H(Y) - H(X,Y) = {I_XY_alt:.4f} nats")


def example_vae_kl():
    """KL divergence in VAE (vs standard normal prior)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: KL Divergence in VAE")
    print("=" * 60)
    
    def kl_to_standard_normal(mu, log_var):
        """
        KL divergence from N(mu, exp(log_var)) to N(0, 1).
        Closed form: 0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        """
        return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    
    print("VAE encoder outputs μ and log(σ²)")
    print("KL term: D_KL(q(z|x) || p(z)) where p(z) = N(0, I)")
    print("\nClosed form: -0.5 × Σ(1 + log(σ²) - μ² - σ²)")
    
    # Different encoder outputs
    print(f"\n{'μ':>10} {'log(σ²)':>10} {'σ':>10} {'D_KL':>12}")
    print("-" * 48)
    
    test_cases = [
        (0, 0),      # Standard normal
        (0, -1),     # Narrow
        (0, 1),      # Wide
        (1, 0),      # Shifted
        (2, 0),      # More shifted
        (1, 1),      # Shifted and wide
    ]
    
    for mu, log_var in test_cases:
        mu_arr = np.array([mu])
        log_var_arr = np.array([log_var])
        sigma = np.sqrt(np.exp(log_var))
        kl = kl_to_standard_normal(mu_arr, log_var_arr)
        print(f"{mu:>10.1f} {log_var:>10.1f} {sigma:>10.2f} {kl:>12.4f}")
    
    print("\nMinimum at μ=0, σ²=1 (matches prior)")


def example_jensen_shannon():
    """Jensen-Shannon divergence (symmetric)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Jensen-Shannon Divergence")
    print("=" * 60)
    
    def kl_divergence(p, q):
        mask = (p > 0) & (q > 0)
        result = np.zeros_like(p)
        result[mask] = p[mask] * np.log(p[mask] / q[mask])
        return np.sum(result)
    
    def js_divergence(p, q):
        """Jensen-Shannon divergence."""
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    
    p = np.array([0.9, 0.1, 0.0])
    q = np.array([0.0, 0.1, 0.9])
    
    print("Jensen-Shannon: symmetric, bounded divergence")
    print("D_JS(P || Q) = 0.5 D_KL(P || M) + 0.5 D_KL(Q || M)")
    print("where M = 0.5(P + Q)")
    
    print(f"\nP = {p}")
    print(f"Q = {q}")
    
    kl_pq = kl_divergence(p + 1e-10, q + 1e-10)
    kl_qp = kl_divergence(q + 1e-10, p + 1e-10)
    js = js_divergence(p, q)
    
    print(f"\nD_KL(P || Q) = {kl_pq:.4f}")
    print(f"D_KL(Q || P) = {kl_qp:.4f}")
    print(f"D_JS(P || Q) = {js:.4f}")
    
    print(f"\nNote: D_JS is symmetric and finite even with disjoint support")
    print(f"Bounded: 0 ≤ D_JS ≤ log(2) ≈ {np.log(2):.4f}")


def example_multivariate_gaussian():
    """KL divergence for multivariate Gaussians."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Multivariate Gaussian KL")
    print("=" * 60)
    
    def mv_gaussian_kl(mu1, cov1, mu2, cov2):
        """KL divergence between two multivariate Gaussians."""
        d = len(mu1)
        
        # Terms
        log_det_ratio = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
        trace_term = np.trace(np.linalg.solve(cov2, cov1))
        
        diff = mu2 - mu1
        quad_term = diff @ np.linalg.solve(cov2, diff)
        
        return 0.5 * (log_det_ratio - d + trace_term + quad_term)
    
    # 2D example
    mu1 = np.array([0, 0])
    cov1 = np.array([[1, 0.5], [0.5, 1]])
    
    mu2 = np.array([0, 0])
    cov2 = np.eye(2)
    
    print("P = N(μ₁, Σ₁), Q = N(μ₂, Σ₂)")
    print(f"\nμ₁ = {mu1}, Σ₁ = \n{cov1}")
    print(f"\nμ₂ = {mu2}, Σ₂ = \n{cov2}")
    
    kl = mv_gaussian_kl(mu1, cov1, mu2, cov2)
    print(f"\nD_KL(P || Q) = {kl:.4f} nats")
    
    # Different scenarios
    print("\nDifferent scenarios:")
    
    cases = [
        ("Same distribution", [0, 0], np.eye(2), [0, 0], np.eye(2)),
        ("Mean shift", [1, 1], np.eye(2), [0, 0], np.eye(2)),
        ("Scale difference", [0, 0], 2*np.eye(2), [0, 0], np.eye(2)),
        ("Correlation", [0, 0], np.array([[1, 0.9], [0.9, 1]]), [0, 0], np.eye(2)),
    ]
    
    print(f"\n{'Case':>20} {'D_KL':>12}")
    print("-" * 35)
    
    for name, m1, c1, m2, c2 in cases:
        kl = mv_gaussian_kl(np.array(m1), np.array(c1), np.array(m2), np.array(c2))
        print(f"{name:>20} {kl:>12.4f}")


def example_kl_estimation():
    """Estimating KL divergence from samples."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: KL Estimation from Samples")
    print("=" * 60)
    
    np.random.seed(42)
    
    # True distributions
    mu_p, sigma_p = 0, 1
    mu_q, sigma_q = 1, 1.5
    
    # True KL
    true_kl = (np.log(sigma_q / sigma_p) + 
               (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5)
    
    print("P = N(0, 1), Q = N(1, 1.5²)")
    print(f"True D_KL(P || Q) = {true_kl:.4f} nats")
    
    # Monte Carlo estimation
    print("\nMonte Carlo estimation: average log(p(x)/q(x)) over x ~ P")
    
    print(f"\n{'n':>8} {'Estimate':>12} {'Std':>12} {'Error':>12}")
    print("-" * 48)
    
    for n in [100, 500, 1000, 5000, 10000]:
        estimates = []
        for _ in range(20):  # Multiple trials
            samples = np.random.normal(mu_p, sigma_p, n)
            
            log_p = stats.norm.logpdf(samples, mu_p, sigma_p)
            log_q = stats.norm.logpdf(samples, mu_q, sigma_q)
            
            kl_est = np.mean(log_p - log_q)
            estimates.append(kl_est)
        
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        error = mean_est - true_kl
        
        print(f"{n:>8} {mean_est:>12.4f} {std_est:>12.4f} {error:>12.4f}")


def example_knowledge_distillation():
    """KL divergence in knowledge distillation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Knowledge Distillation")
    print("=" * 60)
    
    def softmax(logits, temperature=1.0):
        exp_logits = np.exp(logits / temperature)
        return exp_logits / np.sum(exp_logits)
    
    def kl_divergence(p, q):
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-10)))
    
    # Teacher logits (confident)
    teacher_logits = np.array([5.0, 2.0, 1.0, 0.5])
    
    # Student logits (less confident)
    student_logits = np.array([2.0, 1.5, 1.0, 0.8])
    
    print("Knowledge Distillation: Transfer teacher's knowledge to student")
    print("Use soft labels (with temperature) instead of hard labels")
    
    print(f"\nTeacher logits: {teacher_logits}")
    print(f"Student logits: {student_logits}")
    
    print(f"\n{'T':>4} {'Teacher probs':>25} {'Student probs':>25} {'KL':>10}")
    print("-" * 70)
    
    for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
        p_teacher = softmax(teacher_logits, T)
        p_student = softmax(student_logits, T)
        kl = kl_divergence(p_teacher, p_student)
        
        t_str = "[" + ", ".join(f"{x:.2f}" for x in p_teacher) + "]"
        s_str = "[" + ", ".join(f"{x:.2f}" for x in p_student) + "]"
        print(f"{T:>4.1f} {t_str:>25} {s_str:>25} {kl:>10.4f}")
    
    print("\nHigher temperature → softer distributions → more knowledge transfer")


def example_information_bottleneck():
    """Information bottleneck principle."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Information Bottleneck")
    print("=" * 60)
    
    print("Information Bottleneck: Find representation Z that")
    print("  - Maximizes I(Z; Y) (predictive)")
    print("  - Minimizes I(X; Z) (compressed)")
    
    print("\nObjective: max I(Z; Y) - β I(X; Z)")
    
    # Simulate X, Y, Z distributions
    np.random.seed(42)
    n = 1000
    
    # X: input features (4D)
    X = np.random.randn(n, 4)
    
    # Y: target (depends on first 2 dims of X)
    Y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Z: compressed representation
    # Option 1: Keep all info (Z = X)
    # Option 2: Keep relevant info (Z = X[:, :2])
    # Option 3: Too compressed (Z = sign(X[:, 0]))
    
    def estimate_mi(a, b, bins=10):
        """Rough MI estimate via histogram."""
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        
        # Discretize
        a_disc = np.digitize(a, np.linspace(a.min(), a.max(), bins))
        b_disc = np.digitize(b, np.linspace(b.min(), b.max(), bins))
        
        # Joint and marginal
        from collections import Counter
        
        joint = Counter(zip(map(tuple, a_disc), map(tuple, b_disc)))
        p_joint = np.array(list(joint.values())) / n
        
        marginal_a = Counter(map(tuple, a_disc))
        marginal_b = Counter(map(tuple, b_disc))
        
        p_a = np.array(list(marginal_a.values())) / n
        p_b = np.array(list(marginal_b.values())) / n
        
        # Entropies
        H_joint = -np.sum(p_joint * np.log(p_joint + 1e-10))
        H_a = -np.sum(p_a * np.log(p_a + 1e-10))
        H_b = -np.sum(p_b * np.log(p_b + 1e-10))
        
        return H_a + H_b - H_joint
    
    print("\nTrade-off analysis:")
    print(f"{'Representation Z':>20} {'I(X;Z)':>12} {'I(Z;Y)':>12} {'IB (β=0.1)':>15}")
    print("-" * 65)
    
    representations = [
        ("Z = X (all)", X),
        ("Z = X[:,:2] (relevant)", X[:, :2]),
        ("Z = X[:,:1] (partial)", X[:, :1]),
        ("Z = sign(X[:,0])", np.sign(X[:, 0])),
    ]
    
    for name, Z in representations:
        I_XZ = estimate_mi(X, Z)
        I_ZY = estimate_mi(Z, Y)
        IB = I_ZY - 0.1 * I_XZ
        print(f"{name:>20} {I_XZ:>12.4f} {I_ZY:>12.4f} {IB:>15.4f}")
    
    print("\nOptimal Z keeps relevant info while discarding irrelevant features")


if __name__ == "__main__":
    example_basic_kl()
    example_bernoulli_kl()
    example_gaussian_kl()
    example_forward_vs_reverse()
    example_cross_entropy_relation()
    example_mutual_information()
    example_vae_kl()
    example_jensen_shannon()
    example_multivariate_gaussian()
    example_kl_estimation()
    example_knowledge_distillation()
    example_information_bottleneck()
