"""
Entropy - Examples
==================
Computing and applying entropy in various contexts.
"""

import numpy as np
from scipy import stats
from collections import Counter


def example_shannon_entropy():
    """Compute Shannon entropy for discrete distributions."""
    print("=" * 60)
    print("EXAMPLE 1: Shannon Entropy")
    print("=" * 60)
    
    def entropy(p):
        """Compute entropy in bits."""
        p = np.array(p)
        p = p[p > 0]  # Remove zeros
        return -np.sum(p * np.log2(p))
    
    # Different distributions
    distributions = {
        'Uniform (4 outcomes)': [0.25, 0.25, 0.25, 0.25],
        'Biased coin (0.9, 0.1)': [0.9, 0.1],
        'Fair coin': [0.5, 0.5],
        'Deterministic': [1.0, 0.0, 0.0, 0.0],
        'Slightly biased': [0.4, 0.3, 0.2, 0.1]
    }
    
    print("H(X) = -Σ p(x) log₂ p(x)")
    print(f"\n{'Distribution':<30} {'Entropy (bits)':>15}")
    print("-" * 50)
    
    for name, p in distributions.items():
        H = entropy(p)
        print(f"{name:<30} {H:>15.4f}")
    
    print("\nKey observations:")
    print("  - Uniform distribution has maximum entropy")
    print("  - Deterministic distribution has zero entropy")
    print("  - More peaked distributions have lower entropy")


def example_binary_entropy():
    """Binary entropy function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Binary Entropy Function")
    print("=" * 60)
    
    def binary_entropy(p):
        """H_b(p) = -p log p - (1-p) log(1-p)"""
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    print("Binary entropy: H_b(p) = -p log₂(p) - (1-p) log₂(1-p)")
    print(f"\n{'p':>8} {'H_b(p)':>12}")
    print("-" * 25)
    
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        H = binary_entropy(p)
        print(f"{p:>8.1f} {H:>12.4f}")
    
    print("\nMaximum at p = 0.5 with H_b(0.5) = 1 bit")
    print("Symmetric: H_b(p) = H_b(1-p)")


def example_joint_entropy():
    """Joint and conditional entropy."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Joint and Conditional Entropy")
    print("=" * 60)
    
    # Joint distribution P(X, Y)
    # X: Weather (0=sunny, 1=rainy)
    # Y: Umbrella (0=no, 1=yes)
    joint = np.array([
        [0.4, 0.1],   # Sunny: no umbrella, umbrella
        [0.1, 0.4]    # Rainy: no umbrella, umbrella
    ])
    
    print("Joint distribution P(X, Y):")
    print("         Y=0    Y=1")
    print(f"X=0     {joint[0,0]:.1f}    {joint[0,1]:.1f}")
    print(f"X=1     {joint[1,0]:.1f}    {joint[1,1]:.1f}")
    
    # Marginals
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    
    print(f"\nMarginals:")
    print(f"  P(X): {p_x}")
    print(f"  P(Y): {p_y}")
    
    # Entropies
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    H_X = entropy(p_x)
    H_Y = entropy(p_y)
    H_XY = entropy(joint.flatten())
    
    # Conditional entropy H(Y|X)
    H_Y_given_X = 0
    for i in range(2):
        p_y_given_xi = joint[i, :] / p_x[i]
        H_Y_given_X += p_x[i] * entropy(p_y_given_xi)
    
    print(f"\nEntropies:")
    print(f"  H(X) = {H_X:.4f} bits")
    print(f"  H(Y) = {H_Y:.4f} bits")
    print(f"  H(X,Y) = {H_XY:.4f} bits")
    print(f"  H(Y|X) = {H_Y_given_X:.4f} bits")
    
    # Verify chain rule
    print(f"\nChain rule verification:")
    print(f"  H(X,Y) = {H_XY:.4f}")
    print(f"  H(X) + H(Y|X) = {H_X + H_Y_given_X:.4f}")
    
    # Mutual information
    I_XY = H_Y - H_Y_given_X
    print(f"\nMutual information I(X;Y) = H(Y) - H(Y|X) = {I_XY:.4f} bits")


def example_differential_entropy():
    """Differential entropy for continuous distributions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Differential Entropy")
    print("=" * 60)
    
    print("Differential entropy: h(X) = -∫ f(x) log f(x) dx")
    
    # Gaussian
    def gaussian_entropy(sigma):
        return 0.5 * np.log2(2 * np.pi * np.e * sigma**2)
    
    # Uniform
    def uniform_entropy(a, b):
        return np.log2(b - a)
    
    # Exponential
    def exponential_entropy(lam):
        return (1 - np.log2(lam))  # In bits
    
    print("\nGaussian N(0, σ²): h(X) = ½ log₂(2πeσ²)")
    print(f"{'σ':>8} {'h(X) (bits)':>15}")
    print("-" * 28)
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        print(f"{sigma:>8.1f} {gaussian_entropy(sigma):>15.4f}")
    
    print("\nUniform[a, b]: h(X) = log₂(b-a)")
    print(f"{'[a, b]':>12} {'h(X) (bits)':>15}")
    print("-" * 32)
    for a, b in [(0, 1), (0, 2), (0, 10), (-5, 5)]:
        print(f"[{a}, {b}]".rjust(12) + f" {uniform_entropy(a, b):>15.4f}")
    
    print("\nNote: Differential entropy can be negative!")
    print(f"  Uniform[0, 0.5]: h(X) = {uniform_entropy(0, 0.5):.4f} bits")


def example_max_entropy():
    """Maximum entropy distributions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Maximum Entropy Principle")
    print("=" * 60)
    
    print("Maximum entropy distribution subject to constraints:")
    
    # Discrete: uniform is max entropy
    n_outcomes = 6
    uniform = np.ones(n_outcomes) / n_outcomes
    H_uniform = -np.sum(uniform * np.log2(uniform))
    
    print(f"\n1. Discrete with {n_outcomes} outcomes:")
    print(f"   Max entropy = log₂({n_outcomes}) = {np.log2(n_outcomes):.4f} bits")
    print(f"   Achieved by uniform: H = {H_uniform:.4f} bits")
    
    # Fixed mean: exponential
    print("\n2. Non-negative with fixed mean μ:")
    print("   Max entropy distribution: Exponential(λ=1/μ)")
    
    # Fixed mean and variance: Gaussian
    print("\n3. Fixed mean and variance (μ, σ²):")
    print("   Max entropy distribution: Gaussian N(μ, σ²)")
    
    # Numerical verification
    print("\nNumerical verification (comparing distributions with same variance):")
    
    sigma = 1.0
    var = sigma**2
    
    # Gaussian entropy
    h_gaussian = 0.5 * np.log2(2 * np.pi * np.e * var)
    
    # Uniform with same variance: Uniform[-a, a] has var = a²/3
    a = np.sqrt(3 * var)
    h_uniform = np.log2(2 * a)
    
    # Laplace with same variance: var = 2b²
    b = np.sqrt(var / 2)
    h_laplace = 1 + np.log2(2 * b)
    
    print(f"   Variance = {var}")
    print(f"   Gaussian:  h = {h_gaussian:.4f} bits")
    print(f"   Uniform:   h = {h_uniform:.4f} bits")
    print(f"   Laplace:   h = {h_laplace:.4f} bits")
    print("   Gaussian has maximum differential entropy!")


def example_decision_tree():
    """Information gain for decision trees."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Information Gain in Decision Trees")
    print("=" * 60)
    
    def entropy(labels):
        """Compute entropy of label distribution."""
        n = len(labels)
        if n == 0:
            return 0
        counts = Counter(labels)
        probs = np.array([c / n for c in counts.values()])
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def information_gain(data, labels, feature_idx):
        """Compute information gain from splitting on a feature."""
        H_before = entropy(labels)
        
        # Split by feature
        values = set(data[:, feature_idx])
        H_after = 0
        
        for v in values:
            mask = data[:, feature_idx] == v
            subset_labels = labels[mask]
            weight = np.sum(mask) / len(labels)
            H_after += weight * entropy(subset_labels)
        
        return H_before - H_after
    
    # Example: Play tennis dataset
    # Features: [Outlook, Temperature, Humidity, Wind]
    # Outlook: 0=sunny, 1=overcast, 2=rain
    # Temperature: 0=hot, 1=mild, 2=cool
    # Humidity: 0=high, 1=normal
    # Wind: 0=weak, 1=strong
    
    data = np.array([
        [0, 0, 0, 0],  # sunny, hot, high, weak -> No
        [0, 0, 0, 1],  # sunny, hot, high, strong -> No
        [1, 0, 0, 0],  # overcast, hot, high, weak -> Yes
        [2, 1, 0, 0],  # rain, mild, high, weak -> Yes
        [2, 2, 1, 0],  # rain, cool, normal, weak -> Yes
        [2, 2, 1, 1],  # rain, cool, normal, strong -> No
        [1, 2, 1, 1],  # overcast, cool, normal, strong -> Yes
        [0, 1, 0, 0],  # sunny, mild, high, weak -> No
        [0, 2, 1, 0],  # sunny, cool, normal, weak -> Yes
        [2, 1, 1, 0],  # rain, mild, normal, weak -> Yes
        [0, 1, 1, 1],  # sunny, mild, normal, strong -> Yes
        [1, 1, 0, 1],  # overcast, mild, high, strong -> Yes
        [1, 0, 1, 0],  # overcast, hot, normal, weak -> Yes
        [2, 1, 0, 1],  # rain, mild, high, strong -> No
    ])
    
    labels = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])  # 0=No, 1=Yes
    
    feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    
    print("Play Tennis Dataset")
    print(f"Total samples: {len(labels)}")
    print(f"Yes: {np.sum(labels)}, No: {len(labels) - np.sum(labels)}")
    print(f"\nEntropy before split: {entropy(labels):.4f} bits")
    
    print(f"\n{'Feature':<15} {'Information Gain':>20}")
    print("-" * 40)
    
    gains = {}
    for i, name in enumerate(feature_names):
        ig = information_gain(data, labels, i)
        gains[name] = ig
        print(f"{name:<15} {ig:>20.4f}")
    
    best_feature = max(gains, key=gains.get)
    print(f"\nBest feature to split on: {best_feature}")


def example_entropy_estimation():
    """Estimating entropy from samples."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Entropy Estimation")
    print("=" * 60)
    
    np.random.seed(42)
    
    # True distribution
    true_probs = np.array([0.4, 0.3, 0.2, 0.1])
    true_entropy = -np.sum(true_probs * np.log2(true_probs))
    
    print(f"True distribution: {true_probs}")
    print(f"True entropy: {true_entropy:.4f} bits")
    
    def plugin_estimate(samples):
        """Plug-in entropy estimator."""
        counts = Counter(samples)
        n = len(samples)
        probs = np.array([counts.get(i, 0) / n for i in range(4)])
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    def miller_madow(samples):
        """Miller-Madow bias-corrected estimator."""
        plugin = plugin_estimate(samples)
        counts = Counter(samples)
        m = len([c for c in counts.values() if c > 0])
        n = len(samples)
        return plugin + (m - 1) / (2 * n * np.log(2))
    
    print(f"\n{'n':>8} {'Plugin':>12} {'Miller-Madow':>15} {'True':>12}")
    print("-" * 52)
    
    for n in [10, 50, 100, 500, 1000, 5000]:
        samples = np.random.choice(4, n, p=true_probs)
        plugin = plugin_estimate(samples)
        mm = miller_madow(samples)
        print(f"{n:>8} {plugin:>12.4f} {mm:>15.4f} {true_entropy:>12.4f}")
    
    print("\nPlugin estimator is biased low for small samples")
    print("Miller-Madow adds bias correction")


def example_cross_entropy_loss():
    """Cross-entropy as a loss function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Cross-Entropy Loss")
    print("=" * 60)
    
    def cross_entropy_loss(y_true, y_pred):
        """Binary cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def multiclass_cross_entropy(y_true, y_pred):
        """Multiclass cross-entropy (y_true is one-hot)."""
        y_pred = np.clip(y_pred, 1e-10, 1)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    # Binary classification
    print("Binary Cross-Entropy:")
    print("L = -[y log(p) + (1-y) log(1-p)]")
    
    y_true = np.array([1, 0, 1, 1, 0])
    
    print(f"\nTrue labels: {y_true}")
    print(f"\n{'Prediction':>15} {'Loss':>12}")
    print("-" * 30)
    
    for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
        y_pred = np.where(y_true == 1, conf, 1 - conf)
        loss = cross_entropy_loss(y_true, y_pred)
        print(f"p = {conf:.1f}".rjust(15) + f" {loss:>12.4f}")
    
    # Perfect prediction
    y_pred_perfect = y_true.astype(float)
    y_pred_perfect = np.clip(y_pred_perfect, 0.001, 0.999)
    loss_perfect = cross_entropy_loss(y_true, y_pred_perfect)
    print(f"{'Perfect':>15} {loss_perfect:>12.4f}")
    
    print("\n" + "-" * 30)
    print("Multiclass Cross-Entropy:")
    
    # 3-class example
    y_true_mc = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Good predictions
    y_pred_good = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7]
    ])
    
    # Bad predictions
    y_pred_bad = np.array([
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ])
    
    print(f"Good predictions loss: {multiclass_cross_entropy(y_true_mc, y_pred_good):.4f}")
    print(f"Bad predictions loss:  {multiclass_cross_entropy(y_true_mc, y_pred_bad):.4f}")


def example_entropy_regularization():
    """Entropy regularization in RL/optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Entropy Regularization")
    print("=" * 60)
    
    def entropy(p):
        """Entropy of probability distribution."""
        p = np.clip(p, 1e-10, 1)
        return -np.sum(p * np.log(p))
    
    def softmax(logits, temperature=1.0):
        """Softmax with temperature."""
        exp_logits = np.exp(logits / temperature)
        return exp_logits / np.sum(exp_logits)
    
    # Policy in RL
    n_actions = 4
    Q_values = np.array([1.0, 2.0, 0.5, 0.8])  # Action values
    
    print("Reinforcement Learning: Entropy encourages exploration")
    print(f"\nQ-values: {Q_values}")
    
    print(f"\n{'Temperature':>12} {'Policy':>30} {'Entropy':>12}")
    print("-" * 60)
    
    for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
        policy = softmax(Q_values, temp)
        H = entropy(policy)
        policy_str = "[" + ", ".join(f"{p:.2f}" for p in policy) + "]"
        print(f"{temp:>12.1f} {policy_str:>30} {H:>12.4f}")
    
    print("\nHigher temperature → more uniform policy → higher entropy")
    print("Entropy bonus in loss: L = -E[R] - β H(π)")


def example_knn_entropy():
    """K-NN based entropy estimation for continuous variables."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: K-NN Entropy Estimation")
    print("=" * 60)
    
    np.random.seed(42)
    
    def knn_entropy(X, k=3):
        """
        K-NN entropy estimator (Kozachenko-Leonenko).
        For 1D data.
        """
        n = len(X)
        X_sorted = np.sort(X)
        
        # Find k-th nearest neighbor distances
        distances = []
        for i in range(n):
            dists = np.abs(X_sorted - X_sorted[i])
            dists = np.sort(dists)
            if k < len(dists):
                distances.append(dists[k])  # k-th nearest (excluding self)
        
        distances = np.array(distances)
        distances = distances[distances > 0]
        
        # Entropy estimate
        d = 1  # dimension
        c_d = 2  # volume of unit ball in 1D
        H = d * np.mean(np.log(distances)) + np.log(n * c_d / k) + np.euler_gamma
        
        return H / np.log(2)  # Convert to bits
    
    # Compare Gaussian samples
    print("Estimating differential entropy of Gaussian")
    
    for sigma in [0.5, 1.0, 2.0]:
        true_h = 0.5 * np.log2(2 * np.pi * np.e * sigma**2)
        
        estimates = []
        for _ in range(10):
            X = np.random.normal(0, sigma, 1000)
            estimates.append(knn_entropy(X, k=5))
        
        est_mean = np.mean(estimates)
        est_std = np.std(estimates)
        
        print(f"\nσ = {sigma}:")
        print(f"  True h(X):     {true_h:.4f} bits")
        print(f"  KNN estimate:  {est_mean:.4f} ± {est_std:.4f} bits")


def example_renyi_entropy():
    """Rényi entropy generalizations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Rényi Entropy")
    print("=" * 60)
    
    def renyi_entropy(p, alpha):
        """Rényi entropy of order alpha."""
        if alpha == 1:
            # Shannon entropy (limit)
            return -np.sum(p * np.log2(p + 1e-10))
        else:
            return np.log2(np.sum(p**alpha)) / (1 - alpha)
    
    # Test distribution
    p = np.array([0.5, 0.25, 0.125, 0.125])
    
    print(f"Distribution: {p}")
    print(f"\n{'α':>8} {'H_α(X)':>12} {'Name':>20}")
    print("-" * 45)
    
    alphas = [0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    names = ['≈ Max entropy', 'α=0.5', 'Shannon', 'Collision', 'α=5', 'α=10', '≈ Min entropy']
    
    for alpha, name in zip(alphas, names):
        H = renyi_entropy(p, alpha)
        print(f"{alpha:>8.2f} {H:>12.4f} {name:>20}")
    
    print(f"\nMin entropy (α→∞): -log₂(max p) = {-np.log2(np.max(p)):.4f}")
    print(f"Max entropy (α→0): log₂(support) = {np.log2(len(p)):.4f}")


def example_entropy_rate():
    """Entropy rate of a stochastic process."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Entropy Rate of Markov Chain")
    print("=" * 60)
    
    # Transition matrix
    P = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    
    print("Markov chain with transition matrix:")
    print(f"P = \n{P}")
    
    # Stationary distribution
    # π P = π and sum(π) = 1
    # (P.T - I) π = 0
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    pi = np.abs(eigvecs[:, idx])
    pi = pi / np.sum(pi)
    
    print(f"\nStationary distribution π = {pi}")
    
    # Entropy rate: H(X) = -Σ_i π_i Σ_j P_ij log P_ij
    H_rate = 0
    for i in range(2):
        for j in range(2):
            if P[i, j] > 0:
                H_rate -= pi[i] * P[i, j] * np.log2(P[i, j])
    
    print(f"\nEntropy rate H(X) = {H_rate:.4f} bits per symbol")
    
    # Compare to IID with same marginal
    H_iid = -np.sum(pi * np.log2(pi))
    print(f"IID entropy with same marginal: {H_iid:.4f} bits")
    print(f"\nMarkov has lower entropy rate due to temporal dependencies")


if __name__ == "__main__":
    example_shannon_entropy()
    example_binary_entropy()
    example_joint_entropy()
    example_differential_entropy()
    example_max_entropy()
    example_decision_tree()
    example_entropy_estimation()
    example_cross_entropy_loss()
    example_entropy_regularization()
    example_knn_entropy()
    example_renyi_entropy()
    example_entropy_rate()
