"""
Mutual Information - Examples
=============================
Computing and applying mutual information.
"""

import numpy as np
from scipy import stats
from scipy.special import digamma
from collections import Counter


def example_basic_mi():
    """Basic mutual information computation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Mutual Information")
    print("=" * 60)
    
    def mutual_information(p_xy):
        """Compute I(X;Y) from joint distribution."""
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        mi = 0
        for i in range(p_xy.shape[0]):
            for j in range(p_xy.shape[1]):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        return mi
    
    # Weather and umbrella example
    # P(Sunny) = 0.7, P(Rainy) = 0.3
    # If Sunny: P(No umbrella) = 0.9, P(Umbrella) = 0.1
    # If Rainy: P(No umbrella) = 0.2, P(Umbrella) = 0.8
    
    p_xy = np.array([
        [0.63, 0.07],  # Sunny: no umbrella, umbrella
        [0.06, 0.24]   # Rainy: no umbrella, umbrella
    ])
    
    print("Weather (X) and Umbrella (Y):")
    print("Joint distribution P(X, Y):")
    print(f"                No Umbrella  Umbrella")
    print(f"  Sunny             {p_xy[0,0]:.2f}       {p_xy[0,1]:.2f}")
    print(f"  Rainy             {p_xy[1,0]:.2f}       {p_xy[1,1]:.2f}")
    
    mi = mutual_information(p_xy)
    print(f"\nI(Weather; Umbrella) = {mi:.4f} nats")
    
    # Verify with entropy formula
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    
    H_X = entropy(p_xy.sum(axis=1))
    H_Y = entropy(p_xy.sum(axis=0))
    H_XY = entropy(p_xy.flatten())
    
    mi_alt = H_X + H_Y - H_XY
    print(f"\nVerification: H(X) + H(Y) - H(X,Y)")
    print(f"  H(X) = {H_X:.4f}")
    print(f"  H(Y) = {H_Y:.4f}")
    print(f"  H(X,Y) = {H_XY:.4f}")
    print(f"  I(X;Y) = {mi_alt:.4f}")


def example_mi_entropy_relations():
    """Mutual information and entropy relationships."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MI-Entropy Relationships")
    print("=" * 60)
    
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    
    def conditional_entropy(p_xy, axis):
        """H(Y|X) if axis=1, H(X|Y) if axis=0."""
        p_cond = p_xy.sum(axis=axis)
        h_cond = 0
        
        if axis == 1:  # H(Y|X)
            for i in range(p_xy.shape[0]):
                if p_cond[i] > 0:
                    p_y_given_x = p_xy[i, :] / p_cond[i]
                    h_cond += p_cond[i] * entropy(p_y_given_x)
        else:  # H(X|Y)
            for j in range(p_xy.shape[1]):
                if p_cond[j] > 0:
                    p_x_given_y = p_xy[:, j] / p_cond[j]
                    h_cond += p_cond[j] * entropy(p_x_given_y)
        
        return h_cond
    
    p_xy = np.array([
        [0.3, 0.1],
        [0.1, 0.5]
    ])
    
    print("Joint distribution P(X, Y):")
    print(p_xy)
    
    H_X = entropy(p_xy.sum(axis=1))
    H_Y = entropy(p_xy.sum(axis=0))
    H_XY = entropy(p_xy.flatten())
    H_Y_given_X = conditional_entropy(p_xy, axis=1)
    H_X_given_Y = conditional_entropy(p_xy, axis=0)
    
    print(f"\nEntropy values:")
    print(f"  H(X) = {H_X:.4f}")
    print(f"  H(Y) = {H_Y:.4f}")
    print(f"  H(X,Y) = {H_XY:.4f}")
    print(f"  H(Y|X) = {H_Y_given_X:.4f}")
    print(f"  H(X|Y) = {H_X_given_Y:.4f}")
    
    # Different MI formulas
    mi_1 = H_X + H_Y - H_XY
    mi_2 = H_X - H_X_given_Y
    mi_3 = H_Y - H_Y_given_X
    
    print(f"\nMutual Information (different formulas):")
    print(f"  I(X;Y) = H(X) + H(Y) - H(X,Y) = {mi_1:.4f}")
    print(f"  I(X;Y) = H(X) - H(X|Y) = {mi_2:.4f}")
    print(f"  I(X;Y) = H(Y) - H(Y|X) = {mi_3:.4f}")


def example_gaussian_mi():
    """Mutual information for Gaussian variables."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Gaussian Mutual Information")
    print("=" * 60)
    
    def gaussian_mi(rho):
        """MI for bivariate Gaussian with correlation rho."""
        return -0.5 * np.log(1 - rho**2)
    
    print("For jointly Gaussian (X, Y) with correlation ρ:")
    print("I(X; Y) = -½ log(1 - ρ²)")
    
    print(f"\n{'ρ':>8} {'I(X;Y)':>12} nats")
    print("-" * 25)
    
    for rho in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        mi = gaussian_mi(rho)
        print(f"{rho:>8.2f} {mi:>12.4f}")
    
    print("\nNote: I(X;Y) → ∞ as ρ → 1 (perfect dependence)")
    print("      I(X;Y) = 0 when ρ = 0 (independence)")
    
    # Numerical verification
    print("\nNumerical verification with samples:")
    np.random.seed(42)
    
    rho = 0.7
    n = 10000
    
    # Generate correlated Gaussians
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    samples = np.random.multivariate_normal(mean, cov, n)
    
    # Estimate MI using covariance
    sample_cov = np.cov(samples.T)
    det_x = sample_cov[0, 0]
    det_y = sample_cov[1, 1]
    det_xy = np.linalg.det(sample_cov)
    
    mi_est = 0.5 * np.log(det_x * det_y / det_xy)
    mi_true = gaussian_mi(rho)
    
    print(f"  True MI (ρ={rho}): {mi_true:.4f}")
    print(f"  Estimated MI: {mi_est:.4f}")


def example_mi_kl_relation():
    """MI as KL divergence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: MI as KL Divergence")
    print("=" * 60)
    
    def kl_divergence(p, q):
        """KL(P || Q) for flattened arrays."""
        p_flat = p.flatten()
        q_flat = q.flatten()
        mask = p_flat > 0
        return np.sum(p_flat[mask] * np.log(p_flat[mask] / q_flat[mask]))
    
    # Joint distribution
    p_xy = np.array([
        [0.4, 0.1],
        [0.2, 0.3]
    ])
    
    # Marginals
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    
    # Product of marginals
    p_x_p_y = np.outer(p_x, p_y)
    
    print("I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))")
    print("\nJoint P(X,Y):")
    print(p_xy)
    print("\nProduct P(X)P(Y):")
    print(p_x_p_y)
    
    mi = kl_divergence(p_xy, p_x_p_y)
    print(f"\nI(X;Y) = D_KL(P(X,Y) || P(X)P(Y)) = {mi:.4f} nats")
    
    # Interpretation
    print("\nInterpretation:")
    print("  MI measures how different the joint distribution is")
    print("  from assuming X and Y are independent.")


def example_feature_selection():
    """MI for feature selection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Feature Selection with MI")
    print("=" * 60)
    
    np.random.seed(42)
    n = 1000
    
    # Generate features and target
    # X1: highly relevant to Y
    # X2: moderately relevant
    # X3: irrelevant (noise)
    
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)  # Pure noise
    
    # Target depends on X1 and X2
    Y = (X1 + 0.5 * X2 + 0.3 * np.random.randn(n) > 0).astype(int)
    
    def estimate_mi_discrete_cont(X, Y_discrete, bins=10):
        """Estimate MI between continuous X and discrete Y."""
        # Discretize X
        X_bins = np.digitize(X, np.linspace(X.min(), X.max(), bins))
        
        # Count joint and marginal
        joint = Counter(zip(X_bins, Y_discrete))
        n_total = len(X)
        
        p_x = Counter(X_bins)
        p_y = Counter(Y_discrete)
        
        mi = 0
        for (x, y), count in joint.items():
            p_xy = count / n_total
            px = p_x[x] / n_total
            py = p_y[y] / n_total
            if p_xy > 0:
                mi += p_xy * np.log(p_xy / (px * py))
        
        return mi
    
    print("Features: X1 (relevant), X2 (moderately relevant), X3 (noise)")
    print("Target: Y = sign(X1 + 0.5*X2 + noise)")
    
    mi_x1_y = estimate_mi_discrete_cont(X1, Y)
    mi_x2_y = estimate_mi_discrete_cont(X2, Y)
    mi_x3_y = estimate_mi_discrete_cont(X3, Y)
    
    print(f"\nMutual Information scores:")
    print(f"  I(X1; Y) = {mi_x1_y:.4f} (should be highest)")
    print(f"  I(X2; Y) = {mi_x2_y:.4f} (medium)")
    print(f"  I(X3; Y) = {mi_x3_y:.4f} (should be ~0)")
    
    print("\nFeature ranking by MI:")
    scores = [('X1', mi_x1_y), ('X2', mi_x2_y), ('X3', mi_x3_y)]
    for rank, (name, score) in enumerate(sorted(scores, key=lambda x: -x[1]), 1):
        print(f"  {rank}. {name}: {score:.4f}")


def example_pointwise_mi():
    """Pointwise mutual information."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Pointwise Mutual Information (PMI)")
    print("=" * 60)
    
    # Word co-occurrence example
    words = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'fast']
    
    # Simplified co-occurrence counts (within window)
    cooccur = {
        ('the', 'cat'): 50, ('the', 'dog'): 40, ('the', 'mat'): 30,
        ('cat', 'sat'): 45, ('cat', 'mat'): 20,
        ('dog', 'ran'): 40, ('dog', 'fast'): 15,
        ('sat', 'on'): 35, ('sat', 'mat'): 25,
        ('ran', 'fast'): 30,
    }
    
    # Word counts
    word_counts = {
        'the': 200, 'cat': 80, 'sat': 70, 'on': 150,
        'mat': 60, 'dog': 75, 'ran': 50, 'fast': 40
    }
    
    total_pairs = sum(cooccur.values())
    total_words = sum(word_counts.values())
    
    def compute_pmi(word1, word2):
        """Compute PMI(word1, word2)."""
        joint = cooccur.get((word1, word2), cooccur.get((word2, word1), 0))
        if joint == 0:
            return float('-inf')
        
        p_joint = joint / total_pairs
        p_w1 = word_counts[word1] / total_words
        p_w2 = word_counts[word2] / total_words
        
        return np.log(p_joint / (p_w1 * p_w2))
    
    print("Pointwise Mutual Information (PMI)")
    print("PMI(x, y) = log(P(x,y) / (P(x)P(y)))")
    print("\nWord co-occurrence PMI:")
    print(f"{'Word Pair':>20} {'PMI':>10}")
    print("-" * 35)
    
    pairs = [
        ('cat', 'sat'),
        ('dog', 'ran'),
        ('ran', 'fast'),
        ('the', 'cat'),
        ('cat', 'mat'),
    ]
    
    for w1, w2 in pairs:
        pmi = compute_pmi(w1, w2)
        print(f"{w1 + '-' + w2:>20} {pmi:>10.3f}")
    
    print("\nPositive PMI: words co-occur more than expected")
    print("Negative PMI: words co-occur less than expected")


def example_conditional_mi():
    """Conditional mutual information."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Conditional Mutual Information")
    print("=" * 60)
    
    # Example: X, Y, Z where Z mediates the relationship
    # P(X, Y, Z)
    
    # Simpler example with discrete variables
    # Z: Weather (0=sunny, 1=rainy)
    # X: Outdoor activity (0=no, 1=yes)
    # Y: Umbrella (0=no, 1=yes)
    
    # P(Z=sunny) = 0.7, P(Z=rainy) = 0.3
    # Given weather, X and Y become more independent
    
    # Joint P(X, Y, Z)
    p_xyz = np.zeros((2, 2, 2))
    
    # Z=0 (sunny)
    p_xyz[0, 0, 0] = 0.1 * 0.7   # No activity, no umbrella, sunny
    p_xyz[0, 1, 0] = 0.05 * 0.7  # No activity, umbrella, sunny
    p_xyz[1, 0, 0] = 0.8 * 0.7   # Activity, no umbrella, sunny
    p_xyz[1, 1, 0] = 0.05 * 0.7  # Activity, umbrella, sunny
    
    # Z=1 (rainy)
    p_xyz[0, 0, 1] = 0.6 * 0.3   # No activity, no umbrella, rainy
    p_xyz[0, 1, 1] = 0.25 * 0.3  # No activity, umbrella, rainy
    p_xyz[1, 0, 1] = 0.05 * 0.3  # Activity, no umbrella, rainy
    p_xyz[1, 1, 1] = 0.1 * 0.3   # Activity, umbrella, rainy
    
    def mutual_information(p_xy):
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        mi = 0
        for i in range(p_xy.shape[0]):
            for j in range(p_xy.shape[1]):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        return mi
    
    # Marginal P(X, Y)
    p_xy = p_xyz.sum(axis=2)
    
    # I(X; Y)
    mi_xy = mutual_information(p_xy)
    
    # I(X; Y | Z) = sum_z P(z) I(X; Y | Z=z)
    p_z = p_xyz.sum(axis=(0, 1))
    
    mi_xy_given_z = 0
    for z in range(2):
        if p_z[z] > 0:
            p_xy_given_z = p_xyz[:, :, z] / p_z[z]
            mi_given_z = mutual_information(p_xy_given_z)
            mi_xy_given_z += p_z[z] * mi_given_z
    
    print("X: Outdoor activity, Y: Umbrella, Z: Weather")
    print(f"\nI(X; Y) = {mi_xy:.4f}")
    print(f"I(X; Y | Z) = {mi_xy_given_z:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  Unconditional MI: {mi_xy:.4f}")
    print(f"  Conditional MI (knowing weather): {mi_xy_given_z:.4f}")
    print(f"  Difference: {mi_xy - mi_xy_given_z:.4f}")
    print("\nWeather explains some of the dependence between X and Y")


def example_data_processing():
    """Data processing inequality."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Data Processing Inequality")
    print("=" * 60)
    
    print("Data Processing Inequality:")
    print("For Markov chain X → Y → Z:")
    print("I(X; Z) ≤ I(X; Y)")
    print("I(X; Z) ≤ I(Y; Z)")
    print("\n'Processing cannot create information'")
    
    np.random.seed(42)
    n = 5000
    
    # X: Original signal
    X = np.random.randn(n)
    
    # Y: Noisy version of X
    Y = X + 0.3 * np.random.randn(n)
    
    # Z: Processed version (more noise)
    Z = Y + 0.5 * np.random.randn(n)
    
    def estimate_mi_continuous(A, B, bins=20):
        """Estimate MI between continuous variables."""
        hist_2d, _, _ = np.histogram2d(A, B, bins=bins)
        p_xy = hist_2d / hist_2d.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        return mi
    
    mi_xy = estimate_mi_continuous(X, Y)
    mi_yz = estimate_mi_continuous(Y, Z)
    mi_xz = estimate_mi_continuous(X, Z)
    
    print(f"\nX → Y → Z (Markov chain with added noise)")
    print(f"\nMutual Information:")
    print(f"  I(X; Y) = {mi_xy:.4f}")
    print(f"  I(Y; Z) = {mi_yz:.4f}")
    print(f"  I(X; Z) = {mi_xz:.4f}")
    
    print(f"\nVerification:")
    print(f"  I(X; Z) ≤ I(X; Y)? {mi_xz:.4f} ≤ {mi_xy:.4f}: {mi_xz <= mi_xy + 0.01}")
    print(f"  I(X; Z) ≤ I(Y; Z)? {mi_xz:.4f} ≤ {mi_yz:.4f}: {mi_xz <= mi_yz + 0.01}")


def example_normalized_mi():
    """Normalized mutual information for clustering."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Normalized Mutual Information")
    print("=" * 60)
    
    def entropy(labels):
        """Compute entropy of label distribution."""
        counts = Counter(labels)
        n = len(labels)
        probs = np.array(list(counts.values())) / n
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def mutual_info_labels(labels1, labels2):
        """Compute MI between two labelings."""
        joint = Counter(zip(labels1, labels2))
        n = len(labels1)
        
        counts1 = Counter(labels1)
        counts2 = Counter(labels2)
        
        mi = 0
        for (l1, l2), count in joint.items():
            p_joint = count / n
            p1 = counts1[l1] / n
            p2 = counts2[l2] / n
            if p_joint > 0:
                mi += p_joint * np.log(p_joint / (p1 * p2))
        
        return mi
    
    def nmi(labels1, labels2):
        """Normalized Mutual Information."""
        mi = mutual_info_labels(labels1, labels2)
        h1 = entropy(labels1)
        h2 = entropy(labels2)
        return 2 * mi / (h1 + h2 + 1e-10)
    
    # True labels
    true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    
    # Different clusterings
    perfect = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    relabeled = np.array([1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0])  # Same but relabeled
    partial = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])    # One error
    bad = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])        # Random
    
    print("NMI(Clustering1, Clustering2)")
    print("Range: [0, 1], 1 = perfect agreement")
    print("\nTrue labels:", list(true_labels))
    
    clusterings = [
        ("Perfect match", perfect),
        ("Relabeled (same structure)", relabeled),
        ("One error", partial),
        ("Bad clustering", bad),
    ]
    
    print(f"\n{'Clustering':>30} {'NMI':>10}")
    print("-" * 45)
    
    for name, pred in clusterings:
        nmi_score = nmi(true_labels, pred)
        print(f"{name:>30} {nmi_score:>10.4f}")
    
    print("\nNMI is label-invariant: relabeling doesn't change score")


def example_knn_mi_estimation():
    """KNN-based MI estimation (KSG estimator)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: KNN MI Estimation (KSG)")
    print("=" * 60)
    
    def ksg_mi(X, Y, k=3):
        """
        Kraskov-Stögbauer-Grassberger MI estimator.
        Simplified implementation.
        """
        n = len(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        
        # Combined space
        XY = np.hstack([X, Y])
        
        # For each point, find distance to k-th neighbor in joint space
        from scipy.spatial import KDTree
        
        tree_xy = KDTree(XY)
        tree_x = KDTree(X)
        tree_y = KDTree(Y)
        
        mi_sum = 0
        for i in range(n):
            # Distance to k-th neighbor in joint space
            dists_xy, _ = tree_xy.query(XY[i], k=k+1)
            eps = dists_xy[-1]  # Distance to k-th neighbor
            
            # Count points within eps in X and Y spaces
            n_x = len(tree_x.query_ball_point(X[i], eps)) - 1
            n_y = len(tree_y.query_ball_point(Y[i], eps)) - 1
            
            mi_sum += digamma(n_x + 1) + digamma(n_y + 1)
        
        mi = digamma(k) - mi_sum / n + digamma(n)
        return mi
    
    np.random.seed(42)
    
    # Correlated Gaussian
    rho = 0.7
    n = 500
    
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    samples = np.random.multivariate_normal(mean, cov, n)
    X, Y = samples[:, 0], samples[:, 1]
    
    true_mi = -0.5 * np.log(1 - rho**2)
    ksg_mi_est = ksg_mi(X, Y, k=5)
    
    print("KSG (Kraskov-Stögbauer-Grassberger) Estimator")
    print("Uses k-nearest neighbors for density estimation")
    
    print(f"\nBivariate Gaussian with ρ = {rho}")
    print(f"  True MI: {true_mi:.4f}")
    print(f"  KSG estimate (k=5): {ksg_mi_est:.4f}")
    
    # Try different k values
    print(f"\nEffect of k:")
    print(f"{'k':>5} {'Estimate':>12}")
    print("-" * 20)
    
    for k in [1, 3, 5, 10, 20]:
        est = ksg_mi(X, Y, k=k)
        print(f"{k:>5} {est:>12.4f}")


def example_info_bottleneck():
    """Information bottleneck principle."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Information Bottleneck")
    print("=" * 60)
    
    print("Information Bottleneck:")
    print("Find representation Z that:")
    print("  1. Compresses X: minimize I(X; Z)")
    print("  2. Preserves info about Y: maximize I(Z; Y)")
    print("\nObjective: max I(Z; Y) - β I(X; Z)")
    
    np.random.seed(42)
    n = 1000
    
    # X: 10D input with 2 relevant dimensions
    X = np.random.randn(n, 10)
    
    # Y: depends only on first 2 dimensions
    Y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    def estimate_mi(A, B, bins=10):
        """Estimate MI."""
        if A.ndim > 1:
            # Use PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            A = PCA(n_components=1).fit_transform(A).flatten()
        if B.ndim > 1:
            B = PCA(n_components=1).fit_transform(B).flatten()
        
        hist, _, _ = np.histogram2d(A, B, bins=bins)
        p_xy = hist / hist.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        return mi
    
    # Different representations
    Z_all = X  # Keep all
    Z_relevant = X[:, :2]  # Keep relevant
    Z_irrelevant = X[:, 2:]  # Keep irrelevant
    
    print("\nRepresentation comparison:")
    print(f"{'Representation':>20} {'I(X;Z)':>12} {'I(Z;Y)':>12}")
    print("-" * 50)
    
    for name, Z in [('All dims', Z_all), ('First 2 (relevant)', Z_relevant), 
                     ('Last 8 (irrelevant)', Z_irrelevant)]:
        i_xz = estimate_mi(X, Z)
        i_zy = estimate_mi(Z.reshape(-1, 1) if Z.ndim == 1 else Z[:, 0], Y)
        print(f"{name:>20} {i_xz:>12.4f} {i_zy:>12.4f}")
    
    print("\nOptimal: keep relevant dimensions, discard irrelevant")


def example_infonce():
    """InfoNCE contrastive learning objective."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: InfoNCE Objective")
    print("=" * 60)
    
    print("InfoNCE: Contrastive MI estimation")
    print("I(X; Y) ≥ log(N) - L_NCE")
    print("\nL_NCE = -E[log(exp(f(x,y_+)) / sum(exp(f(x,y))))]")
    
    np.random.seed(42)
    
    # Simulated embeddings
    batch_size = 64
    dim = 32
    
    # X and Y are correlated (positive pairs)
    X = np.random.randn(batch_size, dim)
    noise = 0.5 * np.random.randn(batch_size, dim)
    Y_positive = X + noise  # Positive pairs
    
    def similarity(a, b):
        """Cosine similarity."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return a_norm @ b_norm.T
    
    def info_nce_loss(X, Y, temperature=0.5):
        """Compute InfoNCE loss."""
        # Similarity matrix
        sim = similarity(X, Y) / temperature
        
        # For each x_i, y_i is positive, others are negative
        # Loss = -log(exp(sim(x_i, y_i)) / sum_j(exp(sim(x_i, y_j))))
        
        n = len(X)
        losses = []
        
        for i in range(n):
            # Numerator: positive pair
            numerator = np.exp(sim[i, i])
            # Denominator: all pairs
            denominator = np.sum(np.exp(sim[i, :]))
            
            loss_i = -np.log(numerator / denominator)
            losses.append(loss_i)
        
        return np.mean(losses)
    
    # Compute loss for correlated pairs
    loss_correlated = info_nce_loss(X, Y_positive)
    
    # Random pairs (no correlation)
    Y_random = np.random.randn(batch_size, dim)
    loss_random = info_nce_loss(X, Y_random)
    
    # MI lower bound
    mi_bound_correlated = np.log(batch_size) - loss_correlated
    mi_bound_random = np.log(batch_size) - loss_random
    
    print(f"\nBatch size N = {batch_size}")
    print(f"\nCorrelated pairs (X, X+noise):")
    print(f"  InfoNCE loss: {loss_correlated:.4f}")
    print(f"  MI lower bound: {mi_bound_correlated:.4f}")
    
    print(f"\nRandom pairs:")
    print(f"  InfoNCE loss: {loss_random:.4f}")
    print(f"  MI lower bound: {mi_bound_random:.4f}")
    
    print(f"\nMaximum possible bound: log(N) = {np.log(batch_size):.4f}")


if __name__ == "__main__":
    example_basic_mi()
    example_mi_entropy_relations()
    example_gaussian_mi()
    example_mi_kl_relation()
    example_feature_selection()
    example_pointwise_mi()
    example_conditional_mi()
    example_data_processing()
    example_normalized_mi()
    example_knn_mi_estimation()
    example_info_bottleneck()
    example_infonce()
