"""
Loss Functions - Examples
=========================

Implementations demonstrating various loss functions
for machine learning with their properties and applications.
"""

import numpy as np
from typing import Tuple, Callable, Optional


# =============================================================================
# Example 1: Regression Losses
# =============================================================================

def example_1_regression_losses():
    """
    Implement and compare common regression loss functions.
    """
    print("Example 1: Regression Losses")
    print("=" * 60)
    
    def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean Squared Error: (1/n) Σ(ŷ - y)²"""
        return np.mean((y_pred - y_true) ** 2)
    
    def mse_gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Gradient of MSE w.r.t. predictions."""
        return 2 * (y_pred - y_true) / len(y_true)
    
    def mae_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean Absolute Error: (1/n) Σ|ŷ - y|"""
        return np.mean(np.abs(y_pred - y_true))
    
    def mae_subgradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Subgradient of MAE."""
        return np.sign(y_pred - y_true) / len(y_true)
    
    def huber_loss(y_pred: np.ndarray, y_true: np.ndarray, 
                  delta: float = 1.0) -> float:
        """
        Huber loss: quadratic for small errors, linear for large.
        """
        residual = y_pred - y_true
        abs_residual = np.abs(residual)
        
        loss = np.where(
            abs_residual <= delta,
            0.5 * residual ** 2,
            delta * abs_residual - 0.5 * delta ** 2
        )
        return np.mean(loss)
    
    def huber_gradient(y_pred: np.ndarray, y_true: np.ndarray,
                      delta: float = 1.0) -> np.ndarray:
        """Gradient of Huber loss."""
        residual = y_pred - y_true
        return np.where(
            np.abs(residual) <= delta,
            residual,
            delta * np.sign(residual)
        ) / len(y_true)
    
    def log_cosh_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Log-cosh loss: smooth approximation of MAE."""
        return np.mean(np.log(np.cosh(y_pred - y_true)))
    
    def quantile_loss(y_pred: np.ndarray, y_true: np.ndarray,
                     tau: float = 0.5) -> float:
        """Quantile loss for predicting specific quantiles."""
        residual = y_true - y_pred
        return np.mean(np.where(residual >= 0, tau * residual, (tau - 1) * residual))
    
    # Test data with outliers
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n)
    y_pred = y_true + 0.1 * np.random.randn(n)
    
    # Add outliers
    outlier_idx = [0, 1, 2]
    y_pred[outlier_idx] += np.array([10, -8, 12])
    
    print("Comparing losses with outliers:\n")
    print(f"{'Loss':<20} {'Value':<15} {'Gradient Norm':<15}")
    print("-" * 50)
    
    losses = [
        ('MSE', mse_loss(y_pred, y_true), np.linalg.norm(mse_gradient(y_pred, y_true))),
        ('MAE', mae_loss(y_pred, y_true), np.linalg.norm(mae_subgradient(y_pred, y_true))),
        ('Huber (δ=1)', huber_loss(y_pred, y_true, 1.0), np.linalg.norm(huber_gradient(y_pred, y_true, 1.0))),
        ('Log-Cosh', log_cosh_loss(y_pred, y_true), None),
        ('Quantile (τ=0.5)', quantile_loss(y_pred, y_true, 0.5), None),
    ]
    
    for name, value, grad_norm in losses:
        grad_str = f"{grad_norm:.6f}" if grad_norm is not None else "N/A"
        print(f"{name:<20} {value:<15.6f} {grad_str:<15}")
    
    # Without outliers
    y_pred_clean = y_true + 0.1 * np.random.randn(n)
    
    print("\nWithout outliers:")
    print(f"MSE: {mse_loss(y_pred_clean, y_true):.6f}")
    print(f"MAE: {mae_loss(y_pred_clean, y_true):.6f}")
    
    return mse_loss, mae_loss, huber_loss


# =============================================================================
# Example 2: Classification Losses
# =============================================================================

def example_2_classification_losses():
    """
    Implement binary and multi-class classification losses.
    """
    print("\nExample 2: Classification Losses")
    print("=" * 60)
    
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def softmax(z: np.ndarray) -> np.ndarray:
        """Softmax activation (numerically stable)."""
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray,
                            epsilon: float = 1e-7) -> float:
        """
        Binary cross-entropy: -[y log(p) + (1-y) log(1-p)]
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def bce_gradient(y_pred: np.ndarray, y_true: np.ndarray,
                    epsilon: float = 1e-7) -> np.ndarray:
        """Gradient of BCE w.r.t. predictions."""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / len(y_true)
    
    def bce_with_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
        """BCE with logits (numerically stable)."""
        # log(1 + exp(-z)) = max(0, -z) + log(1 + exp(-|z|))
        return np.mean(
            np.maximum(logits, 0) - logits * y_true + np.log(1 + np.exp(-np.abs(logits)))
        )
    
    def categorical_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray,
                                  epsilon: float = 1e-7) -> float:
        """
        Categorical cross-entropy for multi-class.
        y_true: one-hot encoded or class indices
        y_pred: probabilities after softmax
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if y_true.ndim == 1:
            # Class indices
            n = len(y_true)
            return -np.sum(np.log(y_pred[np.arange(n), y_true])) / n
        else:
            # One-hot
            return -np.sum(y_true * np.log(y_pred)) / len(y_true)
    
    def hinge_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Hinge loss: max(0, 1 - y * ŷ)
        y_true in {-1, +1}
        """
        return np.mean(np.maximum(0, 1 - y_true * y_pred))
    
    def squared_hinge_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Squared hinge loss."""
        return np.mean(np.maximum(0, 1 - y_true * y_pred) ** 2)
    
    # Binary classification example
    np.random.seed(42)
    n = 100
    
    # Logits
    logits = np.random.randn(n)
    y_true_binary = (np.random.rand(n) > 0.5).astype(float)
    y_pred_prob = sigmoid(logits)
    
    print("Binary Classification:\n")
    print(f"BCE (from probs): {binary_cross_entropy(y_pred_prob, y_true_binary):.6f}")
    print(f"BCE (from logits): {bce_with_logits(logits, y_true_binary):.6f}")
    
    # Multi-class
    K = 5
    logits_multi = np.random.randn(n, K)
    y_true_multi = np.random.randint(0, K, n)
    y_pred_multi = softmax(logits_multi)
    
    print(f"\nMulti-class CE: {categorical_cross_entropy(y_pred_multi, y_true_multi):.6f}")
    
    # SVM-style
    y_true_svm = 2 * (np.random.rand(n) > 0.5).astype(float) - 1  # {-1, +1}
    scores = np.random.randn(n)
    
    print(f"\nHinge Loss: {hinge_loss(scores, y_true_svm):.6f}")
    print(f"Squared Hinge: {squared_hinge_loss(scores, y_true_svm):.6f}")
    
    return binary_cross_entropy, categorical_cross_entropy, hinge_loss


# =============================================================================
# Example 3: Focal Loss for Class Imbalance
# =============================================================================

def example_3_focal_loss():
    """
    Implement focal loss for handling class imbalance.
    """
    print("\nExample 3: Focal Loss for Class Imbalance")
    print("=" * 60)
    
    def focal_loss(y_pred: np.ndarray, y_true: np.ndarray,
                  gamma: float = 2.0, alpha: float = 0.25,
                  epsilon: float = 1e-7) -> float:
        """
        Focal loss: -α(1-p)^γ log(p) for positive class
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels (0 or 1)
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weight for positive class
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # For positive examples: -α(1-p)^γ log(p)
        # For negative examples: -(1-α)(p)^γ log(1-p)
        
        positive_loss = -alpha * ((1 - y_pred) ** gamma) * np.log(y_pred) * y_true
        negative_loss = -(1 - alpha) * (y_pred ** gamma) * np.log(1 - y_pred) * (1 - y_true)
        
        return np.mean(positive_loss + negative_loss)
    
    def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray,
                            epsilon: float = 1e-7) -> float:
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Create imbalanced dataset
    np.random.seed(42)
    n_pos = 50
    n_neg = 950
    
    # Predictions: assume model tends to predict negative
    y_pred_pos = 0.6 + 0.3 * np.random.rand(n_pos)  # Harder for positive
    y_pred_neg = 0.1 + 0.2 * np.random.rand(n_neg)  # Easy for negative
    
    y_pred = np.concatenate([y_pred_pos, y_pred_neg])
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    
    print(f"Class imbalance: {n_pos} positive, {n_neg} negative")
    
    print("\nLoss comparison:\n")
    print(f"{'Loss':<25} {'Value':<15}")
    print("-" * 40)
    
    print(f"{'BCE':<25} {binary_cross_entropy(y_pred, y_true):<15.6f}")
    
    for gamma in [0, 0.5, 1, 2, 5]:
        loss = focal_loss(y_pred, y_true, gamma=gamma)
        print(f"{'Focal (γ=' + str(gamma) + ')':<25} {loss:<15.6f}")
    
    # Analyze per-sample losses
    print("\nPer-sample loss analysis:")
    
    # Easy negative (low pred, true negative)
    easy_neg = focal_loss(np.array([0.1]), np.array([0.0]), gamma=2)
    
    # Hard negative (high pred, true negative)  
    hard_neg = focal_loss(np.array([0.9]), np.array([0.0]), gamma=2)
    
    # Easy positive (high pred, true positive)
    easy_pos = focal_loss(np.array([0.9]), np.array([1.0]), gamma=2)
    
    # Hard positive (low pred, true positive)
    hard_pos = focal_loss(np.array([0.1]), np.array([1.0]), gamma=2)
    
    print(f"  Easy negative (p=0.1, y=0): {easy_neg:.6f}")
    print(f"  Hard negative (p=0.9, y=0): {hard_neg:.6f}")
    print(f"  Easy positive (p=0.9, y=1): {easy_pos:.6f}")
    print(f"  Hard positive (p=0.1, y=1): {hard_pos:.6f}")
    
    return focal_loss


# =============================================================================
# Example 4: KL Divergence and ELBO
# =============================================================================

def example_4_kl_divergence():
    """
    Implement KL divergence and ELBO for variational inference.
    """
    print("\nExample 4: KL Divergence and ELBO")
    print("=" * 60)
    
    def kl_divergence_discrete(p: np.ndarray, q: np.ndarray,
                               epsilon: float = 1e-10) -> float:
        """
        KL(p||q) = Σ p(x) log(p(x)/q(x))
        """
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        return np.sum(p * np.log(p / q))
    
    def kl_divergence_gaussian(mu_p: np.ndarray, sigma_p: np.ndarray,
                               mu_q: np.ndarray, sigma_q: np.ndarray) -> float:
        """
        KL divergence between two diagonal Gaussians.
        KL(p||q) = 0.5 * Σ[log(σq²/σp²) + (σp² + (μp-μq)²)/σq² - 1]
        """
        var_p = sigma_p ** 2
        var_q = sigma_q ** 2
        
        return 0.5 * np.sum(
            np.log(var_q / var_p) + 
            (var_p + (mu_p - mu_q) ** 2) / var_q - 1
        )
    
    def kl_to_standard_normal(mu: np.ndarray, log_var: np.ndarray) -> float:
        """
        KL(N(μ, σ²) || N(0, 1))
        = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        """
        return -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))
    
    def elbo_loss(x: np.ndarray, x_recon: np.ndarray, 
                 mu: np.ndarray, log_var: np.ndarray,
                 beta: float = 1.0) -> Tuple[float, float, float]:
        """
        Evidence Lower Bound (ELBO) for VAE.
        
        ELBO = E[log p(x|z)] - β * KL(q(z|x)||p(z))
        Loss = -ELBO = reconstruction + β * KL
        
        Returns (total_loss, recon_loss, kl_loss)
        """
        # Reconstruction loss (assuming Gaussian likelihood, i.e., MSE)
        recon_loss = np.mean((x - x_recon) ** 2)
        
        # KL divergence to standard normal prior
        kl_loss = kl_to_standard_normal(mu, log_var) / len(x)
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    # Discrete distributions
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    q_similar = np.array([0.35, 0.32, 0.22, 0.11])
    
    print("Discrete KL Divergence:\n")
    print(f"KL(p || uniform): {kl_divergence_discrete(p, q_uniform):.6f}")
    print(f"KL(p || similar): {kl_divergence_discrete(p, q_similar):.6f}")
    print(f"KL(uniform || p): {kl_divergence_discrete(q_uniform, p):.6f}")
    print("Note: KL is asymmetric!")
    
    # Gaussian KL
    print("\nGaussian KL Divergence:\n")
    mu_p = np.array([0.0, 1.0])
    sigma_p = np.array([1.0, 0.5])
    mu_q = np.array([0.0, 0.0])
    sigma_q = np.array([1.0, 1.0])
    
    print(f"KL(N(μp,σp) || N(μq,σq)): {kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):.6f}")
    print(f"KL to standard normal: {kl_to_standard_normal(mu_p, 2*np.log(sigma_p)):.6f}")
    
    # VAE ELBO
    print("\nVAE ELBO Loss:\n")
    np.random.seed(42)
    x = np.random.randn(100, 10)
    x_recon = x + 0.1 * np.random.randn(100, 10)  # Noisy reconstruction
    mu = 0.5 * np.random.randn(100, 5)
    log_var = -0.5 + 0.1 * np.random.randn(100, 5)
    
    for beta in [0.0, 0.5, 1.0, 2.0, 4.0]:
        total, recon, kl = elbo_loss(x, x_recon, mu, log_var, beta)
        print(f"β={beta:.1f}: Total={total:.4f}, Recon={recon:.4f}, KL={kl:.4f}")
    
    return kl_divergence_gaussian, elbo_loss


# =============================================================================
# Example 5: Contrastive and Triplet Losses
# =============================================================================

def example_5_contrastive_losses():
    """
    Implement contrastive and triplet losses for representation learning.
    """
    print("\nExample 5: Contrastive and Triplet Losses")
    print("=" * 60)
    
    def contrastive_loss(embeddings1: np.ndarray, embeddings2: np.ndarray,
                        labels: np.ndarray, margin: float = 1.0) -> float:
        """
        Contrastive loss for pairs.
        labels: 1 if similar, 0 if dissimilar
        
        L = y * D² + (1-y) * max(0, margin - D)²
        """
        distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
        
        positive_loss = labels * distances ** 2
        negative_loss = (1 - labels) * np.maximum(0, margin - distances) ** 2
        
        return np.mean(positive_loss + negative_loss)
    
    def triplet_loss(anchors: np.ndarray, positives: np.ndarray,
                    negatives: np.ndarray, margin: float = 1.0) -> float:
        """
        Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        """
        dist_pos = np.linalg.norm(anchors - positives, axis=1)
        dist_neg = np.linalg.norm(anchors - negatives, axis=1)
        
        losses = np.maximum(0, dist_pos - dist_neg + margin)
        return np.mean(losses)
    
    def triplet_loss_analysis(anchors: np.ndarray, positives: np.ndarray,
                             negatives: np.ndarray, margin: float = 1.0) -> dict:
        """Analyze triplet loss components."""
        dist_pos = np.linalg.norm(anchors - positives, axis=1)
        dist_neg = np.linalg.norm(anchors - negatives, axis=1)
        
        losses = np.maximum(0, dist_pos - dist_neg + margin)
        
        # Triplet categories
        easy = np.sum(dist_neg > dist_pos + margin)  # Loss = 0
        semi_hard = np.sum((dist_pos < dist_neg) & (dist_neg < dist_pos + margin))
        hard = np.sum(dist_neg < dist_pos)
        
        return {
            'loss': np.mean(losses),
            'mean_pos_dist': np.mean(dist_pos),
            'mean_neg_dist': np.mean(dist_neg),
            'easy_triplets': easy,
            'semi_hard_triplets': semi_hard,
            'hard_triplets': hard
        }
    
    def info_nce_loss(query: np.ndarray, positive: np.ndarray,
                     negatives: np.ndarray, temperature: float = 0.07) -> float:
        """
        InfoNCE loss (used in SimCLR, CLIP).
        
        L = -log(exp(q·k+/τ) / Σexp(q·k/τ))
        """
        # Similarity with positive
        pos_sim = np.dot(query, positive) / temperature
        
        # Similarities with negatives
        neg_sims = negatives @ query / temperature
        
        # Log-sum-exp for numerical stability
        max_sim = max(pos_sim, np.max(neg_sims))
        
        log_sum_exp = max_sim + np.log(
            np.exp(pos_sim - max_sim) + np.sum(np.exp(neg_sims - max_sim))
        )
        
        return -pos_sim + log_sum_exp
    
    # Generate embeddings
    np.random.seed(42)
    n = 100
    dim = 64
    
    # Embeddings (normalized)
    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
    
    anchors = normalize(np.random.randn(n, dim))
    positives = normalize(anchors + 0.3 * np.random.randn(n, dim))  # Similar
    negatives = normalize(np.random.randn(n, dim))  # Random
    
    print("Triplet Loss Analysis:\n")
    for margin in [0.5, 1.0, 2.0]:
        analysis = triplet_loss_analysis(anchors, positives, negatives, margin)
        print(f"Margin = {margin}:")
        print(f"  Loss: {analysis['loss']:.4f}")
        print(f"  Mean pos/neg dist: {analysis['mean_pos_dist']:.4f} / {analysis['mean_neg_dist']:.4f}")
        print(f"  Easy/Semi-hard/Hard: {analysis['easy_triplets']}/{analysis['semi_hard_triplets']}/{analysis['hard_triplets']}")
    
    # Contrastive loss
    print("\nContrastive Loss:")
    labels = np.array([1] * 50 + [0] * 50)  # 50 similar, 50 dissimilar
    e1 = anchors
    e2 = np.vstack([positives[:50], negatives[50:]])
    
    for margin in [0.5, 1.0, 2.0]:
        loss = contrastive_loss(e1, e2, labels, margin)
        print(f"  Margin = {margin}: {loss:.4f}")
    
    # InfoNCE
    print("\nInfoNCE Loss:")
    query = normalize(np.random.randn(dim))
    positive = normalize(query + 0.2 * np.random.randn(dim))
    negatives = normalize(np.random.randn(100, dim))
    
    for temp in [0.05, 0.1, 0.5, 1.0]:
        loss = info_nce_loss(query, positive, negatives, temp)
        print(f"  τ = {temp}: {loss:.4f}")
    
    return triplet_loss, contrastive_loss, info_nce_loss


# =============================================================================
# Example 6: Ranking Losses
# =============================================================================

def example_6_ranking_losses():
    """
    Implement losses for learning-to-rank.
    """
    print("\nExample 6: Ranking Losses")
    print("=" * 60)
    
    def pairwise_hinge_loss(scores: np.ndarray, relevance: np.ndarray,
                           margin: float = 1.0) -> float:
        """
        Pairwise hinge loss for ranking.
        Higher relevance should have higher score.
        """
        n = len(scores)
        total_loss = 0
        n_pairs = 0
        
        for i in range(n):
            for j in range(n):
                if relevance[i] > relevance[j]:
                    # i should rank higher than j
                    loss = max(0, margin - scores[i] + scores[j])
                    total_loss += loss
                    n_pairs += 1
        
        return total_loss / max(n_pairs, 1)
    
    def listwise_softmax_loss(scores: np.ndarray, relevance: np.ndarray) -> float:
        """
        ListNet-style softmax cross-entropy for ranking.
        """
        # Softmax over scores
        scores_shifted = scores - np.max(scores)
        pred_probs = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted))
        
        # Target distribution from relevance
        rel_shifted = relevance - np.max(relevance)
        target_probs = np.exp(rel_shifted) / np.sum(np.exp(rel_shifted))
        
        # Cross-entropy
        return -np.sum(target_probs * np.log(pred_probs + 1e-10))
    
    def ndcg_loss(scores: np.ndarray, relevance: np.ndarray, k: int = None) -> float:
        """
        Compute NDCG (normalized discounted cumulative gain).
        Returns negative for use as loss (maximize NDCG = minimize -NDCG).
        """
        if k is None:
            k = len(scores)
        
        # Get ranking by predicted scores
        pred_ranking = np.argsort(-scores)
        
        # DCG
        dcg = 0
        for i in range(min(k, len(scores))):
            dcg += (2 ** relevance[pred_ranking[i]] - 1) / np.log2(i + 2)
        
        # Ideal DCG
        ideal_ranking = np.argsort(-relevance)
        idcg = 0
        for i in range(min(k, len(scores))):
            idcg += (2 ** relevance[ideal_ranking[i]] - 1) / np.log2(i + 2)
        
        ndcg = dcg / max(idcg, 1e-10)
        
        return -ndcg  # Negative for minimization
    
    def lambda_rank_gradients(scores: np.ndarray, relevance: np.ndarray,
                             sigma: float = 1.0) -> np.ndarray:
        """
        LambdaRank gradients for pairwise learning-to-rank.
        """
        n = len(scores)
        gradients = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if relevance[i] != relevance[j]:
                    # Probability that i is ranked above j
                    p_ij = 1 / (1 + np.exp(-sigma * (scores[i] - scores[j])))
                    
                    # Label: 1 if i should be ranked above j
                    S_ij = 1 if relevance[i] > relevance[j] else 0
                    
                    # Lambda gradient
                    lambda_ij = sigma * (S_ij - p_ij)
                    
                    gradients[i] += lambda_ij
                    gradients[j] -= lambda_ij
        
        return gradients
    
    # Example: search results ranking
    np.random.seed(42)
    n = 10
    
    # Relevance scores (0-4 scale)
    relevance = np.array([4, 3, 3, 2, 2, 1, 1, 0, 0, 0])
    
    # Model predictions (scores)
    good_scores = relevance + 0.5 * np.random.randn(n)  # Correlated with relevance
    bad_scores = np.random.randn(n)  # Random
    
    print("Ranking Loss Comparison:\n")
    print(f"{'Score Type':<15} {'Pairwise':<12} {'Listwise':<12} {'NDCG Loss':<12}")
    print("-" * 50)
    
    for name, scores in [('Good', good_scores), ('Bad', bad_scores)]:
        pw = pairwise_hinge_loss(scores, relevance)
        lw = listwise_softmax_loss(scores, relevance)
        ndcg = ndcg_loss(scores, relevance)
        
        print(f"{name:<15} {pw:<12.4f} {lw:<12.4f} {ndcg:<12.4f}")
    
    print("\nLambdaRank Gradients (good scores):")
    grads = lambda_rank_gradients(good_scores, relevance)
    for i in range(5):
        print(f"  Item {i} (rel={relevance[i]}): gradient = {grads[i]:.4f}")
    
    return pairwise_hinge_loss, listwise_softmax_loss


# =============================================================================
# Example 7: Probabilistic Losses
# =============================================================================

def example_7_probabilistic_losses():
    """
    Implement probabilistic loss functions for uncertainty estimation.
    """
    print("\nExample 7: Probabilistic Losses")
    print("=" * 60)
    
    def gaussian_nll(y_pred_mean: np.ndarray, y_pred_var: np.ndarray,
                    y_true: np.ndarray) -> float:
        """
        Negative log-likelihood for heteroscedastic Gaussian.
        
        NLL = 0.5 * log(2πσ²) + (y - μ)² / (2σ²)
        """
        return np.mean(
            0.5 * np.log(2 * np.pi * y_pred_var) + 
            (y_true - y_pred_mean) ** 2 / (2 * y_pred_var)
        )
    
    def laplace_nll(y_pred_loc: np.ndarray, y_pred_scale: np.ndarray,
                   y_true: np.ndarray) -> float:
        """
        Negative log-likelihood for Laplace distribution.
        
        NLL = log(2b) + |y - μ| / b
        """
        return np.mean(
            np.log(2 * y_pred_scale) + 
            np.abs(y_true - y_pred_loc) / y_pred_scale
        )
    
    def mixture_density_nll(y_pred_weights: np.ndarray, y_pred_means: np.ndarray,
                           y_pred_vars: np.ndarray, y_true: np.ndarray) -> float:
        """
        NLL for Mixture Density Network (MDN).
        
        p(y|x) = Σ_k π_k N(y; μ_k, σ_k²)
        """
        n, k = y_pred_weights.shape
        
        # Compute Gaussian components
        components = np.zeros((n, k))
        for j in range(k):
            components[:, j] = (
                y_pred_weights[:, j] / np.sqrt(2 * np.pi * y_pred_vars[:, j]) *
                np.exp(-(y_true - y_pred_means[:, j]) ** 2 / (2 * y_pred_vars[:, j]))
            )
        
        # Mixture probability
        mixture_prob = np.sum(components, axis=1)
        
        return -np.mean(np.log(mixture_prob + 1e-10))
    
    def calibration_loss(predictions: np.ndarray, true_probs: np.ndarray,
                        n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE).
        
        ECE = Σ_b (|B_b|/n) |acc(B_b) - conf(B_b)|
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i+1])
            if np.sum(mask) > 0:
                avg_confidence = np.mean(predictions[mask])
                avg_accuracy = np.mean(true_probs[mask])
                ece += np.sum(mask) / len(predictions) * abs(avg_accuracy - avg_confidence)
        
        return ece
    
    # Heteroscedastic regression
    np.random.seed(42)
    n = 100
    
    y_true = np.random.randn(n)
    
    # Model with uncertainty estimation
    y_pred_mean = y_true + 0.1 * np.random.randn(n)
    y_pred_var_good = 0.1 + 0.05 * np.abs(y_true)  # Varies with target
    y_pred_var_constant = np.ones(n) * 0.15  # Constant variance
    
    print("Gaussian NLL (heteroscedastic vs homoscedastic):\n")
    print(f"Variable variance: {gaussian_nll(y_pred_mean, y_pred_var_good, y_true):.4f}")
    print(f"Constant variance: {gaussian_nll(y_pred_mean, y_pred_var_constant, y_true):.4f}")
    
    # Compare Gaussian vs Laplace for outliers
    print("\nGaussian vs Laplace NLL:")
    
    # Add outliers
    y_true_outliers = y_true.copy()
    y_true_outliers[:5] += 5 * np.random.randn(5)
    
    gaussian = gaussian_nll(y_pred_mean, y_pred_var_constant, y_true_outliers)
    laplace = laplace_nll(y_pred_mean, np.sqrt(y_pred_var_constant / 2), y_true_outliers)
    
    print(f"With outliers - Gaussian: {gaussian:.4f}, Laplace: {laplace:.4f}")
    
    # Calibration
    print("\nCalibration Analysis:")
    probs_calibrated = np.random.rand(1000)
    true_probs_calibrated = (np.random.rand(1000) < probs_calibrated).astype(float)
    
    probs_overconfident = 0.5 + 0.4 * np.random.rand(1000)  # Always confident
    true_probs_overconfident = (np.random.rand(1000) < 0.5).astype(float)
    
    ece_calibrated = calibration_loss(probs_calibrated, true_probs_calibrated)
    ece_overconfident = calibration_loss(probs_overconfident, true_probs_overconfident)
    
    print(f"Calibrated model ECE: {ece_calibrated:.4f}")
    print(f"Overconfident model ECE: {ece_overconfident:.4f}")
    
    return gaussian_nll, laplace_nll


# =============================================================================
# Example 8: Multi-Task Learning Losses
# =============================================================================

def example_8_multitask_losses():
    """
    Implement multi-task learning loss combination strategies.
    """
    print("\nExample 8: Multi-Task Learning Losses")
    print("=" * 60)
    
    def uniform_weighted_loss(losses: list) -> float:
        """Simple average of losses."""
        return np.mean(losses)
    
    def fixed_weighted_loss(losses: list, weights: list) -> float:
        """Fixed weight combination."""
        return sum(w * l for w, l in zip(weights, losses))
    
    def uncertainty_weighted_loss(losses: list, log_vars: np.ndarray) -> float:
        """
        Uncertainty weighting (Kendall et al., 2018).
        
        L = Σ (1/2σ²) * L_t + log(σ)
        """
        total = 0
        for loss, log_var in zip(losses, log_vars):
            precision = np.exp(-log_var)
            total += precision * loss + log_var
        return total
    
    def gradient_normalization(gradients: list) -> list:
        """
        GradNorm-style normalization.
        Returns normalized gradients.
        """
        norms = [np.linalg.norm(g) for g in gradients]
        avg_norm = np.mean(norms)
        
        normalized = [g * (avg_norm / (norm + 1e-10)) for g, norm in zip(gradients, norms)]
        return normalized
    
    def dynamic_weight_average(losses_t: list, losses_tm1: list, 
                              temperature: float = 2.0) -> np.ndarray:
        """
        Dynamic Weight Average (Liu et al., 2019).
        """
        n_tasks = len(losses_t)
        
        # Relative decrease rate
        r = [losses_t[i] / (losses_tm1[i] + 1e-10) for i in range(n_tasks)]
        
        # Softmax weighting
        exp_r = np.exp(np.array(r) / temperature)
        weights = n_tasks * exp_r / np.sum(exp_r)
        
        return weights
    
    # Simulate multi-task learning
    np.random.seed(42)
    n_tasks = 3
    n_epochs = 10
    
    # Task losses over training (simulated)
    task_losses = []
    for t in range(n_tasks):
        # Different convergence rates
        base = 1.0 / (t + 1)
        losses = [base * np.exp(-0.2 * (t + 1) * e) + 0.1 * np.random.rand() 
                 for e in range(n_epochs)]
        task_losses.append(losses)
    
    print("Multi-Task Loss Weighting:\n")
    
    # Compare strategies at different epochs
    print(f"{'Epoch':<8} {'Uniform':<12} {'Fixed':<12} {'Uncertainty':<12}")
    print("-" * 50)
    
    fixed_weights = [0.5, 0.3, 0.2]
    log_vars = np.array([-1.0, 0.0, 1.0])  # Learnable
    
    for epoch in [0, 4, 9]:
        current_losses = [task_losses[t][epoch] for t in range(n_tasks)]
        
        uniform = uniform_weighted_loss(current_losses)
        fixed = fixed_weighted_loss(current_losses, fixed_weights)
        uncertainty = uncertainty_weighted_loss(current_losses, log_vars)
        
        print(f"{epoch:<8} {uniform:<12.4f} {fixed:<12.4f} {uncertainty:<12.4f}")
    
    # Dynamic weight average
    print("\nDynamic Weight Average:")
    for epoch in range(1, n_epochs):
        losses_t = [task_losses[t][epoch] for t in range(n_tasks)]
        losses_tm1 = [task_losses[t][epoch-1] for t in range(n_tasks)]
        
        weights = dynamic_weight_average(losses_t, losses_tm1)
        
        if epoch in [1, 5, 9]:
            print(f"  Epoch {epoch}: weights = {weights.round(3)}")
    
    return uncertainty_weighted_loss, dynamic_weight_average


# =============================================================================
# Example 9: Label Smoothing and Regularization
# =============================================================================

def example_9_label_smoothing():
    """
    Implement label smoothing and related regularization.
    """
    print("\nExample 9: Label Smoothing and Regularization")
    print("=" * 60)
    
    def cross_entropy(logits: np.ndarray, labels: np.ndarray,
                     epsilon: float = 1e-7) -> float:
        """Standard cross-entropy."""
        probs = softmax(logits)
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        n = len(labels)
        return -np.sum(np.log(probs[np.arange(n), labels])) / n
    
    def softmax(x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def label_smoothing_ce(logits: np.ndarray, labels: np.ndarray,
                          smoothing: float = 0.1) -> float:
        """
        Cross-entropy with label smoothing.
        
        Smooth labels: y_smooth = (1 - ε) * y + ε / K
        """
        n, K = logits.shape
        
        probs = softmax(logits)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        
        # One-hot encoding
        one_hot = np.zeros((n, K))
        one_hot[np.arange(n), labels] = 1
        
        # Smooth labels
        smooth_labels = (1 - smoothing) * one_hot + smoothing / K
        
        return -np.sum(smooth_labels * np.log(probs)) / n
    
    def confidence_penalty(logits: np.ndarray, labels: np.ndarray,
                          beta: float = 0.1) -> float:
        """
        Add negative entropy as confidence penalty.
        
        L = CE + β * H(p)
        """
        probs = softmax(logits)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        
        n = len(labels)
        ce = -np.sum(np.log(probs[np.arange(n), labels])) / n
        
        # Negative entropy (penalize confident predictions)
        entropy = -np.sum(probs * np.log(probs)) / n
        
        return ce - beta * entropy  # Encourage higher entropy
    
    def mixup_loss(logits: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray,
                  lam: float) -> float:
        """
        MixUp training loss.
        
        x_mix = λ * x_a + (1-λ) * x_b
        L = λ * CE(f(x_mix), y_a) + (1-λ) * CE(f(x_mix), y_b)
        """
        n, K = logits.shape
        probs = softmax(logits)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        
        ce_a = -np.sum(np.log(probs[np.arange(n), labels_a])) / n
        ce_b = -np.sum(np.log(probs[np.arange(n), labels_b])) / n
        
        return lam * ce_a + (1 - lam) * ce_b
    
    # Example
    np.random.seed(42)
    n, K = 100, 10
    
    logits = np.random.randn(n, K)
    labels = np.random.randint(0, K, n)
    
    print("Loss Comparison:\n")
    print(f"{'Method':<25} {'Loss':<12}")
    print("-" * 40)
    
    print(f"{'Standard CE':<25} {cross_entropy(logits, labels):<12.4f}")
    
    for smoothing in [0.05, 0.1, 0.2]:
        loss = label_smoothing_ce(logits, labels, smoothing)
        print(f"{'Label Smooth (ε=' + str(smoothing) + ')':<25} {loss:<12.4f}")
    
    for beta in [0.1, 0.5]:
        loss = confidence_penalty(logits, labels, beta)
        print(f"{'Conf Penalty (β=' + str(beta) + ')':<25} {loss:<12.4f}")
    
    # MixUp
    print("\nMixUp Loss:")
    labels_b = np.random.randint(0, K, n)
    for lam in [0.5, 0.7, 0.9]:
        loss = mixup_loss(logits, labels, labels_b, lam)
        print(f"  λ = {lam}: {loss:.4f}")
    
    return label_smoothing_ce, confidence_penalty


# =============================================================================
# Example 10: Loss Landscape Analysis
# =============================================================================

def example_10_loss_landscape():
    """
    Analyze loss landscape properties.
    """
    print("\nExample 10: Loss Landscape Analysis")
    print("=" * 60)
    
    def compute_loss_surface(loss_fn: Callable, theta_center: np.ndarray,
                            direction1: np.ndarray, direction2: np.ndarray,
                            range_val: float = 1.0, n_points: int = 20) -> Tuple:
        """Compute 2D loss surface around a point."""
        alphas = np.linspace(-range_val, range_val, n_points)
        betas = np.linspace(-range_val, range_val, n_points)
        
        surface = np.zeros((n_points, n_points))
        
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                theta = theta_center + alpha * direction1 + beta * direction2
                surface[i, j] = loss_fn(theta)
        
        return alphas, betas, surface
    
    def estimate_smoothness(loss_fn: Callable, theta: np.ndarray,
                           n_samples: int = 100, epsilon: float = 0.1) -> float:
        """
        Estimate Lipschitz constant of gradient.
        L ≈ max ||∇L(θ1) - ∇L(θ2)|| / ||θ1 - θ2||
        """
        d = len(theta)
        max_L = 0
        
        for _ in range(n_samples):
            # Random perturbation
            delta = epsilon * np.random.randn(d)
            theta2 = theta + delta
            
            # Numerical gradients
            grad1 = numerical_gradient(loss_fn, theta)
            grad2 = numerical_gradient(loss_fn, theta2)
            
            L = np.linalg.norm(grad1 - grad2) / np.linalg.norm(delta)
            max_L = max(max_L, L)
        
        return max_L
    
    def numerical_gradient(f: Callable, x: np.ndarray, 
                          epsilon: float = 1e-5) -> np.ndarray:
        """Compute numerical gradient."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            x_minus = x.copy()
            x_minus[i] -= epsilon
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
        return grad
    
    def compute_hessian(loss_fn: Callable, theta: np.ndarray,
                       epsilon: float = 1e-5) -> np.ndarray:
        """Compute Hessian matrix numerically."""
        d = len(theta)
        H = np.zeros((d, d))
        
        for i in range(d):
            for j in range(d):
                # Second partial derivative
                theta_pp = theta.copy()
                theta_pp[i] += epsilon
                theta_pp[j] += epsilon
                
                theta_pm = theta.copy()
                theta_pm[i] += epsilon
                theta_pm[j] -= epsilon
                
                theta_mp = theta.copy()
                theta_mp[i] -= epsilon
                theta_mp[j] += epsilon
                
                theta_mm = theta.copy()
                theta_mm[i] -= epsilon
                theta_mm[j] -= epsilon
                
                H[i, j] = (loss_fn(theta_pp) - loss_fn(theta_pm) - 
                          loss_fn(theta_mp) + loss_fn(theta_mm)) / (4 * epsilon**2)
        
        return H
    
    # Create a simple loss function
    np.random.seed(42)
    d = 5
    
    # Quadratic loss with known properties
    A = np.random.randn(d, d)
    A = A @ A.T + 0.1 * np.eye(d)  # PSD
    b = np.random.randn(d)
    
    def quadratic_loss(theta):
        return 0.5 * theta @ A @ theta - b @ theta
    
    # Non-convex loss
    def nonconvex_loss(theta):
        return np.sum(np.sin(theta) + 0.1 * theta**2)
    
    theta_opt = np.linalg.solve(A, b)  # Optimal for quadratic
    
    print("Quadratic Loss Analysis:\n")
    
    # Condition number
    eigvals = np.linalg.eigvalsh(A)
    condition = eigvals.max() / eigvals.min()
    print(f"Condition number: {condition:.4f}")
    print(f"Eigenvalue range: [{eigvals.min():.4f}, {eigvals.max():.4f}]")
    
    # Smoothness
    smoothness = estimate_smoothness(quadratic_loss, theta_opt, n_samples=50)
    print(f"Estimated Lipschitz constant: {smoothness:.4f}")
    print(f"True Lipschitz constant: {eigvals.max():.4f}")
    
    # Hessian at optimum
    H = compute_hessian(quadratic_loss, theta_opt)
    H_eigvals = np.linalg.eigvalsh(H)
    
    print(f"\nHessian eigenvalues at optimum:")
    print(f"  Min: {H_eigvals.min():.4f}, Max: {H_eigvals.max():.4f}")
    print(f"  All positive (convex): {np.all(H_eigvals > 0)}")
    
    # Non-convex analysis
    print("\nNon-convex Loss Analysis:")
    theta_test = np.zeros(d)
    H_nonconvex = compute_hessian(nonconvex_loss, theta_test)
    H_nc_eigvals = np.linalg.eigvalsh(H_nonconvex)
    
    print(f"Hessian eigenvalues: {H_nc_eigvals.round(4)}")
    print(f"Has negative eigenvalues (non-convex): {np.any(H_nc_eigvals < 0)}")
    
    return compute_loss_surface, estimate_smoothness


def run_all_examples():
    """Run all loss function examples."""
    print("=" * 70)
    print("LOSS FUNCTIONS - EXAMPLES")
    print("=" * 70)
    
    example_1_regression_losses()
    example_2_classification_losses()
    example_3_focal_loss()
    example_4_kl_divergence()
    example_5_contrastive_losses()
    example_6_ranking_losses()
    example_7_probabilistic_losses()
    example_8_multitask_losses()
    example_9_label_smoothing()
    example_10_loss_landscape()
    
    print("\n" + "=" * 70)
    print("All loss function examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
