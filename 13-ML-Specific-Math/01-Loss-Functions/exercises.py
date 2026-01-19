"""
Loss Functions - Exercises
==========================

Practice problems for loss functions
with solutions and implementations.
"""

import numpy as np
from typing import Tuple, Callable, List, Dict, Optional


# =============================================================================
# Exercise 1: Implement and Compare Regression Losses
# =============================================================================

def exercise_1_regression_losses():
    """
    Exercise: Implement regression losses and analyze robustness.
    
    Tasks:
    1. Implement MSE, MAE, Huber, and Quantile losses
    2. Implement their gradients
    3. Compare robustness to outliers
    4. Find optimal predictions for each loss
    """
    print("Exercise 1: Regression Losses")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class RegressionLosses:
        """Collection of regression losses with gradients."""
        
        @staticmethod
        def mse(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
            """MSE loss and gradient."""
            diff = y_pred - y_true
            loss = np.mean(diff ** 2)
            grad = 2 * diff / len(y_true)
            return loss, grad
        
        @staticmethod
        def mae(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
            """MAE loss and subgradient."""
            diff = y_pred - y_true
            loss = np.mean(np.abs(diff))
            grad = np.sign(diff) / len(y_true)
            return loss, grad
        
        @staticmethod
        def huber(y_pred: np.ndarray, y_true: np.ndarray, 
                 delta: float = 1.0) -> Tuple[float, np.ndarray]:
            """Huber loss and gradient."""
            diff = y_pred - y_true
            abs_diff = np.abs(diff)
            
            loss = np.where(
                abs_diff <= delta,
                0.5 * diff ** 2,
                delta * (abs_diff - 0.5 * delta)
            )
            
            grad = np.where(
                abs_diff <= delta,
                diff,
                delta * np.sign(diff)
            )
            
            return np.mean(loss), grad / len(y_true)
        
        @staticmethod
        def quantile(y_pred: np.ndarray, y_true: np.ndarray,
                    tau: float = 0.5) -> Tuple[float, np.ndarray]:
            """Quantile loss and subgradient."""
            diff = y_true - y_pred
            loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
            grad = np.where(diff >= 0, -tau, 1 - tau)
            return np.mean(loss), grad / len(y_true)
        
        @staticmethod
        def find_optimal_prediction(y_true: np.ndarray, 
                                   loss_type: str) -> float:
            """Find prediction that minimizes loss."""
            if loss_type == 'mse':
                return np.mean(y_true)
            elif loss_type == 'mae':
                return np.median(y_true)
            elif loss_type == 'quantile_25':
                return np.percentile(y_true, 25)
            elif loss_type == 'quantile_75':
                return np.percentile(y_true, 75)
            else:
                raise ValueError(f"Unknown loss: {loss_type}")
    
    # Test with outliers
    np.random.seed(42)
    y_true = np.concatenate([
        np.random.randn(90),  # Normal data
        np.random.randn(10) + 10  # Outliers
    ])
    
    losses = RegressionLosses()
    
    print("\nOptimal Predictions:")
    print(f"  Mean (MSE optimal): {losses.find_optimal_prediction(y_true, 'mse'):.4f}")
    print(f"  Median (MAE optimal): {losses.find_optimal_prediction(y_true, 'mae'):.4f}")
    print(f"  True data mean (without outliers): {np.mean(y_true[:90]):.4f}")
    
    # Compare gradients at different predictions
    print("\nGradient Magnitudes at Mean:")
    y_pred = np.full_like(y_true, np.mean(y_true))
    
    for name, loss_fn in [('MSE', losses.mse), ('MAE', losses.mae), 
                          ('Huber', lambda y, t: losses.huber(y, t, 1.0))]:
        loss, grad = loss_fn(y_pred, y_true)
        print(f"  {name}: loss={loss:.4f}, ||grad||={np.linalg.norm(grad):.4f}")
    
    # Breakdown by data type
    print("\nOutlier Contribution:")
    y_pred_median = np.full_like(y_true, np.median(y_true))
    
    mse_normal = np.mean((y_pred_median[:90] - y_true[:90])**2)
    mse_outliers = np.mean((y_pred_median[90:] - y_true[90:])**2)
    
    mae_normal = np.mean(np.abs(y_pred_median[:90] - y_true[:90]))
    mae_outliers = np.mean(np.abs(y_pred_median[90:] - y_true[90:]))
    
    print(f"  MSE - Normal: {mse_normal:.4f}, Outliers: {mse_outliers:.4f}")
    print(f"  MAE - Normal: {mae_normal:.4f}, Outliers: {mae_outliers:.4f}")
    
    return RegressionLosses


# =============================================================================
# Exercise 2: Binary Classification Losses
# =============================================================================

def exercise_2_classification_losses():
    """
    Exercise: Implement and analyze binary classification losses.
    
    Tasks:
    1. Implement BCE, hinge, and exponential losses
    2. Compare decision boundaries
    3. Analyze gradient behavior
    4. Study effect of class imbalance
    """
    print("\nExercise 2: Binary Classification Losses")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class BinaryClassificationLosses:
        """Binary classification losses."""
        
        @staticmethod
        def sigmoid(z: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
        @staticmethod
        def bce(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
            """Binary cross-entropy with gradient w.r.t. logits."""
            probs = BinaryClassificationLosses.sigmoid(logits)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            
            loss = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
            grad = (probs - y_true) / len(y_true)
            
            return loss, grad
        
        @staticmethod
        def hinge(scores: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
            """Hinge loss (y in {-1, +1})."""
            y_signed = 2 * y_true - 1  # Convert 0/1 to -1/+1
            margin = y_signed * scores
            
            loss = np.mean(np.maximum(0, 1 - margin))
            grad = -y_signed * (margin < 1) / len(y_true)
            
            return loss, grad
        
        @staticmethod
        def exponential(scores: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
            """Exponential loss (AdaBoost)."""
            y_signed = 2 * y_true - 1
            
            loss = np.mean(np.exp(-y_signed * scores))
            grad = -y_signed * np.exp(-y_signed * scores) / len(y_true)
            
            return loss, grad
        
        @staticmethod
        def weighted_bce(logits: np.ndarray, y_true: np.ndarray,
                        pos_weight: float = 1.0) -> Tuple[float, np.ndarray]:
            """BCE with class weights for imbalance."""
            probs = BinaryClassificationLosses.sigmoid(logits)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            
            # Weight positive class
            weights = np.where(y_true == 1, pos_weight, 1.0)
            
            loss = -np.mean(weights * (y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)))
            grad = weights * (probs - y_true) / len(y_true)
            
            return loss, grad
    
    losses = BinaryClassificationLosses()
    
    # Compare loss values across score range
    print("\nLoss Values vs Score (y=1):")
    print(f"{'Score':<10} {'BCE':<12} {'Hinge':<12} {'Exp':<12}")
    print("-" * 50)
    
    for score in [-3, -1, 0, 1, 3]:
        y = np.array([1.0])
        bce_val, _ = losses.bce(np.array([score]), y)
        hinge_val, _ = losses.hinge(np.array([score]), y)
        exp_val, _ = losses.exponential(np.array([score]), y)
        
        print(f"{score:<10} {bce_val:<12.4f} {hinge_val:<12.4f} {exp_val:<12.4f}")
    
    # Class imbalance
    print("\nClass Imbalance Effect:")
    np.random.seed(42)
    n = 1000
    
    # 10% positive class
    y_true = np.zeros(n)
    y_true[:100] = 1
    logits = 0.5 * np.random.randn(n)  # Random predictions
    
    bce_unweighted, _ = losses.bce(logits, y_true)
    bce_weighted, _ = losses.weighted_bce(logits, y_true, pos_weight=9.0)
    
    print(f"  Unweighted BCE: {bce_unweighted:.4f}")
    print(f"  Weighted BCE (w=9): {bce_weighted:.4f}")
    
    # Gradient analysis
    print("\nGradient at Decision Boundary (score=0):")
    score_0 = np.array([0.0])
    
    for y in [0, 1]:
        y_arr = np.array([float(y)])
        _, bce_grad = losses.bce(score_0, y_arr)
        _, hinge_grad = losses.hinge(score_0, y_arr)
        
        print(f"  y={y}: BCE grad={bce_grad[0]:.4f}, Hinge grad={hinge_grad[0]:.4f}")
    
    return BinaryClassificationLosses


# =============================================================================
# Exercise 3: Multi-Class Cross-Entropy
# =============================================================================

def exercise_3_multiclass_ce():
    """
    Exercise: Implement multi-class cross-entropy with variations.
    
    Tasks:
    1. Implement softmax and log-softmax
    2. Implement cross-entropy and label smoothing
    3. Compute gradients
    4. Analyze numerical stability
    """
    print("\nExercise 3: Multi-Class Cross-Entropy")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class MultiClassCE:
        """Multi-class cross-entropy implementations."""
        
        @staticmethod
        def softmax(logits: np.ndarray) -> np.ndarray:
            """Numerically stable softmax."""
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exp_shifted = np.exp(shifted)
            return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        
        @staticmethod
        def log_softmax(logits: np.ndarray) -> np.ndarray:
            """Numerically stable log-softmax."""
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        
        @staticmethod
        def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
            """CE with gradient (labels are indices)."""
            n, K = logits.shape
            log_probs = MultiClassCE.log_softmax(logits)
            
            # Loss
            loss = -np.mean(log_probs[np.arange(n), labels])
            
            # Gradient
            probs = MultiClassCE.softmax(logits)
            grad = probs.copy()
            grad[np.arange(n), labels] -= 1
            grad /= n
            
            return loss, grad
        
        @staticmethod
        def label_smoothing_ce(logits: np.ndarray, labels: np.ndarray,
                              smoothing: float = 0.1) -> Tuple[float, np.ndarray]:
            """CE with label smoothing."""
            n, K = logits.shape
            log_probs = MultiClassCE.log_softmax(logits)
            probs = MultiClassCE.softmax(logits)
            
            # One-hot
            one_hot = np.zeros((n, K))
            one_hot[np.arange(n), labels] = 1
            
            # Smooth
            smooth_labels = (1 - smoothing) * one_hot + smoothing / K
            
            # Loss
            loss = -np.sum(smooth_labels * log_probs) / n
            
            # Gradient
            grad = (probs - smooth_labels) / n
            
            return loss, grad
        
        @staticmethod
        def verify_gradient(logits: np.ndarray, labels: np.ndarray,
                           epsilon: float = 1e-5) -> float:
            """Verify gradient numerically."""
            _, analytical_grad = MultiClassCE.cross_entropy(logits, labels)
            
            numerical_grad = np.zeros_like(logits)
            for i in range(logits.shape[0]):
                for j in range(logits.shape[1]):
                    logits_plus = logits.copy()
                    logits_plus[i, j] += epsilon
                    loss_plus, _ = MultiClassCE.cross_entropy(logits_plus, labels)
                    
                    logits_minus = logits.copy()
                    logits_minus[i, j] -= epsilon
                    loss_minus, _ = MultiClassCE.cross_entropy(logits_minus, labels)
                    
                    numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
            return np.max(np.abs(analytical_grad - numerical_grad))
    
    ce = MultiClassCE()
    
    # Numerical stability test
    print("\nNumerical Stability:")
    
    # Large logits (would overflow naive implementation)
    logits_large = np.array([[1000, 1001, 999]], dtype=float)
    probs = ce.softmax(logits_large)
    log_probs = ce.log_softmax(logits_large)
    
    print(f"  Large logits: {logits_large}")
    print(f"  Softmax: {probs}")
    print(f"  Log-softmax: {log_probs}")
    print(f"  Sum of probs: {probs.sum()}")
    
    # Gradient verification
    np.random.seed(42)
    logits = np.random.randn(10, 5)
    labels = np.random.randint(0, 5, 10)
    
    grad_error = ce.verify_gradient(logits, labels)
    print(f"\nGradient Verification Error: {grad_error:.2e}")
    
    # Label smoothing comparison
    print("\nLabel Smoothing Effect:")
    
    # Confident prediction
    logits_confident = np.array([[10, 0, 0, 0, 0]], dtype=float)
    labels_correct = np.array([0])
    
    loss_standard, _ = ce.cross_entropy(logits_confident, labels_correct)
    loss_smooth, _ = ce.label_smoothing_ce(logits_confident, labels_correct, 0.1)
    
    print(f"  Standard CE: {loss_standard:.6f}")
    print(f"  Label Smoothed (ε=0.1): {loss_smooth:.6f}")
    
    return MultiClassCE


# =============================================================================
# Exercise 4: Focal Loss Implementation
# =============================================================================

def exercise_4_focal_loss():
    """
    Exercise: Implement focal loss with gradient.
    
    Tasks:
    1. Implement binary focal loss
    2. Implement multi-class focal loss
    3. Analyze focusing behavior
    4. Compare with standard CE
    """
    print("\nExercise 4: Focal Loss")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class FocalLoss:
        """Focal loss for class imbalance."""
        
        @staticmethod
        def sigmoid(z: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
        @staticmethod
        def binary_focal(logits: np.ndarray, y_true: np.ndarray,
                        gamma: float = 2.0, alpha: float = 0.25) -> Tuple[float, np.ndarray]:
            """
            Binary focal loss: -α(1-p)^γ log(p) for y=1
            """
            p = FocalLoss.sigmoid(logits)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            
            # Focal weights
            pt = np.where(y_true == 1, p, 1 - p)
            focal_weight = (1 - pt) ** gamma
            
            # Alpha weights
            alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
            
            # BCE
            bce = -np.where(y_true == 1, np.log(p), np.log(1 - p))
            
            loss = np.mean(alpha_t * focal_weight * bce)
            
            # Gradient (complex due to focal term)
            # d/dz [-(1-p)^γ log(p)] for y=1
            # = (1-p)^γ (p-1) + γ(1-p)^(γ-1) p log(p)
            # Simplified: (p - y)(1 + γ log(pt))
            
            grad_focal = (p - y_true) * (1 - pt) ** (gamma - 1) * (
                gamma * pt * np.log(pt + 1e-10) + pt - y_true * gamma * np.log(pt + 1e-10)
            )
            
            # Approximate gradient
            grad = alpha_t * focal_weight * (p - y_true) / len(y_true)
            
            return loss, grad
        
        @staticmethod
        def multiclass_focal(logits: np.ndarray, labels: np.ndarray,
                            gamma: float = 2.0) -> Tuple[float, np.ndarray]:
            """Multi-class focal loss."""
            n, K = logits.shape
            
            # Softmax
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exp_shifted = np.exp(shifted)
            probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
            
            # Focal weight
            pt = probs[np.arange(n), labels]
            focal_weight = (1 - pt) ** gamma
            
            # Cross-entropy
            ce = -np.log(pt + 1e-7)
            
            loss = np.mean(focal_weight * ce)
            
            # Gradient
            grad = probs.copy()
            grad[np.arange(n), labels] -= 1
            grad *= focal_weight[:, np.newaxis]
            grad /= n
            
            return loss, grad
        
        @staticmethod
        def analyze_focusing(gamma_values: List[float]) -> Dict:
            """Analyze how focal loss focuses on hard examples."""
            results = {}
            
            for gamma in gamma_values:
                # Loss at different confidence levels
                confidences = np.linspace(0.1, 0.99, 50)
                focal_weights = (1 - confidences) ** gamma
                ce = -np.log(confidences)
                focal_losses = focal_weights * ce
                
                results[gamma] = {
                    'confidences': confidences,
                    'focal_weights': focal_weights,
                    'focal_losses': focal_losses
                }
            
            return results
    
    focal = FocalLoss()
    
    # Compare with CE for imbalanced data
    np.random.seed(42)
    n = 1000
    y_true = np.zeros(n)
    y_true[:50] = 1  # 5% positive
    
    # Model predicts mostly negative
    logits = -1 + 0.5 * np.random.randn(n)
    
    print("\nImbalanced Classification (5% positive):\n")
    print(f"{'Loss':<25} {'Value':<12}")
    print("-" * 40)
    
    # BCE
    bce_loss, _ = focal.binary_focal(logits, y_true, gamma=0, alpha=0.5)
    print(f"{'BCE':<25} {bce_loss:<12.4f}")
    
    # Focal with different gamma
    for gamma in [0.5, 1, 2, 5]:
        loss, _ = focal.binary_focal(logits, y_true, gamma=gamma)
        print(f"{'Focal (γ=' + str(gamma) + ')':<25} {loss:<12.4f}")
    
    # Per-sample analysis
    print("\nPer-Sample Loss Analysis:")
    
    # Easy true negative (low pred, y=0)
    easy_tn = focal.sigmoid(np.array([-3]))
    loss_easy_tn_ce, _ = focal.binary_focal(np.array([-3]), np.array([0]), gamma=0)
    loss_easy_tn_focal, _ = focal.binary_focal(np.array([-3]), np.array([0]), gamma=2)
    
    # Hard false negative (low pred, y=1)
    loss_hard_fn_ce, _ = focal.binary_focal(np.array([-3]), np.array([1]), gamma=0)
    loss_hard_fn_focal, _ = focal.binary_focal(np.array([-3]), np.array([1]), gamma=2)
    
    print(f"  Easy TN - CE: {loss_easy_tn_ce:.4f}, Focal: {loss_easy_tn_focal:.4f}")
    print(f"  Hard FN - CE: {loss_hard_fn_ce:.4f}, Focal: {loss_hard_fn_focal:.4f}")
    print(f"  Ratio (FN/TN) - CE: {loss_hard_fn_ce/loss_easy_tn_ce:.2f}, "
          f"Focal: {loss_hard_fn_focal/loss_easy_tn_focal:.2f}")
    
    return FocalLoss


# =============================================================================
# Exercise 5: KL Divergence and Entropy
# =============================================================================

def exercise_5_kl_entropy():
    """
    Exercise: Implement KL divergence and entropy-related losses.
    
    Tasks:
    1. Implement discrete and Gaussian KL
    2. Implement forward and reverse KL
    3. Implement JS divergence
    4. Analyze mode-seeking vs mode-covering
    """
    print("\nExercise 5: KL Divergence and Entropy")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class DivergenceLosses:
        """KL and related divergences."""
        
        @staticmethod
        def entropy(p: np.ndarray, epsilon: float = 1e-10) -> float:
            """Shannon entropy: H(p) = -Σ p log(p)"""
            p = np.clip(p, epsilon, 1)
            return -np.sum(p * np.log(p))
        
        @staticmethod
        def cross_entropy_dist(p: np.ndarray, q: np.ndarray, 
                              epsilon: float = 1e-10) -> float:
            """Cross-entropy: H(p,q) = -Σ p log(q)"""
            q = np.clip(q, epsilon, 1)
            return -np.sum(p * np.log(q))
        
        @staticmethod
        def kl_divergence(p: np.ndarray, q: np.ndarray,
                         epsilon: float = 1e-10) -> float:
            """KL(p||q) = Σ p log(p/q)"""
            p = np.clip(p, epsilon, 1)
            q = np.clip(q, epsilon, 1)
            return np.sum(p * np.log(p / q))
        
        @staticmethod
        def reverse_kl(p: np.ndarray, q: np.ndarray,
                      epsilon: float = 1e-10) -> float:
            """KL(q||p) - reverse KL"""
            return DivergenceLosses.kl_divergence(q, p, epsilon)
        
        @staticmethod
        def js_divergence(p: np.ndarray, q: np.ndarray,
                         epsilon: float = 1e-10) -> float:
            """Jensen-Shannon: JS(p||q) = 0.5*KL(p||m) + 0.5*KL(q||m)"""
            m = 0.5 * (p + q)
            return 0.5 * DivergenceLosses.kl_divergence(p, m, epsilon) + \
                   0.5 * DivergenceLosses.kl_divergence(q, m, epsilon)
        
        @staticmethod
        def gaussian_kl(mu_p: np.ndarray, logvar_p: np.ndarray,
                       mu_q: np.ndarray, logvar_q: np.ndarray) -> float:
            """KL between diagonal Gaussians."""
            var_p = np.exp(logvar_p)
            var_q = np.exp(logvar_q)
            
            return 0.5 * np.sum(
                logvar_q - logvar_p + 
                (var_p + (mu_p - mu_q)**2) / var_q - 1
            )
        
        @staticmethod
        def kl_to_prior(mu: np.ndarray, logvar: np.ndarray) -> float:
            """KL to standard normal: KL(N(μ,σ²)||N(0,1))"""
            return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
    
    div = DivergenceLosses()
    
    # Relationship between H, H(p,q), KL
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.5, 0.3, 0.2])
    
    print("\nFundamental Relationships:")
    H_p = div.entropy(p)
    H_pq = div.cross_entropy_dist(p, q)
    KL_pq = div.kl_divergence(p, q)
    
    print(f"  H(p) = {H_p:.4f}")
    print(f"  H(p,q) = {H_pq:.4f}")
    print(f"  KL(p||q) = {KL_pq:.4f}")
    print(f"  H(p,q) = H(p) + KL(p||q): {H_pq:.4f} ≈ {H_p + KL_pq:.4f}")
    
    # Asymmetry
    print("\nAsymmetry of KL:")
    KL_qp = div.kl_divergence(q, p)
    JS = div.js_divergence(p, q)
    
    print(f"  KL(p||q) = {KL_pq:.4f}")
    print(f"  KL(q||p) = {KL_qp:.4f}")
    print(f"  JS(p||q) = {JS:.4f} (symmetric)")
    
    # Mode-seeking vs mode-covering
    print("\nMode-Seeking vs Mode-Covering:")
    
    # Bimodal target
    p_bimodal = np.array([0.5, 0, 0.5])  # Modes at 0 and 2
    
    # Unimodal approximation
    q_mode1 = np.array([1.0, 0, 0])  # Only first mode
    q_spread = np.array([0.33, 0.34, 0.33])  # Spread across
    
    print(f"  p (bimodal): {p_bimodal}")
    print(f"  q1 (single mode): {q_mode1}")
    print(f"  q2 (spread): {q_spread}")
    
    print(f"\n  Forward KL(p||q) - mode covering:")
    print(f"    KL(p||q1) = {div.kl_divergence(p_bimodal, q_mode1 + 1e-10):.4f}")
    print(f"    KL(p||q2) = {div.kl_divergence(p_bimodal, q_spread):.4f}")
    
    print(f"\n  Reverse KL(q||p) - mode seeking:")
    print(f"    KL(q1||p) = {div.kl_divergence(q_mode1 + 1e-10, p_bimodal):.4f}")
    print(f"    KL(q2||p) = {div.kl_divergence(q_spread, p_bimodal):.4f}")
    
    return DivergenceLosses


# =============================================================================
# Exercise 6: Contrastive Learning Losses
# =============================================================================

def exercise_6_contrastive():
    """
    Exercise: Implement contrastive learning losses.
    
    Tasks:
    1. Implement triplet loss with different margins
    2. Implement N-pairs loss
    3. Implement InfoNCE/NT-Xent
    4. Analyze temperature effects
    """
    print("\nExercise 6: Contrastive Learning Losses")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class ContrastiveLosses:
        """Contrastive learning losses."""
        
        @staticmethod
        def triplet_loss(anchor: np.ndarray, positive: np.ndarray, 
                        negative: np.ndarray, margin: float = 1.0) -> float:
            """
            Triplet loss: max(0, d(a,p) - d(a,n) + margin)
            """
            dist_pos = np.linalg.norm(anchor - positive, axis=-1)
            dist_neg = np.linalg.norm(anchor - negative, axis=-1)
            
            return np.mean(np.maximum(0, dist_pos - dist_neg + margin))
        
        @staticmethod
        def n_pairs_loss(anchors: np.ndarray, positives: np.ndarray) -> float:
            """
            N-pairs loss: -log(exp(a·p+) / Σexp(a·p))
            Uses other positives in batch as negatives.
            """
            n = len(anchors)
            
            # Similarity matrix
            sims = anchors @ positives.T
            
            # Loss: softmax cross-entropy with diagonal as positive
            exp_sims = np.exp(sims - np.max(sims, axis=1, keepdims=True))
            softmax = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)
            
            return -np.mean(np.log(np.diag(softmax) + 1e-10))
        
        @staticmethod
        def info_nce(anchor: np.ndarray, positive: np.ndarray,
                    negatives: np.ndarray, temperature: float = 0.07) -> float:
            """
            InfoNCE loss: -log(exp(s+/τ) / (exp(s+/τ) + Σexp(s-/τ)))
            """
            pos_sim = np.dot(anchor, positive) / temperature
            neg_sims = negatives @ anchor / temperature
            
            # Log-sum-exp for stability
            max_sim = max(pos_sim, np.max(neg_sims))
            log_sum_exp = max_sim + np.log(
                np.exp(pos_sim - max_sim) + np.sum(np.exp(neg_sims - max_sim))
            )
            
            return -pos_sim + log_sum_exp
        
        @staticmethod
        def nt_xent(z_i: np.ndarray, z_j: np.ndarray,
                   temperature: float = 0.5) -> float:
            """
            NT-Xent (SimCLR): symmetric contrastive loss for augmented pairs.
            """
            n = len(z_i)
            
            # Normalize
            z_i = z_i / (np.linalg.norm(z_i, axis=1, keepdims=True) + 1e-10)
            z_j = z_j / (np.linalg.norm(z_j, axis=1, keepdims=True) + 1e-10)
            
            # Concatenate representations
            z = np.vstack([z_i, z_j])  # 2n x d
            
            # Similarity matrix
            sim = z @ z.T / temperature
            
            # Mask self-similarity
            mask = np.eye(2 * n, dtype=bool)
            sim[mask] = -np.inf
            
            # Loss
            total_loss = 0
            for i in range(n):
                # Positive pair: (i, n+i) and (n+i, i)
                pos_sim_1 = sim[i, n + i]
                pos_sim_2 = sim[n + i, i]
                
                # Softmax denominator (all others)
                denom_1 = np.sum(np.exp(sim[i]))
                denom_2 = np.sum(np.exp(sim[n + i]))
                
                total_loss -= np.log(np.exp(pos_sim_1) / denom_1 + 1e-10)
                total_loss -= np.log(np.exp(pos_sim_2) / denom_2 + 1e-10)
            
            return total_loss / (2 * n)
        
        @staticmethod
        def analyze_temperature(anchor: np.ndarray, positive: np.ndarray,
                               negatives: np.ndarray, temps: List[float]) -> Dict:
            """Analyze temperature effect on InfoNCE."""
            results = {}
            
            for tau in temps:
                loss = ContrastiveLosses.info_nce(anchor, positive, negatives, tau)
                
                # Effective number of negatives
                pos_sim = np.dot(anchor, positive)
                neg_sims = negatives @ anchor
                
                probs = np.exp((neg_sims - pos_sim) / tau)
                effective_negs = np.sum(probs > 0.01)
                
                results[tau] = {'loss': loss, 'effective_negatives': effective_negs}
            
            return results
    
    cl = ContrastiveLosses()
    
    np.random.seed(42)
    dim = 128
    
    # Normalized embeddings
    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)
    
    anchor = normalize(np.random.randn(dim))
    positive = normalize(anchor + 0.3 * np.random.randn(dim))  # Similar
    negatives = normalize(np.random.randn(100, dim))  # Random
    
    print("\nContrastive Loss Comparison:\n")
    
    # Triplet loss
    for margin in [0.5, 1.0, 2.0]:
        # Single negative
        loss = cl.triplet_loss(anchor, positive, negatives[0], margin)
        print(f"Triplet (margin={margin}): {loss:.4f}")
    
    print()
    
    # InfoNCE with different temperatures
    print("InfoNCE Temperature Effect:")
    results = cl.analyze_temperature(anchor, positive, negatives, [0.05, 0.1, 0.5, 1.0])
    
    for tau, res in results.items():
        print(f"  τ={tau}: loss={res['loss']:.4f}, "
              f"effective_negs={res['effective_negatives']}")
    
    # NT-Xent batch
    batch_size = 32
    z_i = normalize(np.random.randn(batch_size, dim))
    z_j = normalize(z_i + 0.2 * np.random.randn(batch_size, dim))
    
    nt_xent_loss = cl.nt_xent(z_i, z_j, temperature=0.5)
    print(f"\nNT-Xent (batch={batch_size}, τ=0.5): {nt_xent_loss:.4f}")
    
    return ContrastiveLosses


# =============================================================================
# Exercise 7: Ranking Losses
# =============================================================================

def exercise_7_ranking():
    """
    Exercise: Implement learning-to-rank losses.
    
    Tasks:
    1. Implement pairwise losses (hinge, logistic)
    2. Implement listwise losses
    3. Implement NDCG-based losses
    4. Compare ranking quality
    """
    print("\nExercise 7: Ranking Losses")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class RankingLosses:
        """Learning-to-rank losses."""
        
        @staticmethod
        def pairwise_hinge(scores: np.ndarray, relevance: np.ndarray,
                         margin: float = 1.0) -> float:
            """Pairwise hinge ranking loss."""
            n = len(scores)
            loss = 0
            count = 0
            
            for i in range(n):
                for j in range(n):
                    if relevance[i] > relevance[j]:
                        loss += max(0, margin - scores[i] + scores[j])
                        count += 1
            
            return loss / max(count, 1)
        
        @staticmethod
        def pairwise_logistic(scores: np.ndarray, relevance: np.ndarray,
                             sigma: float = 1.0) -> float:
            """RankNet loss: logistic pairwise."""
            n = len(scores)
            loss = 0
            count = 0
            
            for i in range(n):
                for j in range(n):
                    if relevance[i] > relevance[j]:
                        diff = sigma * (scores[i] - scores[j])
                        loss += np.log(1 + np.exp(-diff))
                        count += 1
            
            return loss / max(count, 1)
        
        @staticmethod
        def listnet_loss(scores: np.ndarray, relevance: np.ndarray) -> float:
            """ListNet: softmax cross-entropy over permutation probs."""
            # Top-1 probability approximation
            def top1_prob(s):
                s = s - np.max(s)
                return np.exp(s) / np.sum(np.exp(s))
            
            pred_probs = top1_prob(scores)
            true_probs = top1_prob(relevance)
            
            return -np.sum(true_probs * np.log(pred_probs + 1e-10))
        
        @staticmethod
        def ndcg(scores: np.ndarray, relevance: np.ndarray, k: int = None) -> float:
            """Compute NDCG@k."""
            if k is None:
                k = len(scores)
            
            # Ranking by scores
            ranking = np.argsort(-scores)
            
            # DCG
            dcg = sum(
                (2 ** relevance[ranking[i]] - 1) / np.log2(i + 2)
                for i in range(min(k, len(scores)))
            )
            
            # Ideal DCG
            ideal_ranking = np.argsort(-relevance)
            idcg = sum(
                (2 ** relevance[ideal_ranking[i]] - 1) / np.log2(i + 2)
                for i in range(min(k, len(scores)))
            )
            
            return dcg / max(idcg, 1e-10)
        
        @staticmethod
        def approx_ndcg_loss(scores: np.ndarray, relevance: np.ndarray,
                            tau: float = 1.0) -> float:
            """Approximate NDCG loss using softmax relaxation."""
            n = len(scores)
            
            # Soft ranks via softmax
            soft_ranks = np.zeros(n)
            for i in range(n):
                # Probability of being ranked above each other item
                probs = 1 / (1 + np.exp(-(scores[i] - scores) / tau))
                soft_ranks[i] = np.sum(probs)  # Expected rank (1-indexed approx)
            
            # Approximate DCG with soft ranks
            gains = 2 ** relevance - 1
            discounts = 1 / np.log2(soft_ranks + 1)
            
            dcg = np.sum(gains * discounts)
            
            # Ideal DCG
            ideal_ranks = np.argsort(-relevance) + 1
            ideal_dcg = np.sum(gains / np.log2(ideal_ranks + 1))
            
            # Return negative NDCG (for minimization)
            return -dcg / max(ideal_dcg, 1e-10)
    
    rl = RankingLosses()
    
    # Example: search ranking
    np.random.seed(42)
    n = 10
    relevance = np.array([4, 3, 3, 2, 2, 1, 1, 0, 0, 0])
    
    # Different scoring models
    good_scores = relevance + 0.3 * np.random.randn(n)  # Correlated
    bad_scores = np.random.randn(n)  # Random
    inverted_scores = -relevance + 0.3 * np.random.randn(n)  # Inverted
    
    print("\nRanking Loss Comparison:\n")
    print(f"{'Scores':<12} {'Hinge':<10} {'Logistic':<10} {'ListNet':<10} {'NDCG':<10}")
    print("-" * 55)
    
    for name, scores in [('Good', good_scores), ('Random', bad_scores), 
                         ('Inverted', inverted_scores)]:
        hinge = rl.pairwise_hinge(scores, relevance)
        logistic = rl.pairwise_logistic(scores, relevance)
        listnet = rl.listnet_loss(scores, relevance)
        ndcg = rl.ndcg(scores, relevance)
        
        print(f"{name:<12} {hinge:<10.4f} {logistic:<10.4f} {listnet:<10.4f} {ndcg:<10.4f}")
    
    # NDCG@k analysis
    print("\nNDCG@k (Good scores):")
    for k in [1, 3, 5, 10]:
        ndcg_k = rl.ndcg(good_scores, relevance, k)
        print(f"  NDCG@{k}: {ndcg_k:.4f}")
    
    return RankingLosses


# =============================================================================
# Exercise 8: Probabilistic Losses
# =============================================================================

def exercise_8_probabilistic():
    """
    Exercise: Implement probabilistic loss functions.
    
    Tasks:
    1. Implement Gaussian and Laplace NLL
    2. Implement mixture density loss
    3. Implement calibration loss (ECE)
    4. Compare aleatoric and epistemic uncertainty
    """
    print("\nExercise 8: Probabilistic Losses")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class ProbabilisticLosses:
        """Losses for uncertainty estimation."""
        
        @staticmethod
        def gaussian_nll(mean: np.ndarray, var: np.ndarray,
                        target: np.ndarray) -> float:
            """Negative log-likelihood for heteroscedastic Gaussian."""
            return np.mean(
                0.5 * np.log(2 * np.pi * var) + 
                (target - mean) ** 2 / (2 * var)
            )
        
        @staticmethod
        def laplace_nll(loc: np.ndarray, scale: np.ndarray,
                       target: np.ndarray) -> float:
            """NLL for Laplace distribution (robust to outliers)."""
            return np.mean(
                np.log(2 * scale) + np.abs(target - loc) / scale
            )
        
        @staticmethod
        def mixture_gaussian_nll(weights: np.ndarray, means: np.ndarray,
                                vars: np.ndarray, target: np.ndarray) -> float:
            """MDN loss: mixture of Gaussians."""
            n = len(target)
            k = len(weights[0])
            
            log_probs = np.zeros(n)
            
            for i in range(n):
                # Mixture probability
                mixture_prob = 0
                for j in range(k):
                    component_prob = weights[i, j] / np.sqrt(2 * np.pi * vars[i, j])
                    component_prob *= np.exp(-(target[i] - means[i, j])**2 / (2 * vars[i, j]))
                    mixture_prob += component_prob
                
                log_probs[i] = np.log(mixture_prob + 1e-10)
            
            return -np.mean(log_probs)
        
        @staticmethod
        def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                       n_bins: int = 10) -> float:
            """ECE: expected absolute difference between confidence and accuracy."""
            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0
            
            for i in range(n_bins):
                mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
                if np.sum(mask) > 0:
                    avg_conf = np.mean(probs[mask])
                    avg_acc = np.mean(labels[mask])
                    ece += np.sum(mask) / len(probs) * abs(avg_acc - avg_conf)
            
            return ece
        
        @staticmethod
        def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
            """Brier score: MSE of probability estimates."""
            return np.mean((probs - labels) ** 2)
    
    pl = ProbabilisticLosses()
    
    np.random.seed(42)
    n = 500
    
    # Generate heteroscedastic data
    x = np.sort(np.random.uniform(0, 10, n))
    true_mean = np.sin(x)
    true_var = 0.1 + 0.1 * np.abs(x - 5)  # Variance increases away from center
    y = true_mean + np.sqrt(true_var) * np.random.randn(n)
    
    print("\nHeteroscedastic Regression:\n")
    
    # Homoscedastic model (constant variance)
    pred_mean = true_mean + 0.05 * np.random.randn(n)  # Good mean
    pred_var_const = np.ones(n) * np.var(y - true_mean)
    
    # Heteroscedastic model (learned variance)
    pred_var_learned = true_var + 0.02 * np.random.randn(n)
    pred_var_learned = np.maximum(pred_var_learned, 0.01)
    
    nll_const = pl.gaussian_nll(pred_mean, pred_var_const, y)
    nll_learned = pl.gaussian_nll(pred_mean, pred_var_learned, y)
    
    print(f"Constant variance NLL: {nll_const:.4f}")
    print(f"Learned variance NLL: {nll_learned:.4f}")
    
    # Calibration analysis
    print("\nCalibration Analysis:")
    
    # Generate classification predictions
    true_probs = np.random.rand(1000)
    labels = (np.random.rand(1000) < true_probs).astype(float)
    
    # Calibrated model
    pred_calibrated = true_probs + 0.05 * np.random.randn(1000)
    pred_calibrated = np.clip(pred_calibrated, 0, 1)
    
    # Overconfident model
    pred_overconf = 0.5 + 0.8 * (true_probs - 0.5)
    pred_overconf = np.clip(pred_overconf, 0.05, 0.95)
    
    ece_calib = pl.expected_calibration_error(pred_calibrated, labels)
    ece_over = pl.expected_calibration_error(pred_overconf, labels)
    
    brier_calib = pl.brier_score(pred_calibrated, labels)
    brier_over = pl.brier_score(pred_overconf, labels)
    
    print(f"Calibrated - ECE: {ece_calib:.4f}, Brier: {brier_calib:.4f}")
    print(f"Overconfident - ECE: {ece_over:.4f}, Brier: {brier_over:.4f}")
    
    return ProbabilisticLosses


# =============================================================================
# Exercise 9: Multi-Task Loss Balancing
# =============================================================================

def exercise_9_multitask():
    """
    Exercise: Implement multi-task loss balancing strategies.
    
    Tasks:
    1. Implement uncertainty weighting
    2. Implement gradient normalization
    3. Implement dynamic weight averaging
    4. Compare strategies on multi-task problem
    """
    print("\nExercise 9: Multi-Task Loss Balancing")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class MultiTaskLosses:
        """Multi-task learning loss strategies."""
        
        @staticmethod
        def uniform(losses: List[float]) -> float:
            """Simple average."""
            return np.mean(losses)
        
        @staticmethod
        def weighted(losses: List[float], weights: List[float]) -> float:
            """Fixed weights."""
            return sum(w * l for w, l in zip(weights, losses))
        
        @staticmethod
        def uncertainty_weighting(losses: List[float], 
                                 log_vars: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Homoscedastic uncertainty weighting.
            L = Σ (1/2σ²)L_t + log(σ)
            
            Returns (total_loss, gradients for log_vars)
            """
            total_loss = 0
            grads = np.zeros(len(losses))
            
            for i, (loss, log_var) in enumerate(zip(losses, log_vars)):
                precision = np.exp(-log_var)
                total_loss += 0.5 * precision * loss + 0.5 * log_var
                grads[i] = -0.5 * precision * loss + 0.5
            
            return total_loss, grads
        
        @staticmethod
        def gradnorm(task_losses: List[float], task_losses_initial: List[float],
                    weights: np.ndarray, alpha: float = 0.5) -> np.ndarray:
            """
            GradNorm: balance gradient magnitudes.
            Returns updated weights.
            """
            n_tasks = len(task_losses)
            
            # Relative training rates
            ratios = np.array([
                task_losses[i] / (task_losses_initial[i] + 1e-10)
                for i in range(n_tasks)
            ])
            
            # Inverse relative training rate
            inv_rate = ratios ** (-alpha)
            
            # Target gradient norm ratio
            target_ratio = inv_rate / np.mean(inv_rate)
            
            # Update weights
            new_weights = weights * target_ratio
            new_weights = new_weights / new_weights.sum() * n_tasks  # Normalize
            
            return new_weights
        
        @staticmethod
        def dwa(losses_t: List[float], losses_tm1: List[float],
               temperature: float = 2.0) -> np.ndarray:
            """Dynamic Weight Average."""
            n_tasks = len(losses_t)
            
            ratios = np.array([
                losses_t[i] / (losses_tm1[i] + 1e-10)
                for i in range(n_tasks)
            ])
            
            # Softmax
            exp_ratios = np.exp(ratios / temperature)
            weights = n_tasks * exp_ratios / np.sum(exp_ratios)
            
            return weights
        
        @staticmethod
        def pcgrad(grads: List[np.ndarray]) -> List[np.ndarray]:
            """
            PCGrad: project conflicting gradients.
            """
            n_tasks = len(grads)
            modified_grads = [g.copy() for g in grads]
            
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i != j:
                        # Check for conflict
                        dot_product = np.dot(modified_grads[i], grads[j])
                        if dot_product < 0:
                            # Project out conflicting component
                            proj = dot_product / (np.dot(grads[j], grads[j]) + 1e-10)
                            modified_grads[i] -= proj * grads[j]
            
            return modified_grads
    
    mtl = MultiTaskLosses()
    
    # Simulate multi-task learning
    np.random.seed(42)
    n_epochs = 20
    n_tasks = 3
    
    # Simulated task losses (different convergence rates)
    task_losses_history = []
    for t in range(n_tasks):
        decay = 0.3 * (t + 1)  # Different decay rates
        losses = [np.exp(-decay * e) + 0.1 for e in range(n_epochs)]
        task_losses_history.append(losses)
    
    print("\nComparing Loss Balancing Strategies:\n")
    
    # Track total losses over epochs
    uniform_total = []
    uncertainty_total = []
    dwa_total = []
    
    log_vars = np.zeros(n_tasks)  # Learnable
    dwa_weights = np.ones(n_tasks)
    
    for epoch in range(n_epochs):
        current_losses = [task_losses_history[t][epoch] for t in range(n_tasks)]
        
        # Uniform
        uniform_total.append(mtl.uniform(current_losses))
        
        # Uncertainty weighting
        uw_loss, uw_grads = mtl.uncertainty_weighting(current_losses, log_vars)
        uncertainty_total.append(uw_loss)
        log_vars -= 0.1 * uw_grads  # Update log_vars
        
        # DWA (need previous losses)
        if epoch > 0:
            prev_losses = [task_losses_history[t][epoch-1] for t in range(n_tasks)]
            dwa_weights = mtl.dwa(current_losses, prev_losses)
        
        dwa_total.append(mtl.weighted(current_losses, dwa_weights / n_tasks))
    
    print(f"{'Epoch':<8} {'Uniform':<12} {'Uncertainty':<12} {'DWA':<12}")
    print("-" * 45)
    
    for epoch in [0, 5, 10, 19]:
        print(f"{epoch:<8} {uniform_total[epoch]:<12.4f} "
              f"{uncertainty_total[epoch]:<12.4f} {dwa_total[epoch]:<12.4f}")
    
    # PCGrad example
    print("\nPCGrad Conflict Resolution:")
    
    # Conflicting gradients
    grad1 = np.array([1.0, 0.5])
    grad2 = np.array([-0.5, 0.3])
    
    conflict = np.dot(grad1, grad2)
    print(f"  Original: g1·g2 = {conflict:.4f} (negative = conflict)")
    
    modified = mtl.pcgrad([grad1, grad2])
    new_conflict = np.dot(modified[0], modified[1])
    print(f"  After PCGrad: g1'·g2' = {new_conflict:.4f}")
    
    return MultiTaskLosses


# =============================================================================
# Exercise 10: Complete Loss Function Pipeline
# =============================================================================

def exercise_10_complete_pipeline():
    """
    Exercise: Build a complete loss function analysis pipeline.
    
    Tasks:
    1. Implement loss with regularization
    2. Add gradient clipping
    3. Implement learning rate scheduling based on loss
    4. Monitor and visualize loss landscape
    """
    print("\nExercise 10: Complete Loss Pipeline")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class LossPipeline:
        """Complete loss function pipeline."""
        
        def __init__(self, base_loss: Callable, l2_lambda: float = 0.01,
                    grad_clip: float = 1.0):
            self.base_loss = base_loss
            self.l2_lambda = l2_lambda
            self.grad_clip = grad_clip
            self.history = []
        
        def compute_loss(self, predictions: np.ndarray, targets: np.ndarray,
                        parameters: np.ndarray) -> Tuple[float, np.ndarray]:
            """Compute loss with regularization."""
            # Base loss
            base_loss, base_grad = self.base_loss(predictions, targets)
            
            # L2 regularization
            l2_reg = 0.5 * self.l2_lambda * np.sum(parameters ** 2)
            l2_grad = self.l2_lambda * parameters
            
            total_loss = base_loss + l2_reg
            
            return total_loss, base_grad  # Note: l2_grad applies to params, not predictions
        
        def clip_gradients(self, grads: np.ndarray) -> np.ndarray:
            """Gradient clipping."""
            grad_norm = np.linalg.norm(grads)
            if grad_norm > self.grad_clip:
                grads = grads * self.grad_clip / grad_norm
            return grads
        
        def step(self, predictions: np.ndarray, targets: np.ndarray,
                parameters: np.ndarray) -> Tuple[float, np.ndarray]:
            """Complete forward-backward step."""
            loss, grad = self.compute_loss(predictions, targets, parameters)
            grad = self.clip_gradients(grad)
            
            self.history.append({
                'loss': loss,
                'grad_norm': np.linalg.norm(grad)
            })
            
            return loss, grad
        
        def learning_rate_schedule(self, epoch: int, initial_lr: float,
                                   schedule: str = 'cosine') -> float:
            """Compute learning rate for epoch."""
            if schedule == 'cosine':
                return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / 100))
            elif schedule == 'exponential':
                return initial_lr * 0.95 ** epoch
            elif schedule == 'reduce_on_plateau':
                # Check if loss is not decreasing
                if len(self.history) >= 5:
                    recent = [h['loss'] for h in self.history[-5:]]
                    if min(recent) == recent[0]:  # No improvement
                        return initial_lr * 0.5
                return initial_lr
            else:
                return initial_lr
        
        def analyze_loss_landscape(self, loss_fn: Callable, params: np.ndarray,
                                   data: Tuple, n_points: int = 20) -> Dict:
            """2D loss landscape analysis."""
            x, y = data
            
            # Random directions
            dir1 = np.random.randn(*params.shape)
            dir1 /= np.linalg.norm(dir1)
            
            dir2 = np.random.randn(*params.shape)
            dir2 -= np.dot(dir2.flatten(), dir1.flatten()) * dir1
            dir2 /= np.linalg.norm(dir2)
            
            # Compute surface
            alphas = np.linspace(-1, 1, n_points)
            surface = np.zeros((n_points, n_points))
            
            for i, a in enumerate(alphas):
                for j, b in enumerate(alphas):
                    params_perturbed = params + a * dir1 + b * dir2
                    # Simplified: compute MSE as loss
                    pred = x @ params_perturbed  # Linear model
                    surface[i, j] = np.mean((pred - y) ** 2)
            
            return {
                'surface': surface,
                'min_loss': surface.min(),
                'max_loss': surface.max(),
                'curvature': np.max(np.gradient(np.gradient(surface)))
            }
    
    # Example base loss
    def mse_loss(pred, target):
        loss = np.mean((pred - target) ** 2)
        grad = 2 * (pred - target) / len(target)
        return loss, grad
    
    pipeline = LossPipeline(mse_loss, l2_lambda=0.01, grad_clip=1.0)
    
    # Simulate training
    np.random.seed(42)
    n = 100
    d = 10
    
    X = np.random.randn(n, d)
    true_params = np.random.randn(d)
    y = X @ true_params + 0.1 * np.random.randn(n)
    
    params = np.random.randn(d)
    lr = 0.1
    
    print("\nTraining with Complete Pipeline:\n")
    print(f"{'Epoch':<8} {'Loss':<12} {'Grad Norm':<12} {'LR':<10}")
    print("-" * 45)
    
    for epoch in range(20):
        predictions = X @ params
        loss, grad = pipeline.step(predictions, y, params)
        
        # Update with scheduled LR
        scheduled_lr = pipeline.learning_rate_schedule(epoch, lr, 'cosine')
        params -= scheduled_lr * (grad + pipeline.l2_lambda * params)
        
        if epoch % 5 == 0:
            print(f"{epoch:<8} {loss:<12.6f} "
                  f"{pipeline.history[-1]['grad_norm']:<12.6f} "
                  f"{scheduled_lr:<10.6f}")
    
    # Landscape analysis
    print("\nLoss Landscape Analysis:")
    landscape = pipeline.analyze_loss_landscape(mse_loss, params, (X, y))
    print(f"  Min loss: {landscape['min_loss']:.6f}")
    print(f"  Max loss: {landscape['max_loss']:.6f}")
    print(f"  Curvature: {landscape['curvature']:.6f}")
    
    return LossPipeline


def run_all_exercises():
    """Run all loss function exercises."""
    print("=" * 70)
    print("LOSS FUNCTIONS - EXERCISES")
    print("=" * 70)
    
    exercise_1_regression_losses()
    exercise_2_classification_losses()
    exercise_3_multiclass_ce()
    exercise_4_focal_loss()
    exercise_5_kl_entropy()
    exercise_6_contrastive()
    exercise_7_ranking()
    exercise_8_probabilistic()
    exercise_9_multitask()
    exercise_10_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("All loss function exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_exercises()
