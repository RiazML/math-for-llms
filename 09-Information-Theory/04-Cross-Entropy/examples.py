"""
Cross-Entropy - Examples
========================
Computing and applying cross-entropy loss.
"""

import numpy as np
from scipy.special import softmax, logsumexp


def example_basic_cross_entropy():
    """Basic cross-entropy computation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Cross-Entropy")
    print("=" * 60)
    
    def cross_entropy(p, q):
        """H(P, Q) = -sum(P * log(Q))"""
        p = np.array(p)
        q = np.array(q)
        # Add small epsilon for numerical stability
        return -np.sum(p * np.log(q + 1e-10))
    
    def entropy(p):
        """H(P) = -sum(P * log(P))"""
        p = np.array(p)
        return -np.sum(p * np.log(p + 1e-10))
    
    def kl_divergence(p, q):
        """D_KL(P || Q)"""
        p = np.array(p)
        q = np.array(q)
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    # True distribution P
    p = np.array([0.7, 0.2, 0.1])
    
    # Different predictions Q
    q_good = np.array([0.65, 0.25, 0.1])    # Close to P
    q_bad = np.array([0.33, 0.33, 0.34])    # Uniform
    q_worse = np.array([0.1, 0.2, 0.7])     # Reversed
    
    print(f"True distribution P = {p}")
    print(f"Entropy H(P) = {entropy(p):.4f}\n")
    
    print(f"{'Prediction Q':>25} {'H(P,Q)':>10} {'D_KL':>10}")
    print("-" * 50)
    
    for name, q in [("Close to P", q_good), ("Uniform", q_bad), ("Reversed", q_worse)]:
        ce = cross_entropy(p, q)
        kl = kl_divergence(p, q)
        print(f"{name:>25} {ce:>10.4f} {kl:>10.4f}")
    
    print(f"\nVerification: H(P,Q) = H(P) + D_KL(P||Q)")
    ce_good = cross_entropy(p, q_good)
    kl_good = kl_divergence(p, q_good)
    print(f"  {ce_good:.4f} = {entropy(p):.4f} + {kl_good:.4f} ✓")


def example_binary_cross_entropy():
    """Binary cross-entropy loss."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Binary Cross-Entropy Loss")
    print("=" * 60)
    
    def binary_cross_entropy(y_true, y_pred):
        """BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]"""
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    print("BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]")
    print("\nLoss for different predictions:")
    
    print(f"\n{'y_true':>8} {'y_pred':>10} {'BCE':>10}")
    print("-" * 32)
    
    test_cases = [
        (1, 0.99),   # Correct, confident
        (1, 0.7),    # Correct, less confident
        (1, 0.5),    # Uncertain
        (1, 0.1),    # Wrong, confident
        (0, 0.1),    # Correct, confident
        (0, 0.5),    # Uncertain
        (0, 0.9),    # Wrong, confident
    ]
    
    for y, y_hat in test_cases:
        bce = binary_cross_entropy(y, y_hat)
        print(f"{y:>8} {y_hat:>10.2f} {bce:>10.4f}")
    
    print("\nNote: Wrong confident predictions have high loss")


def example_multiclass_cross_entropy():
    """Multi-class cross-entropy loss."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multi-Class Cross-Entropy")
    print("=" * 60)
    
    def cross_entropy_loss(y_true_idx, y_pred_probs):
        """CE = -log(predicted probability of true class)"""
        return -np.log(y_pred_probs[y_true_idx] + 1e-10)
    
    # 4-class classification
    classes = ['cat', 'dog', 'bird', 'fish']
    
    # True class: cat (index 0)
    y_true = 0
    
    # Different predictions
    predictions = [
        np.array([0.9, 0.05, 0.03, 0.02]),   # Confident correct
        np.array([0.5, 0.3, 0.1, 0.1]),      # Less confident
        np.array([0.25, 0.25, 0.25, 0.25]),  # Uniform
        np.array([0.1, 0.7, 0.1, 0.1]),      # Confident wrong
    ]
    
    print(f"True class: {classes[y_true]} (index {y_true})")
    print("\nCE = -log(P(true class))")
    
    print(f"\n{'Prediction':>35} {'CE':>10}")
    print("-" * 50)
    
    for pred in predictions:
        pred_str = "[" + ", ".join(f"{p:.2f}" for p in pred) + "]"
        ce = cross_entropy_loss(y_true, pred)
        print(f"{pred_str:>35} {ce:>10.4f}")
    
    print(f"\nBaseline (random 4-class): -log(0.25) = {-np.log(0.25):.4f}")


def example_softmax_cross_entropy():
    """Combined softmax and cross-entropy."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Softmax + Cross-Entropy")
    print("=" * 60)
    
    def softmax_cross_entropy_naive(logits, y_true_idx):
        """Naive implementation (can be unstable)."""
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return -np.log(probs[y_true_idx])
    
    def softmax_cross_entropy_stable(logits, y_true_idx):
        """Numerically stable implementation."""
        # log(softmax(z)) = z - log(sum(exp(z)))
        # = z - logsumexp(z)
        log_probs = logits - logsumexp(logits)
        return -log_probs[y_true_idx]
    
    # Example with large logits
    logits = np.array([10.0, 5.0, 1.0])
    y_true = 0
    
    print(f"Logits: {logits}")
    print(f"True class: {y_true}")
    
    probs = softmax(logits)
    print(f"Softmax probabilities: {probs}")
    
    ce_naive = softmax_cross_entropy_naive(logits, y_true)
    ce_stable = softmax_cross_entropy_stable(logits, y_true)
    
    print(f"\nNaive CE: {ce_naive:.6f}")
    print(f"Stable CE: {ce_stable:.6f}")
    
    # Show gradient
    print("\nGradient of CE w.r.t. logits:")
    grad = probs.copy()
    grad[y_true] -= 1
    print(f"∂L/∂z = softmax(z) - one_hot(y) = {grad}")
    print("Simple! Just (predicted - actual)")


def example_mle_connection():
    """Cross-entropy as negative log-likelihood."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Cross-Entropy = Negative Log-Likelihood")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulated dataset
    n_samples = 5
    n_classes = 3
    
    # True labels
    y_true = np.array([0, 1, 2, 0, 1])
    
    # Model predictions (probabilities)
    y_pred = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1],
        [0.3, 0.5, 0.2]
    ])
    
    print("Dataset with", n_samples, "samples,", n_classes, "classes")
    print(f"True labels: {y_true}")
    print(f"Predictions:\n{y_pred}")
    
    # Likelihood
    likelihood = 1
    log_likelihood = 0
    
    for i in range(n_samples):
        p = y_pred[i, y_true[i]]
        likelihood *= p
        log_likelihood += np.log(p)
    
    # Cross-entropy
    ce_total = 0
    for i in range(n_samples):
        ce_total -= np.log(y_pred[i, y_true[i]])
    ce_avg = ce_total / n_samples
    
    print(f"\nLikelihood = ∏ P(y_i|x_i) = {likelihood:.6f}")
    print(f"Log-likelihood = Σ log P(y_i|x_i) = {log_likelihood:.4f}")
    print(f"Negative log-likelihood = {-log_likelihood:.4f}")
    print(f"Average CE loss = {ce_avg:.4f}")
    print(f"\nNLL / n = {-log_likelihood/n_samples:.4f} = CE ✓")


def example_gradient_computation():
    """Gradient of cross-entropy loss."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Cross-Entropy Gradient")
    print("=" * 60)
    
    def forward(logits, y_true_idx):
        """Forward pass: compute loss."""
        probs = softmax(logits)
        loss = -np.log(probs[y_true_idx])
        return loss, probs
    
    def backward(probs, y_true_idx):
        """Backward pass: compute gradient."""
        grad = probs.copy()
        grad[y_true_idx] -= 1
        return grad
    
    logits = np.array([2.0, 1.0, 0.1])
    y_true = 0
    
    loss, probs = forward(logits, y_true)
    grad = backward(probs, y_true)
    
    print(f"Logits: {logits}")
    print(f"True class: {y_true}")
    print(f"Softmax probs: {probs}")
    print(f"Loss: {loss:.4f}")
    print(f"\nGradient: {grad}")
    
    # Numerical verification
    eps = 1e-5
    numerical_grad = np.zeros_like(logits)
    
    for i in range(len(logits)):
        logits_plus = logits.copy()
        logits_plus[i] += eps
        loss_plus, _ = forward(logits_plus, y_true)
        
        logits_minus = logits.copy()
        logits_minus[i] -= eps
        loss_minus, _ = forward(logits_minus, y_true)
        
        numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)
    
    print(f"Numerical gradient: {numerical_grad}")
    print(f"Difference: {np.abs(grad - numerical_grad).max():.2e}")


def example_label_smoothing():
    """Label smoothing for cross-entropy."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Label Smoothing")
    print("=" * 60)
    
    def cross_entropy(y_true, logits):
        """Standard cross-entropy."""
        probs = softmax(logits)
        return -np.sum(y_true * np.log(probs + 1e-10))
    
    def smooth_labels(y_one_hot, epsilon, n_classes):
        """Apply label smoothing."""
        return (1 - epsilon) * y_one_hot + epsilon / n_classes
    
    n_classes = 5
    logits = np.array([3.0, 1.0, 0.5, 0.2, 0.1])
    y_true_idx = 0
    
    # One-hot encoding
    y_one_hot = np.zeros(n_classes)
    y_one_hot[y_true_idx] = 1
    
    print(f"Logits: {logits}")
    print(f"Softmax: {softmax(logits).round(3)}")
    print(f"True class: {y_true_idx}")
    
    print(f"\nLabel smoothing comparison:")
    print(f"{'ε':>6} {'Labels':>30} {'CE Loss':>12}")
    print("-" * 55)
    
    for epsilon in [0.0, 0.1, 0.2, 0.3]:
        y_smooth = smooth_labels(y_one_hot, epsilon, n_classes)
        ce = cross_entropy(y_smooth, logits)
        label_str = "[" + ", ".join(f"{l:.2f}" for l in y_smooth) + "]"
        print(f"{epsilon:>6.1f} {label_str:>30} {ce:>12.4f}")
    
    print("\nLabel smoothing prevents overconfident predictions")
    print("and provides implicit regularization")


def example_focal_loss():
    """Focal loss for imbalanced classification."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Focal Loss")
    print("=" * 60)
    
    def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
        """
        Focal Loss = -α(1-p)^γ log(p) for positive class
        """
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # For positive class
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1 - pt) ** gamma
        
        ce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return alpha * focal_weight * ce
    
    def binary_ce(y_true, y_pred):
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    print("Focal Loss: -α(1-p_t)^γ log(p_t)")
    print("Down-weights easy examples, focuses on hard ones\n")
    
    y_true = 1  # Positive class
    
    print(f"{'y_pred':>10} {'BCE':>10} {'Focal (γ=2)':>15}")
    print("-" * 40)
    
    for y_pred in [0.9, 0.7, 0.5, 0.3, 0.1]:
        bce = binary_ce(y_true, y_pred)
        fl = focal_loss(y_true, y_pred)
        print(f"{y_pred:>10.1f} {bce:>10.4f} {fl:>15.4f}")
    
    print("\nFocal loss is much smaller for easy examples (high p)")
    print("Forces model to focus on hard examples")


def example_knowledge_distillation():
    """Knowledge distillation loss."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Knowledge Distillation")
    print("=" * 60)
    
    def softmax_temperature(logits, T=1.0):
        """Softmax with temperature."""
        return softmax(logits / T)
    
    def distillation_loss(student_logits, teacher_logits, y_true, 
                          alpha=0.5, T=4.0):
        """
        Combined loss for knowledge distillation.
        """
        # Hard loss (with ground truth)
        student_probs = softmax(student_logits)
        hard_loss = -np.log(student_probs[y_true])
        
        # Soft loss (with teacher)
        student_soft = softmax_temperature(student_logits, T)
        teacher_soft = softmax_temperature(teacher_logits, T)
        soft_loss = -np.sum(teacher_soft * np.log(student_soft + 1e-10))
        
        # Combined (T^2 scaling for soft loss)
        total_loss = alpha * hard_loss + (1 - alpha) * T**2 * soft_loss
        
        return total_loss, hard_loss, soft_loss
    
    # Teacher: confident predictions
    teacher_logits = np.array([5.0, 2.0, 1.0, 0.5])
    
    # Student: learning
    student_logits = np.array([2.0, 1.5, 1.0, 0.8])
    
    y_true = 0  # True class
    
    print(f"Teacher logits: {teacher_logits}")
    print(f"Student logits: {student_logits}")
    print(f"True class: {y_true}")
    
    print(f"\n{'T':>6} {'Teacher soft':>30} {'Student soft':>30}")
    print("-" * 70)
    
    for T in [1.0, 2.0, 4.0, 10.0]:
        t_soft = softmax_temperature(teacher_logits, T)
        s_soft = softmax_temperature(student_logits, T)
        t_str = "[" + ", ".join(f"{p:.2f}" for p in t_soft) + "]"
        s_str = "[" + ", ".join(f"{p:.2f}" for p in s_soft) + "]"
        print(f"{T:>6.1f} {t_str:>30} {s_str:>30}")
    
    print("\nHigher temperature → softer distributions → more knowledge transfer")
    
    # Loss comparison
    print(f"\n{'T':>6} {'Total':>10} {'Hard':>10} {'Soft':>10}")
    print("-" * 40)
    
    for T in [1.0, 2.0, 4.0]:
        total, hard, soft = distillation_loss(student_logits, teacher_logits, 
                                               y_true, alpha=0.5, T=T)
        print(f"{T:>6.1f} {total:>10.4f} {hard:>10.4f} {soft:>10.4f}")


def example_sequence_model_ce():
    """Cross-entropy for sequence models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Sequence Model Cross-Entropy")
    print("=" * 60)
    
    # Vocabulary
    vocab = ['<pad>', 'the', 'cat', 'sat', 'on', 'mat', '<eos>']
    vocab_size = len(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    # Sequence: "the cat sat"
    sequence = ['the', 'cat', 'sat', '<eos>']
    seq_idx = [word_to_idx[w] for w in sequence]
    
    # Simulated model predictions (logits for each position)
    np.random.seed(42)
    # Shape: (seq_len, vocab_size)
    logits = np.random.randn(len(sequence), vocab_size)
    # Make model somewhat correct
    for i, idx in enumerate(seq_idx):
        logits[i, idx] += 3.0  # Boost correct token
    
    print(f"Vocabulary: {vocab}")
    print(f"Sequence: {sequence}")
    
    total_loss = 0
    print(f"\n{'Position':>10} {'True':>10} {'P(true)':>10} {'Loss':>10}")
    print("-" * 45)
    
    for t, (word, true_idx) in enumerate(zip(sequence, seq_idx)):
        probs = softmax(logits[t])
        loss_t = -np.log(probs[true_idx])
        total_loss += loss_t
        print(f"{t:>10} {word:>10} {probs[true_idx]:>10.3f} {loss_t:>10.4f}")
    
    avg_loss = total_loss / len(sequence)
    perplexity = np.exp(avg_loss)
    
    print(f"\nTotal loss: {total_loss:.4f}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("\nPerplexity interpretation: model is as uncertain as")
    print(f"choosing uniformly among {perplexity:.1f} words")


def example_multilabel_bce():
    """Multi-label binary cross-entropy."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Multi-Label Classification")
    print("=" * 60)
    
    def multilabel_bce(y_true, y_pred):
        """BCE for each label independently."""
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        bce_per_label = -(y_true * np.log(y_pred) + 
                         (1 - y_true) * np.log(1 - y_pred))
        return bce_per_label.sum(), bce_per_label
    
    # Labels: [happy, sad, angry, surprised]
    labels = ['happy', 'sad', 'angry', 'surprised']
    
    # Image can have multiple emotions
    y_true = np.array([1, 0, 0, 1])  # Happy and surprised
    y_pred = np.array([0.8, 0.2, 0.1, 0.7])
    
    print("Multi-label: each label is independent binary classification")
    print(f"Labels: {labels}")
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    
    total_loss, losses = multilabel_bce(y_true, y_pred)
    
    print(f"\n{'Label':>12} {'y':>5} {'ŷ':>8} {'BCE':>10}")
    print("-" * 40)
    
    for i, label in enumerate(labels):
        print(f"{label:>12} {y_true[i]:>5} {y_pred[i]:>8.2f} {losses[i]:>10.4f}")
    
    print(f"\nTotal loss: {total_loss:.4f}")
    print(f"Average per label: {total_loss/len(labels):.4f}")


def example_ce_vs_mse():
    """Comparing cross-entropy and MSE for classification."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Cross-Entropy vs MSE")
    print("=" * 60)
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def bce_loss(y, z):
        """BCE with logit input."""
        p = sigmoid(z)
        return -y * np.log(p + 1e-10) - (1 - y) * np.log(1 - p + 1e-10)
    
    def mse_loss(y, z):
        """MSE with sigmoid output."""
        p = sigmoid(z)
        return (y - p) ** 2
    
    def bce_gradient(y, z):
        """Gradient of BCE w.r.t. z."""
        return sigmoid(z) - y
    
    def mse_gradient(y, z):
        """Gradient of MSE w.r.t. z."""
        p = sigmoid(z)
        return 2 * (p - y) * p * (1 - p)
    
    y = 1  # True label
    
    print("Comparing BCE and MSE for binary classification")
    print(f"True label: {y}")
    
    print(f"\n{'z':>8} {'p=σ(z)':>10} {'BCE':>10} {'MSE':>10} {'∂BCE/∂z':>12} {'∂MSE/∂z':>12}")
    print("-" * 70)
    
    for z in [-5, -3, -1, 0, 1, 3, 5]:
        p = sigmoid(z)
        bce = bce_loss(y, z)
        mse = mse_loss(y, z)
        grad_bce = bce_gradient(y, z)
        grad_mse = mse_gradient(y, z)
        
        print(f"{z:>8.1f} {p:>10.4f} {bce:>10.4f} {mse:>10.4f} "
              f"{grad_bce:>12.4f} {grad_mse:>12.4f}")
    
    print("\nKey observation:")
    print("- BCE gradient is linear in error: p - y")
    print("- MSE gradient vanishes when p ≈ 0 or p ≈ 1")
    print("- BCE provides stronger gradients for wrong predictions")


if __name__ == "__main__":
    example_basic_cross_entropy()
    example_binary_cross_entropy()
    example_multiclass_cross_entropy()
    example_softmax_cross_entropy()
    example_mle_connection()
    example_gradient_computation()
    example_label_smoothing()
    example_focal_loss()
    example_knowledge_distillation()
    example_sequence_model_ce()
    example_multilabel_bce()
    example_ce_vs_mse()
