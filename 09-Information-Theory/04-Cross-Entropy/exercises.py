"""
Cross-Entropy - Exercises
=========================
Practice problems for cross-entropy.
"""

import numpy as np
from scipy.special import softmax


class CrossEntropyExercises:
    """Exercises for cross-entropy."""
    
    def exercise_1_basic_computation(self):
        """
        Exercise 1: Compute Cross-Entropy
        
        Calculate cross-entropy and verify decomposition.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Basic Cross-Entropy")
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
        
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        
        H_p = entropy(p)
        H_pq = cross_entropy(p, q)
        D_kl = kl_divergence(p, q)
        
        print(f"P = {p}")
        print(f"Q = {q}")
        print(f"\nH(P) = {H_p:.4f}")
        print(f"H(P, Q) = {H_pq:.4f}")
        print(f"D_KL(P || Q) = {D_kl:.4f}")
        
        print(f"\nVerification: H(P,Q) = H(P) + D_KL(P||Q)")
        print(f"  {H_pq:.4f} = {H_p:.4f} + {D_kl:.4f} = {H_p + D_kl:.4f} ✓")
    
    def exercise_2_binary_ce(self):
        """
        Exercise 2: Binary Cross-Entropy
        
        Compute BCE for different scenarios.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Binary Cross-Entropy")
        print("=" * 60)
        
        def bce(y, y_hat):
            eps = 1e-10
            return -(y * np.log(y_hat + eps) + (1-y) * np.log(1-y_hat + eps))
        
        print("BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]")
        
        print(f"\n{'Scenario':>25} {'y':>5} {'ŷ':>8} {'BCE':>10}")
        print("-" * 55)
        
        scenarios = [
            ("Perfect positive", 1, 1.0),
            ("Perfect negative", 0, 0.0),
            ("Good positive", 1, 0.9),
            ("Good negative", 0, 0.1),
            ("Random guess", 1, 0.5),
            ("Wrong positive", 1, 0.1),
            ("Wrong negative", 0, 0.9),
        ]
        
        for name, y, y_hat in scenarios:
            loss = bce(y, y_hat)
            print(f"{name:>25} {y:>5} {y_hat:>8.1f} {loss:>10.4f}")
        
        print(f"\nRandom baseline: BCE(y, 0.5) = {bce(1, 0.5):.4f}")
    
    def exercise_3_multiclass_ce(self):
        """
        Exercise 3: Multi-Class Cross-Entropy
        
        Compute CE loss for classification.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Multi-Class Cross-Entropy")
        print("=" * 60)
        
        def ce_loss(y_true_idx, probs):
            """CE = -log(p_true)"""
            return -np.log(probs[y_true_idx] + 1e-10)
        
        # 3-class problem
        y_true = 1  # True class is 1
        
        predictions = [
            np.array([0.1, 0.8, 0.1]),  # Confident correct
            np.array([0.2, 0.6, 0.2]),  # Less confident
            np.array([0.33, 0.34, 0.33]),  # Near uniform
            np.array([0.7, 0.2, 0.1]),  # Confident wrong
        ]
        
        print(f"True class: {y_true}")
        print(f"\n{'Prediction':>30} {'P(true)':>10} {'CE':>10}")
        print("-" * 55)
        
        for pred in predictions:
            pred_str = "[" + ", ".join(f"{p:.2f}" for p in pred) + "]"
            loss = ce_loss(y_true, pred)
            print(f"{pred_str:>30} {pred[y_true]:>10.2f} {loss:>10.4f}")
        
        print(f"\nRandom baseline (3 classes): -log(1/3) = {-np.log(1/3):.4f}")
    
    def exercise_4_softmax_ce(self):
        """
        Exercise 4: Softmax + Cross-Entropy
        
        Compute combined softmax-CE and gradients.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Softmax + Cross-Entropy")
        print("=" * 60)
        
        def softmax_ce(logits, y_true):
            probs = softmax(logits)
            loss = -np.log(probs[y_true])
            return loss, probs
        
        def gradient(probs, y_true):
            grad = probs.copy()
            grad[y_true] -= 1
            return grad
        
        logits = np.array([2.0, 1.0, 0.5, 0.0])
        y_true = 0
        
        loss, probs = softmax_ce(logits, y_true)
        grad = gradient(probs, y_true)
        
        print(f"Logits z = {logits}")
        print(f"True class = {y_true}")
        print(f"\nSoftmax(z) = {probs.round(4)}")
        print(f"Loss = -log(p_{y_true}) = {loss:.4f}")
        
        print(f"\nGradient ∂L/∂z = softmax(z) - one_hot(y)")
        print(f"           = {probs.round(4)} - {np.eye(len(logits))[y_true]}")
        print(f"           = {grad.round(4)}")
        
        print("\nInterpretation:")
        print("  - Positive gradient pushes logit down")
        print("  - Negative gradient pushes logit up")
        print(f"  - True class has gradient {grad[y_true]:.4f} (push up)")
    
    def exercise_5_mle_equivalence(self):
        """
        Exercise 5: CE as Negative Log-Likelihood
        
        Show equivalence between CE minimization and MLE.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: CE = Negative Log-Likelihood")
        print("=" * 60)
        
        # Small dataset
        y_true = np.array([0, 1, 1, 0, 2])
        y_pred = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.6, 0.3, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        # Likelihood
        likelihood = 1
        for i in range(len(y_true)):
            likelihood *= y_pred[i, y_true[i]]
        
        # Log-likelihood
        log_likelihood = sum(np.log(y_pred[i, y_true[i]]) 
                            for i in range(len(y_true)))
        
        # Cross-entropy
        ce_total = sum(-np.log(y_pred[i, y_true[i]]) 
                       for i in range(len(y_true)))
        
        print("Dataset:")
        for i in range(len(y_true)):
            print(f"  Sample {i}: y={y_true[i]}, pred={y_pred[i].round(2)}")
        
        print(f"\nLikelihood = ∏P(y_i|x_i) = {likelihood:.6f}")
        print(f"Log-likelihood = Σlog P(y_i|x_i) = {log_likelihood:.4f}")
        print(f"CE total = -Σlog P(y_i|x_i) = {ce_total:.4f}")
        print(f"\nCE = -log-likelihood ✓")
        print("\nMinimizing CE ⟺ Maximizing likelihood")
    
    def exercise_6_label_smoothing(self):
        """
        Exercise 6: Label Smoothing
        
        Apply label smoothing and observe effects.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Label Smoothing")
        print("=" * 60)
        
        def smooth_labels(y_one_hot, epsilon, K):
            return (1 - epsilon) * y_one_hot + epsilon / K
        
        def ce_loss(y, logits):
            probs = softmax(logits)
            return -np.sum(y * np.log(probs + 1e-10))
        
        K = 4  # 4 classes
        y_true_idx = 0
        y_one_hot = np.eye(K)[y_true_idx]
        
        # Very confident logits
        logits = np.array([10.0, 1.0, 0.5, 0.1])
        
        print(f"Logits: {logits}")
        print(f"Softmax: {softmax(logits).round(4)}")
        print(f"True class: {y_true_idx}")
        
        print(f"\n{'ε':>6} {'Smoothed labels':>25} {'CE Loss':>12}")
        print("-" * 50)
        
        for epsilon in [0.0, 0.05, 0.1, 0.2]:
            y_smooth = smooth_labels(y_one_hot, epsilon, K)
            loss = ce_loss(y_smooth, logits)
            label_str = "[" + ", ".join(f"{l:.2f}" for l in y_smooth) + "]"
            print(f"{epsilon:>6.2f} {label_str:>25} {loss:>12.4f}")
        
        print("\nLabel smoothing:")
        print("  - Prevents overconfident predictions")
        print("  - Encourages model to be less certain")
        print("  - Acts as regularization")
    
    def exercise_7_focal_loss(self):
        """
        Exercise 7: Focal Loss
        
        Compute focal loss for imbalanced data.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Focal Loss")
        print("=" * 60)
        
        def ce_loss(y, p):
            return -y * np.log(p + 1e-10) - (1-y) * np.log(1-p + 1e-10)
        
        def focal_loss(y, p, gamma=2.0, alpha=1.0):
            pt = y * p + (1 - y) * (1 - p)  # p if y=1, else 1-p
            focal_weight = (1 - pt) ** gamma
            ce = ce_loss(y, p)
            return alpha * focal_weight * ce
        
        print("Focal Loss: FL = -α(1-p_t)^γ log(p_t)")
        print("where p_t = p if y=1 else 1-p")
        
        y = 1
        print(f"\nTrue label: {y}")
        
        print(f"\n{'p':>8} {'CE':>10} {'FL(γ=0)':>12} {'FL(γ=2)':>12} {'FL(γ=5)':>12}")
        print("-" * 60)
        
        for p in [0.9, 0.7, 0.5, 0.3, 0.1]:
            ce = ce_loss(y, p)
            fl0 = focal_loss(y, p, gamma=0)
            fl2 = focal_loss(y, p, gamma=2)
            fl5 = focal_loss(y, p, gamma=5)
            print(f"{p:>8.1f} {ce:>10.4f} {fl0:>12.4f} {fl2:>12.4f} {fl5:>12.4f}")
        
        print("\nγ=0: reduces to standard CE")
        print("Higher γ: more focus on hard examples (low p)")
    
    def exercise_8_perplexity(self):
        """
        Exercise 8: Perplexity
        
        Compute perplexity for language modeling.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Perplexity")
        print("=" * 60)
        
        # Sequence with probabilities
        # Model predictions for next token
        sequence_probs = [0.3, 0.5, 0.2, 0.8, 0.6]  # P(correct token)
        
        print("Perplexity = exp(average CE)")
        print("           = exp(-1/T Σ log p(x_t))")
        
        print(f"\nSequence probabilities: {sequence_probs}")
        
        T = len(sequence_probs)
        avg_nll = -sum(np.log(p) for p in sequence_probs) / T
        perplexity = np.exp(avg_nll)
        
        print(f"\nStep-by-step:")
        for t, p in enumerate(sequence_probs):
            nll = -np.log(p)
            print(f"  t={t}: p={p:.1f}, -log(p)={nll:.4f}")
        
        print(f"\nAverage NLL = {avg_nll:.4f}")
        print(f"Perplexity = exp({avg_nll:.4f}) = {perplexity:.2f}")
        
        print(f"\nInterpretation: Model is as uncertain as choosing")
        print(f"uniformly among {perplexity:.1f} options at each step")
        
        # Compare with random
        random_ppl = np.exp(-np.log(1/10))  # Assuming vocab size 10
        print(f"\nRandom baseline (vocab=10): PPL = {random_ppl:.1f}")
    
    def exercise_9_weighted_ce(self):
        """
        Exercise 9: Weighted Cross-Entropy
        
        Handle class imbalance with weights.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Weighted Cross-Entropy")
        print("=" * 60)
        
        # Imbalanced dataset
        # Class 0: 900 samples (90%)
        # Class 1: 100 samples (10%)
        
        class_counts = np.array([900, 100])
        total = class_counts.sum()
        
        # Inverse frequency weights
        weights_inv_freq = total / (len(class_counts) * class_counts)
        
        # Effective number of samples
        beta = 0.9999
        effective_num = 1 - np.power(beta, class_counts)
        weights_effective = (1 - beta) / effective_num
        weights_effective = weights_effective / weights_effective.sum() * len(class_counts)
        
        print(f"Class counts: {class_counts}")
        print(f"Class frequencies: {class_counts/total}")
        
        print(f"\nWeight strategies:")
        print(f"  No weighting: [1.0, 1.0]")
        print(f"  Inverse frequency: {weights_inv_freq.round(3)}")
        print(f"  Effective samples: {weights_effective.round(3)}")
        
        # Effect on loss
        def weighted_ce(y, p, weight):
            return -weight * np.log(p + 1e-10)
        
        p = 0.3  # Predicted probability for minority class (class 1)
        
        print(f"\nLoss for minority class (y=1) with p={p}:")
        print(f"  Unweighted: {weighted_ce(1, p, 1.0):.4f}")
        print(f"  Inv freq:   {weighted_ce(1, p, weights_inv_freq[1]):.4f}")
        print(f"  Effective:  {weighted_ce(1, p, weights_effective[1]):.4f}")
    
    def exercise_10_temperature_scaling(self):
        """
        Exercise 10: Temperature Scaling
        
        Calibrate probabilities with temperature.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Temperature Scaling")
        print("=" * 60)
        
        def softmax_temp(logits, T):
            return softmax(logits / T)
        
        logits = np.array([3.0, 1.0, 0.5])
        
        print("Temperature scaling: softmax(z/T)")
        print(f"Logits: {logits}")
        
        print(f"\n{'T':>6} {'Softmax(z/T)':>30} {'Max prob':>12} {'Entropy':>10}")
        print("-" * 65)
        
        for T in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            probs = softmax_temp(logits, T)
            max_p = probs.max()
            H = -np.sum(probs * np.log(probs + 1e-10))
            
            prob_str = "[" + ", ".join(f"{p:.3f}" for p in probs) + "]"
            print(f"{T:>6.1f} {prob_str:>30} {max_p:>12.4f} {H:>10.4f}")
        
        print("\nT < 1: Sharper (more confident)")
        print("T > 1: Softer (less confident)")
        print("T → ∞: Uniform distribution")
        print("T → 0: One-hot (argmax)")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = CrossEntropyExercises()
    
    print("CROSS-ENTROPY EXERCISES")
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
