"""
Floating-Point Arithmetic - Exercises
=====================================
Practice problems for numerical computing.
"""

import numpy as np


def exercise_1_machine_epsilon():
    """
    EXERCISE 1: Compute Machine Epsilon
    ===================================
    
    Write a function to compute machine epsilon without using np.finfo.
    
    Machine epsilon is the smallest ε such that 1 + ε > 1
    in floating-point arithmetic.
    
    Tasks:
    a) Implement compute_epsilon(dtype)
    b) Verify for float16, float32, float64
    c) Explain why this works
    """
    print("=" * 60)
    print("EXERCISE 1: Compute Machine Epsilon")
    print("=" * 60)
    
    # YOUR CODE HERE
    def compute_epsilon(dtype):
        """Compute machine epsilon for given dtype."""
        # TODO: Start with eps = 1.0
        # Keep halving until 1 + eps == 1
        # Return the last eps where 1 + eps > 1
        pass
    
    # Test
    # for dtype in [np.float16, np.float32, np.float64]:
    #     eps = compute_epsilon(dtype)
    #     print(f"{dtype.__name__}: {eps}")


def exercise_1_solution():
    """Solution for Exercise 1."""
    print("=" * 60)
    print("SOLUTION 1: Machine Epsilon")
    print("=" * 60)
    
    def compute_epsilon(dtype):
        eps = dtype(1.0)
        while dtype(1.0) + eps > dtype(1.0):
            prev = eps
            eps = eps / dtype(2.0)
        return prev
    
    print(f"{'Type':>12} {'Computed':>20} {'np.finfo':>20}")
    print("-" * 55)
    
    for dtype in [np.float16, np.float32, np.float64]:
        computed = compute_epsilon(dtype)
        expected = np.finfo(dtype).eps
        print(f"{dtype.__name__:>12} {float(computed):>20.2e} {expected:>20.2e}")
    
    print("\nExplanation:")
    print("  We repeatedly halve eps until 1 + eps rounds to 1")
    print("  The last eps that made a difference is machine epsilon")


def exercise_2_stable_variance():
    """
    EXERCISE 2: Numerically Stable Variance
    =======================================
    
    The one-pass variance formula is unstable:
    Var(X) = E[X²] - E[X]²
    
    Implement Welford's online algorithm for stable variance.
    
    Tasks:
    a) Implement welford_variance(data)
    b) Compare with naive one-pass
    c) Test with data that causes issues
    """
    print("\n" + "=" * 60)
    print("EXERCISE 2: Stable Variance")
    print("=" * 60)
    
    # YOUR CODE HERE
    def variance_one_pass(data):
        """Unstable one-pass variance."""
        n = len(data)
        sum_x = sum(data)
        sum_x2 = sum(x**2 for x in data)
        return sum_x2/n - (sum_x/n)**2
    
    def welford_variance(data):
        """
        Welford's online algorithm for variance.
        
        For each new value x:
        - delta = x - mean
        - mean += delta / n
        - M2 += delta * (x - mean)
        
        Variance = M2 / n
        """
        # TODO: Implement
        pass
    
    # Test with problematic data
    # data = [1e9 + i for i in range(100)]


def exercise_2_solution():
    """Solution for Exercise 2."""
    print("\n" + "=" * 60)
    print("SOLUTION 2: Stable Variance")
    print("=" * 60)
    
    def variance_one_pass(data):
        n = len(data)
        sum_x = sum(data)
        sum_x2 = sum(x**2 for x in data)
        return sum_x2/n - (sum_x/n)**2
    
    def welford_variance(data):
        n = 0
        mean = 0.0
        M2 = 0.0
        
        for x in data:
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2
        
        return M2 / n if n > 0 else 0.0
    
    # Test with normal data
    data_normal = [1, 2, 3, 4, 5]
    print("Normal data [1,2,3,4,5]:")
    print(f"  One-pass: {variance_one_pass(data_normal):.6f}")
    print(f"  Welford:  {welford_variance(data_normal):.6f}")
    print(f"  NumPy:    {np.var(data_normal):.6f}")
    
    # Test with problematic data
    data_large = [1e9 + i for i in range(100)]
    print(f"\nLarge data [1e9, 1e9+1, ..., 1e9+99]:")
    print(f"  True variance: {np.var(data_large):.6f}")
    print(f"  One-pass: {variance_one_pass(data_large):.6f}")
    print(f"  Welford:  {welford_variance(data_large):.6f}")
    
    print("\nNote: One-pass may give negative or wrong values!")


def exercise_3_stable_softmax():
    """
    EXERCISE 3: Implement Stable Softmax
    ====================================
    
    Implement a numerically stable softmax that:
    1. Handles large positive inputs
    2. Handles large negative inputs
    3. Handles batch inputs
    
    Tasks:
    a) Implement stable_softmax(x, axis=-1)
    b) Implement stable_log_softmax(x, axis=-1)
    c) Test edge cases
    """
    print("\n" + "=" * 60)
    print("EXERCISE 3: Stable Softmax")
    print("=" * 60)
    
    # YOUR CODE HERE
    def stable_softmax(x, axis=-1):
        """Numerically stable softmax."""
        # TODO: Subtract max before exp
        pass
    
    def stable_log_softmax(x, axis=-1):
        """Numerically stable log-softmax."""
        # TODO: x - max - log(sum(exp(x - max)))
        pass


def exercise_3_solution():
    """Solution for Exercise 3."""
    print("\n" + "=" * 60)
    print("SOLUTION 3: Stable Softmax")
    print("=" * 60)
    
    def stable_softmax(x, axis=-1):
        x = np.asarray(x)
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def stable_log_softmax(x, axis=-1):
        x = np.asarray(x)
        x_max = np.max(x, axis=axis, keepdims=True)
        log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
        return x - log_sum_exp
    
    # Test cases
    print("Test 1: Normal values")
    x = np.array([1.0, 2.0, 3.0])
    print(f"  Input: {x}")
    print(f"  Softmax: {stable_softmax(x)}")
    print(f"  Log-softmax: {stable_log_softmax(x)}")
    
    print("\nTest 2: Large positive values")
    x = np.array([1000.0, 1001.0, 1002.0])
    print(f"  Input: {x}")
    print(f"  Softmax: {stable_softmax(x)}")
    
    print("\nTest 3: Large negative values")
    x = np.array([-1000.0, -999.0, -998.0])
    print(f"  Input: {x}")
    print(f"  Softmax: {stable_softmax(x)}")
    
    print("\nTest 4: Batch input")
    x = np.array([[1, 2, 3], [1000, 1001, 1002]])
    print(f"  Input:\n{x}")
    print(f"  Softmax:\n{stable_softmax(x, axis=1)}")


def exercise_4_cross_entropy():
    """
    EXERCISE 4: Stable Cross-Entropy Loss
    =====================================
    
    Implement cross-entropy loss that:
    1. Takes logits (pre-softmax) as input
    2. Avoids log(0)
    3. Uses log-sum-exp trick
    
    Formula: L = -sum(y * log(softmax(z)))
            = -sum(y * (z - logsumexp(z)))
    
    Tasks:
    a) Implement cross_entropy_from_logits(logits, targets)
    b) Handle one-hot and class index targets
    c) Test numerical stability
    """
    print("\n" + "=" * 60)
    print("EXERCISE 4: Stable Cross-Entropy")
    print("=" * 60)
    
    # YOUR CODE HERE
    def cross_entropy_from_logits(logits, targets):
        """
        Compute cross-entropy loss from logits.
        
        Args:
            logits: (batch_size, num_classes) - raw scores
            targets: (batch_size,) class indices or (batch_size, num_classes) one-hot
        """
        # TODO: Implement using logsumexp trick
        pass


def exercise_4_solution():
    """Solution for Exercise 4."""
    print("\n" + "=" * 60)
    print("SOLUTION 4: Stable Cross-Entropy")
    print("=" * 60)
    
    def logsumexp(x, axis=-1, keepdims=False):
        x_max = np.max(x, axis=axis, keepdims=True)
        result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    
    def cross_entropy_from_logits(logits, targets):
        logits = np.asarray(logits)
        targets = np.asarray(targets)
        
        # Convert class indices to one-hot if needed
        if targets.ndim == 1:
            num_classes = logits.shape[-1]
            one_hot = np.zeros_like(logits)
            one_hot[np.arange(len(targets)), targets] = 1
            targets = one_hot
        
        # log_softmax = logits - logsumexp(logits)
        log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
        
        # Cross-entropy = -sum(targets * log_probs)
        return -np.sum(targets * log_probs, axis=-1).mean()
    
    # Test
    print("Test 1: Normal logits")
    logits = np.array([[1.0, 2.0, 3.0],
                       [3.0, 2.0, 1.0]])
    targets = np.array([2, 0])  # Class indices
    loss = cross_entropy_from_logits(logits, targets)
    print(f"  Logits:\n{logits}")
    print(f"  Targets: {targets}")
    print(f"  Loss: {loss:.4f}")
    
    print("\nTest 2: Large logits")
    logits = np.array([[1000.0, 1001.0, 1002.0]])
    targets = np.array([2])
    loss = cross_entropy_from_logits(logits, targets)
    print(f"  Logits: {logits}")
    print(f"  Loss: {loss:.4f}")
    
    print("\nTest 3: Perfect prediction")
    logits = np.array([[0.0, 0.0, 100.0]])
    targets = np.array([2])
    loss = cross_entropy_from_logits(logits, targets)
    print(f"  Loss (should be ~0): {loss:.6f}")


def exercise_5_kahan_summation():
    """
    EXERCISE 5: Implement Kahan Summation
    =====================================
    
    Implement Kahan's compensated summation algorithm.
    
    Tasks:
    a) Implement kahan_sum(data)
    b) Compare accuracy with naive sum
    c) Implement pairwise summation as alternative
    """
    print("\n" + "=" * 60)
    print("EXERCISE 5: Kahan Summation")
    print("=" * 60)
    
    # YOUR CODE HERE
    def kahan_sum(data):
        """Kahan compensated summation."""
        # TODO: Implement
        # sum = 0, c = 0
        # for x in data:
        #   y = x - c
        #   t = sum + y
        #   c = (t - sum) - y
        #   sum = t
        pass
    
    def pairwise_sum(data):
        """Pairwise summation (divide and conquer)."""
        # TODO: Recursively sum pairs
        pass


def exercise_5_solution():
    """Solution for Exercise 5."""
    print("\n" + "=" * 60)
    print("SOLUTION 5: Kahan Summation")
    print("=" * 60)
    
    def naive_sum(data):
        total = 0.0
        for x in data:
            total += x
        return total
    
    def kahan_sum(data):
        total = 0.0
        c = 0.0
        for x in data:
            y = x - c
            t = total + y
            c = (t - total) - y
            total = t
        return total
    
    def pairwise_sum(data):
        n = len(data)
        if n == 0:
            return 0.0
        if n == 1:
            return data[0]
        mid = n // 2
        return pairwise_sum(data[:mid]) + pairwise_sum(data[mid:])
    
    # Test
    np.random.seed(42)
    n = 1000000
    data = np.ones(n, dtype=np.float32) * 0.1
    
    expected = n * 0.1
    
    print(f"Summing 0.1 × {n:,}:")
    print(f"  Expected:    {expected:.6f}")
    print(f"  Naive:       {naive_sum(data):.6f}")
    print(f"  Kahan:       {kahan_sum(data):.6f}")
    print(f"  Pairwise:    {pairwise_sum(list(data)):.6f}")
    print(f"  NumPy:       {np.sum(data):.6f}")
    
    print(f"\nErrors:")
    print(f"  Naive:    {abs(naive_sum(data) - expected):.6f}")
    print(f"  Kahan:    {abs(kahan_sum(data) - expected):.6f}")
    print(f"  Pairwise: {abs(pairwise_sum(list(data)) - expected):.6f}")


def exercise_6_condition_number():
    """
    EXERCISE 6: Analyze Condition Number
    ====================================
    
    Study how condition number affects numerical accuracy.
    
    Tasks:
    a) Create matrices with known condition numbers
    b) Solve Ax = b and measure error sensitivity
    c) Relate condition number to digit loss
    """
    print("\n" + "=" * 60)
    print("EXERCISE 6: Condition Number Analysis")
    print("=" * 60)
    
    # YOUR CODE HERE
    def create_matrix_with_condition(n, condition_number):
        """
        Create n×n matrix with specified condition number.
        
        Hint: Use SVD - A = U @ diag(s) @ V.T
        Set s[0] = condition_number, s[-1] = 1
        """
        # TODO: Implement
        pass
    
    def analyze_sensitivity(A, b):
        """
        Analyze sensitivity of Ax = b to perturbations.
        """
        # TODO: Solve with small perturbation, measure change
        pass


def exercise_6_solution():
    """Solution for Exercise 6."""
    print("\n" + "=" * 60)
    print("SOLUTION 6: Condition Number Analysis")
    print("=" * 60)
    
    def create_matrix_with_condition(n, condition_number):
        U, _ = np.linalg.qr(np.random.randn(n, n))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        
        s = np.linspace(condition_number, 1, n)
        S = np.diag(s)
        
        return U @ S @ V.T
    
    def analyze_sensitivity(A, b):
        cond = np.linalg.cond(A)
        x = np.linalg.solve(A, b)
        
        # Perturb b
        delta_b = np.random.randn(len(b)) * 1e-10 * np.linalg.norm(b)
        x_pert = np.linalg.solve(A, b + delta_b)
        
        rel_change_b = np.linalg.norm(delta_b) / np.linalg.norm(b)
        rel_change_x = np.linalg.norm(x - x_pert) / np.linalg.norm(x)
        
        return cond, rel_change_b, rel_change_x
    
    np.random.seed(42)
    n = 10
    
    print(f"{'Condition':>12} {'δb/||b||':>15} {'δx/||x||':>15} {'Amplification':>15}")
    print("-" * 60)
    
    for target_cond in [1, 10, 100, 1000, 10000, 100000]:
        A = create_matrix_with_condition(n, target_cond)
        b = np.random.randn(n)
        
        cond, rel_b, rel_x = analyze_sensitivity(A, b)
        amp = rel_x / rel_b if rel_b > 0 else 0
        
        print(f"{cond:>12.2e} {rel_b:>15.2e} {rel_x:>15.2e} {amp:>15.2f}")
    
    print("\nRule: Amplification ≤ condition number")
    print("Lost digits ≈ log₁₀(condition number)")


def exercise_7_numerical_gradient():
    """
    EXERCISE 7: Stable Numerical Gradient
    =====================================
    
    Implement numerical gradient checking with proper step size.
    
    Tasks:
    a) Implement central difference gradient
    b) Choose optimal step size
    c) Compare with analytical gradient
    """
    print("\n" + "=" * 60)
    print("EXERCISE 7: Numerical Gradient")
    print("=" * 60)
    
    # YOUR CODE HERE
    def numerical_gradient(f, x, eps=None):
        """
        Compute gradient using central differences.
        
        Optimal eps ≈ (machine_epsilon)^(1/3) for central diff
        """
        # TODO: Implement
        # For each dimension i:
        #   grad[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
        pass


def exercise_7_solution():
    """Solution for Exercise 7."""
    print("\n" + "=" * 60)
    print("SOLUTION 7: Numerical Gradient")
    print("=" * 60)
    
    def numerical_gradient(f, x, eps=None):
        if eps is None:
            # Optimal for central difference: eps ~ eps_mach^(1/3)
            eps = np.cbrt(np.finfo(float).eps)
        
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        
        return grad
    
    # Test function: f(x) = x₁² + 2x₂² + x₁x₂
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1]
    
    def analytical_grad(x):
        return np.array([2*x[0] + x[1], 4*x[1] + x[0]])
    
    x = np.array([1.0, 2.0])
    
    print(f"f(x) = x₁² + 2x₂² + x₁x₂ at x = {x}")
    print(f"Analytical gradient: {analytical_grad(x)}")
    
    print(f"\nNumerical gradient by step size:")
    for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        num_grad = numerical_gradient(f, x, eps)
        error = np.linalg.norm(num_grad - analytical_grad(x))
        print(f"  eps={eps:.0e}: grad={num_grad}, error={error:.2e}")
    
    # Optimal
    opt_eps = np.cbrt(np.finfo(float).eps)
    print(f"\nOptimal eps ≈ {opt_eps:.2e}")
    print(f"With optimal: {numerical_gradient(f, x)}")


def exercise_8_stable_norm():
    """
    EXERCISE 8: Stable Vector Norm
    ==============================
    
    Implement numerically stable L2 norm that avoids overflow.
    
    Tasks:
    a) Implement stable_norm(x)
    b) Handle very large and very small values
    c) Compare with naive implementation
    """
    print("\n" + "=" * 60)
    print("EXERCISE 8: Stable Norm")
    print("=" * 60)
    
    # YOUR CODE HERE
    def stable_norm(x):
        """
        Compute ||x||₂ without overflow.
        
        Hint: ||x||₂ = |max(|x|)| * ||x/max(|x|)||₂
        """
        # TODO: Implement
        pass


def exercise_8_solution():
    """Solution for Exercise 8."""
    print("\n" + "=" * 60)
    print("SOLUTION 8: Stable Norm")
    print("=" * 60)
    
    def naive_norm(x):
        return np.sqrt(np.sum(x ** 2))
    
    def stable_norm(x):
        x = np.asarray(x)
        if len(x) == 0:
            return 0.0
        
        max_abs = np.max(np.abs(x))
        if max_abs == 0:
            return 0.0
        
        # Scale to avoid overflow
        return max_abs * np.sqrt(np.sum((x / max_abs) ** 2))
    
    # Test normal values
    x_normal = np.array([3.0, 4.0])
    print(f"Normal [3, 4]:")
    print(f"  Naive:  {naive_norm(x_normal)}")
    print(f"  Stable: {stable_norm(x_normal)}")
    
    # Test large values
    x_large = np.array([1e200, 1e200])
    print(f"\nLarge [1e200, 1e200]:")
    with np.errstate(over='ignore'):
        print(f"  Naive:  {naive_norm(x_large)}")  # inf
    print(f"  Stable: {stable_norm(x_large):.6e}")
    print(f"  True:   {np.sqrt(2) * 1e200:.6e}")
    
    # Test small values
    x_small = np.array([1e-200, 1e-200])
    print(f"\nSmall [1e-200, 1e-200]:")
    print(f"  Naive:  {naive_norm(x_small)}")  # May underflow
    print(f"  Stable: {stable_norm(x_small):.6e}")


def exercise_9_log_probability():
    """
    EXERCISE 9: Log-Space Probability Operations
    =============================================
    
    Implement probability operations in log space to avoid underflow.
    
    Tasks:
    a) Implement log_add(log_a, log_b) = log(a + b)
    b) Implement log_sum(log_probs) = log(sum(probs))
    c) Apply to HMM forward algorithm
    """
    print("\n" + "=" * 60)
    print("EXERCISE 9: Log Probabilities")
    print("=" * 60)
    
    # YOUR CODE HERE
    def log_add(log_a, log_b):
        """Compute log(a + b) given log(a) and log(b)."""
        # TODO: Use log-sum-exp trick
        # log(a + b) = log(a) + log(1 + b/a)
        #            = log(a) + log(1 + exp(log(b) - log(a)))
        pass
    
    def log_sum(log_probs):
        """Compute log(sum(probs)) from log_probs."""
        # TODO: Implement
        pass


def exercise_9_solution():
    """Solution for Exercise 9."""
    print("\n" + "=" * 60)
    print("SOLUTION 9: Log Probabilities")
    print("=" * 60)
    
    def log_add(log_a, log_b):
        """Compute log(exp(log_a) + exp(log_b)) = log(a + b)."""
        if log_a > log_b:
            return log_a + np.log1p(np.exp(log_b - log_a))
        else:
            return log_b + np.log1p(np.exp(log_a - log_b))
    
    def log_sum(log_probs):
        """Compute log(sum(exp(log_probs)))."""
        log_probs = np.asarray(log_probs)
        max_log = np.max(log_probs)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log)))
    
    # Test log_add
    print("log_add test:")
    a, b = 0.1, 0.2
    log_a, log_b = np.log(a), np.log(b)
    result = log_add(log_a, log_b)
    print(f"  log(0.1 + 0.2) = log(0.3) = {np.log(0.3):.6f}")
    print(f"  log_add result: {result:.6f}")
    
    # Test with very small probabilities
    print("\nVery small probabilities:")
    p1, p2 = 1e-300, 2e-300
    log_p1, log_p2 = np.log(p1), np.log(p2)
    
    print(f"  p1 = {p1:.2e}, p2 = {p2:.2e}")
    print(f"  p1 + p2 = {p1 + p2:.2e}")
    print(f"  exp(log_add) = {np.exp(log_add(log_p1, log_p2)):.2e}")
    
    # Test log_sum
    print("\nlog_sum test:")
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    log_probs = np.log(probs)
    
    print(f"  sum(probs) = {np.sum(probs):.6f}")
    print(f"  exp(log_sum) = {np.exp(log_sum(log_probs)):.6f}")


def exercise_10_mixed_precision():
    """
    EXERCISE 10: Mixed Precision Simulation
    =======================================
    
    Simulate mixed precision training:
    - Master weights in float32
    - Forward/backward in float16
    - Loss scaling to prevent underflow
    
    Tasks:
    a) Implement loss scaling
    b) Simulate gradient computation
    c) Detect and handle gradient overflow/underflow
    """
    print("\n" + "=" * 60)
    print("EXERCISE 10: Mixed Precision")
    print("=" * 60)
    
    # YOUR CODE HERE
    def dynamic_loss_scale(grads, scale, scale_factor=2.0):
        """
        Dynamically adjust loss scale.
        
        - If overflow/nan in grads: reduce scale
        - If no overflow for N steps: increase scale
        """
        # TODO: Implement
        pass


def exercise_10_solution():
    """Solution for Exercise 10."""
    print("\n" + "=" * 60)
    print("SOLUTION 10: Mixed Precision")
    print("=" * 60)
    
    class MixedPrecisionTrainer:
        def __init__(self, initial_scale=65536.0):
            self.scale = initial_scale
            self.growth_interval = 100
            self.steps_since_growth = 0
        
        def has_overflow(self, grads):
            """Check for overflow/nan in gradients."""
            for g in grads:
                if np.any(~np.isfinite(g)):
                    return True
            return False
        
        def update_scale(self, grads):
            """Update loss scale based on gradient health."""
            if self.has_overflow(grads):
                self.scale /= 2.0
                self.steps_since_growth = 0
                return False  # Skip this step
            else:
                self.steps_since_growth += 1
                if self.steps_since_growth >= self.growth_interval:
                    self.scale *= 2.0
                    self.steps_since_growth = 0
                return True  # Apply gradients
        
        def train_step(self, x, y, weights, lr=0.01):
            """Simulate one training step."""
            # Convert to fp16 for forward pass
            x_fp16 = x.astype(np.float16)
            w_fp16 = weights.astype(np.float16)
            
            # Forward (fp16)
            pred = x_fp16 @ w_fp16
            
            # Loss (fp32 accumulation)
            loss = np.mean((pred.astype(np.float32) - y) ** 2)
            
            # Scaled loss
            scaled_loss = loss * self.scale
            
            # Backward (fp16) - simplified
            grad = 2 * x_fp16.T @ (pred - y.astype(np.float16)) / len(y)
            grad = grad.astype(np.float32)
            
            # Unscale gradients
            grad = grad / self.scale
            
            # Check and update scale
            if self.update_scale([grad]):
                weights -= lr * grad
            
            return loss, weights
    
    # Simulate training
    np.random.seed(42)
    
    n, d = 100, 10
    x = np.random.randn(n, d).astype(np.float32)
    true_w = np.random.randn(d, 1).astype(np.float32)
    y = (x @ true_w).astype(np.float32)
    
    trainer = MixedPrecisionTrainer()
    weights = np.zeros((d, 1), dtype=np.float32)
    
    print("Mixed Precision Training Simulation:")
    print(f"{'Step':>6} {'Loss':>12} {'Scale':>12}")
    print("-" * 35)
    
    for step in range(10):
        loss, weights = trainer.train_step(x, y, weights)
        if step % 1 == 0:
            print(f"{step:>6} {loss:>12.6f} {trainer.scale:>12.0f}")
    
    print(f"\nFinal scale: {trainer.scale}")


def run_all_exercises():
    """Run all exercise solutions."""
    exercise_1_solution()
    exercise_2_solution()
    exercise_3_solution()
    exercise_4_solution()
    exercise_5_solution()
    exercise_6_solution()
    exercise_7_solution()
    exercise_8_solution()
    exercise_9_solution()
    exercise_10_solution()


if __name__ == "__main__":
    run_all_exercises()
